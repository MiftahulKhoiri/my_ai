# train.py
"""
train.py — Entry point untuk melatih model.

Contoh pemakaian:
    python train.py
    python train.py --train_path data/train.txt --epochs 10 --device cuda
    python train.py --grad_accum_steps 4 --num_threads 4
    python train.py --tokenizer char          # pakai tokenizer level karakter (lama)
    python train.py --bpe_vocab_size 2048      # ukuran vocab BPE lebih besar
    python train.py --max_seq_len 32 --epochs 50 --eval_every 50 --checkpoint_every 100

    # --- Alur vocab tetap + lanjut training tanpa ulang dari 0 ---
    # 1) Training pertama: bangun vocab dari file referensi terpisah (kamus.txt),
    #    lalu latih model dari data/training/ seperti biasa.
    python train.py --vocab_path data/kamus.txt

    # 2) Nanti nambah file percakapan baru ke data/training/, lanjut training
    #    dari checkpoint lama. tokenizer.json sudah ada -> otomatis dipakai
    #    ulang, TIDAK di-fit ulang, jadi checkpoint lama tetap kompatibel.
    python train.py --resume_from checkpoints/final.pt

    # Paksa fit ulang vocab dari nol (HATI-HATI: checkpoint lama jadi tidak
    # kompatibel lagi kalau dipakai bareng --resume_from):
    python train.py --rebuild_tokenizer
"""

import argparse
import os
import torch

from config import Config
from data import get_tokenizer_class
from data.dataset import make_train_val_datasets, load_text
from model.transformer import TransformerLM
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Training my_ai transformer")
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--val_path", type=str, default=None)

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--max_steps", type=int, default=None,
        help="Jumlah step training eksplisit; kalau diisi, override --epochs.",
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--min_learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)

    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None)

    p.add_argument("--eval_every", type=int, default=None, help="Evaluasi tiap N step.")
    p.add_argument(
        "--eval_iters", type=int, default=None,
        help="Jumlah batch validasi yang dipakai tiap evaluasi.",
    )
    p.add_argument(
        "--checkpoint_every", type=int, default=None,
        help="Simpan checkpoint step_N.pt tiap N step.",
    )
    p.add_argument("--log_every", type=int, default=None, help="Cetak log training tiap N step.")
    p.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Folder tempat checkpoint/log/tokenizer/config disimpan "
             "(tokenizer.json otomatis ikut pindah ke folder ini).",
    )
    p.add_argument("--seed", type=int, default=None)

    p.add_argument(
        "--grad_accum_steps", type=int, default=None,
        help="Akumulasi gradien N micro-batch sebelum optimizer.step() "
             "(batch efektif = batch_size * grad_accum_steps, tanpa nambah RAM per langkah)",
    )
    p.add_argument(
        "--num_threads", type=int, default=None,
        help="Jumlah thread CPU intra-op PyTorch (mis. 4 di Raspberry Pi 5). "
             "Default: biarkan PyTorch auto-detect.",
    )
    p.add_argument(
        "--tokenizer", type=str, default=None, choices=["bpe", "char"],
        help="Jenis tokenizer: 'bpe' (default, lebih hemat data) atau 'char' (level karakter).",
    )
    p.add_argument(
        "--bpe_vocab_size", type=int, default=None,
        help="Target ukuran vocab BPE (dipakai kalau --tokenizer bpe).",
    )
    p.add_argument(
        "--vocab_path", type=str, default=None,
        help="Sumber teks untuk membangun vocab tokenizer, dipakai HANYA saat "
             "tokenizer.json belum ada (mis. data/kamus.txt sebagai acuan vocab "
             "bahasa). Kalau tidak diisi, vocab dibangun dari --train_path seperti "
             "biasa. Diabaikan kalau tokenizer.json sudah ada -- pakai "
             "--rebuild_tokenizer untuk memaksa fit ulang.",
    )
    p.add_argument(
        "--rebuild_tokenizer", action="store_true",
        help="Paksa fit ulang tokenizer dari awal meskipun tokenizer.json sudah "
             "ada. HATI-HATI: ini membuat checkpoint lama tidak kompatibel lagi "
             "kalau dipakai bareng --resume_from (vocab berubah -> ukuran "
             "embedding beda -> load_state_dict gagal).",
    )
    return p.parse_args()


def apply_overrides(config: Config, args) -> Config:
    if args.train_path:
        config.data.train_dir = args.train_path
    if args.val_path:
        config.data.val_path = args.val_path

    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.min_learning_rate is not None:
        config.training.min_learning_rate = args.min_learning_rate
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
    if args.grad_clip is not None:
        config.training.grad_clip = args.grad_clip

    if args.d_model is not None:
        config.model.d_model = args.d_model
        config.model.d_ff = 4 * args.d_model
    if args.n_layers is not None:
        config.model.n_layers = args.n_layers
    if args.n_heads is not None:
        config.model.n_heads = args.n_heads
    if args.max_seq_len is not None:
        config.model.max_seq_len = args.max_seq_len

    if args.device is not None:
        config.training.device = args.device
    if args.resume_from is not None:
        config.training.resume_from = args.resume_from

    if args.eval_every is not None:
        config.training.eval_every = args.eval_every
    if args.eval_iters is not None:
        config.training.eval_iters = args.eval_iters
    if args.checkpoint_every is not None:
        config.training.checkpoint_every = args.checkpoint_every
    if args.log_every is not None:
        config.training.log_every = args.log_every
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
        # Tokenizer ikut disimpan di folder checkpoint yang sama, supaya
        # semua artefak satu eksperimen (checkpoint, log, tokenizer, config)
        # tetap satu folder, tidak tercecer di "checkpoints/" default lama.
        config.data.tokenizer_path = f"{args.checkpoint_dir}/tokenizer.json"
    if args.seed is not None:
        config.training.seed = args.seed

    if args.grad_accum_steps is not None:
        config.training.grad_accum_steps = args.grad_accum_steps
    if args.num_threads is not None:
        config.training.num_threads = args.num_threads
    if args.tokenizer is not None:
        config.data.tokenizer = args.tokenizer
    if args.bpe_vocab_size is not None:
        config.data.bpe_vocab_size = args.bpe_vocab_size
    return config


def main():
    args = parse_args()
    config = Config()
    config = apply_overrides(config, args)

    torch.manual_seed(config.training.seed)

    if config.training.num_threads is not None:
        torch.set_num_threads(config.training.num_threads)
    print(f"PyTorch memakai {torch.get_num_threads()} thread CPU")

    # --- Tokenizer ---
    # Supaya training lanjutan (--resume_from) dengan file data baru tidak
    # merusak checkpoint lama, tokenizer HANYA di-fit kalau memang belum ada
    # file tokenizer tersimpan (atau kalau --rebuild_tokenizer dipaksa).
    # Kalau sudah ada, vocab lama dipakai apa adanya supaya ukuran & isi
    # vocab tetap konsisten dengan checkpoint model yang mau di-resume.
    TokenizerClass = get_tokenizer_class(config.data.tokenizer)
    tokenizer_exists = os.path.exists(config.data.tokenizer_path)

    if tokenizer_exists and not args.rebuild_tokenizer:
        print(f"Memuat tokenizer yang sudah ada: {config.data.tokenizer_path}")
        tokenizer = TokenizerClass.load(config.data.tokenizer_path)
        print(
            f"Tokenizer: {config.data.tokenizer} | Vocab size: {tokenizer.vocab_size} "
            "(dipakai ulang, tidak di-fit ulang)"
        )
    else:
        vocab_source = args.vocab_path or config.data.train_dir
        print(f"Melatih tokenizer baru ({config.data.tokenizer}) dari: {vocab_source}")
        vocab_text = load_text(vocab_source)
        tokenizer = TokenizerClass()
        if config.data.tokenizer == "bpe":
            tokenizer.fit(vocab_text, vocab_size=config.data.bpe_vocab_size)
        else:
            tokenizer.fit(vocab_text)
        tokenizer.save(config.data.tokenizer_path)
        print(
            f"Tokenizer: {config.data.tokenizer} | Vocab size: {tokenizer.vocab_size} "
            f"(baru, tersimpan di {config.data.tokenizer_path})"
        )

    config.model.vocab_size = tokenizer.vocab_size

    # --- Dataset ---
    print("Meng-encode corpus training/validasi...")
    train_ds, val_ds = make_train_val_datasets(
        tokenizer,
        config.data.train_dir,
        config.data.val_path,
        config.model.max_seq_len,
        config.data.val_split,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # --- Model ---
    model = TransformerLM(config.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Jumlah parameter: {n_params / 1e6:.2f}M")

    # --- Simpan config supaya inference.py bisa membangun ulang model ---
    config.save(f"{config.training.checkpoint_dir}/config.json")

    # --- Training ---
    trainer = Trainer(model, train_ds, val_ds, config)
    trainer.train()


if __name__ == "__main__":
    main()