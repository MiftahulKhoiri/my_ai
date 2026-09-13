"""
train.py — Entry point untuk melatih model.

Contoh pemakaian:
    python train.py
    python train.py --train_path data/train.txt --epochs 10 --device cuda
    python train.py --grad_accum_steps 4 --num_threads 4
"""

import argparse
import torch

from config import Config
from data.tokenizer import CharTokenizer
from data.dataset import make_train_val_datasets, load_text
from model.transformer import TransformerLM
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Training my_ai transformer")
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--val_path", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None)
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
    return p.parse_args()


def apply_overrides(config: Config, args) -> Config:
    if args.train_path:
        config.data.train_path = args.train_path
    if args.val_path:
        config.data.val_path = args.val_path
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
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
    if args.grad_accum_steps is not None:
        config.training.grad_accum_steps = args.grad_accum_steps
    if args.num_threads is not None:
        config.training.num_threads = args.num_threads
    return config


def main():
    args = parse_args()
    config = Config()
    config = apply_overrides(config, args)

    torch.manual_seed(config.training.seed)

    if config.training.num_threads is not None:
        torch.set_num_threads(config.training.num_threads)
    print(f"PyTorch memakai {torch.get_num_threads()} thread CPU")

    # --- Tokenizer: fit dari corpus training, lalu simpan untuk inference ---
    tokenizer = CharTokenizer()
    train_text = load_text(config.data.train_path)
    tokenizer.fit(train_text)
    tokenizer.save(config.data.tokenizer_path)
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {tokenizer.vocab_size}")

    # --- Dataset ---
    train_ds, val_ds = make_train_val_datasets(
        tokenizer,
        config.data.train_path,
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