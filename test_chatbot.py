"""
test_chatbot.py — Skrip uji tanya-jawab untuk model my_ai.

Dua mode:
1. Otomatis (default): menjalankan beberapa contoh pertanyaan bawaan lalu
   mencetak hasilnya — cara cepat sanity-check model setelah training.
2. Interaktif (--interactive): kamu ketik pertanyaan sendiri, model
   menjawab langsung, berulang sampai diketik 'exit' atau 'keluar'.

Contoh pemakaian:
    python test_chatbot.py
    python test_chatbot.py --interactive
    python test_chatbot.py --checkpoint checkpoints/final.pt --num_threads 4
"""

import argparse
import torch

from config import Config
from data import get_tokenizer_class
from model.transformer import TransformerLM
from training.checkpoint import load_checkpoint


DEFAULT_TEST_PROMPTS = [
    "Apa itu machine learning?",
    "Bagaimana cara belajar Python dari awal?",
    "Kenapa penting menulis komentar di kode?",
    "Apa itu overfitting?",
    "Ada tips biar lebih produktif?",
    "Apa bedanya list dan tuple di Python?",
    "Selamat pagi!",
    "Kenapa langit berwarna biru?",
]

STOP_SEQUENCE = "\nUSER:"


def parse_args():
    p = argparse.ArgumentParser(description="Uji tanya-jawab model my_ai")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--config_path", type=str, default="checkpoints/config.json")
    p.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer.json")
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--num_threads", type=int, default=None,
        help="Jumlah thread CPU intra-op PyTorch (mis. 4 di Raspberry Pi 5).",
    )
    p.add_argument(
        "--interactive", action="store_true",
        help="Mode interaktif: ketik pertanyaan sendiri, bukan pakai contoh bawaan.",
    )
    return p.parse_args()


def load_model(args):
    config = Config.load(args.config_path)
    TokenizerClass = get_tokenizer_class(config.data.tokenizer)
    tokenizer = TokenizerClass.load(args.tokenizer_path)

    model = TransformerLM(config.model)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=args.device)
    model.to(args.device)
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, question: str, args) -> str:
    """Format pertanyaan gaya training (USER:/ASSISTANT:), generate, lalu
    potong jawabannya sebelum model mulai berhalusinasi bikin giliran USER
    berikutnya sendiri."""
    prompt = f"USER: {question}\nASSISTANT:"
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    full_text = tokenizer.decode(out[0].tolist())

    # Ambil bagian setelah "ASSISTANT:" pertama, lalu potong sebelum giliran
    # USER berikutnya kalau model mulai menghasilkan itu sendiri.
    answer = full_text.split("ASSISTANT:", 1)[-1]
    if STOP_SEQUENCE in answer:
        answer = answer.split(STOP_SEQUENCE, 1)[0]
    return answer.strip()


def run_batch(model, tokenizer, args):
    print(f"Menjalankan {len(DEFAULT_TEST_PROMPTS)} pertanyaan uji otomatis...\n")
    for i, question in enumerate(DEFAULT_TEST_PROMPTS, 1):
        answer = ask(model, tokenizer, question, args)
        print(f"[{i}] USER: {question}")
        print(f"    ASSISTANT: {answer}")
        print()


def run_interactive(model, tokenizer, args):
    print("Mode interaktif. Ketik 'exit' atau 'keluar' untuk berhenti.\n")
    while True:
        try:
            question = input("USER: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "keluar", "quit"}:
            break
        if not question:
            continue
        answer = ask(model, tokenizer, question, args)
        print(f"ASSISTANT: {answer}\n")


def main():
    args = parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    model, tokenizer = load_model(args)

    if args.interactive:
        run_interactive(model, tokenizer, args)
    else:
        run_batch(model, tokenizer, args)


if __name__ == "__main__":
    main()