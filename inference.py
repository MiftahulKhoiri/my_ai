"""
inference.py — Generate teks dari checkpoint yang sudah dilatih.

Contoh pemakaian:
    python inference.py --checkpoint checkpoints/best.pt --prompt "Halo dunia"
"""

import argparse
import torch

from config import Config
from data.tokenizer import CharTokenizer
from model.transformer import TransformerLM
from training.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Inference dengan model my_ai")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config_path", type=str, default="checkpoints/config.json")
    p.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer.json")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    config = Config.load(args.config_path)
    tokenizer = CharTokenizer.load(args.tokenizer_path)

    model = TransformerLM(config.model)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=args.device)
    model.to(args.device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt) or [tokenizer.bos_id]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
