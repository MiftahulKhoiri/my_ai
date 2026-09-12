"""
evaluation/metrics.py — Metrik evaluasi untuk language model: loss rata-rata,
perplexity, dan akurasi prediksi token berikutnya.
"""

import math
import torch


def compute_perplexity(loss: float) -> float:
    """Perplexity = exp(loss), loss adalah cross-entropy rata-rata."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Akurasi prediksi token berikutnya (argmax logits vs target)."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


@torch.no_grad()
def evaluate(model, dataloader, device: str, max_iters: int = 50):
    """Rata-rata loss, perplexity, dan akurasi pada `max_iters` batch dari
    dataloader (biasanya dataloader validasi)."""
    model.eval()
    losses, accs = [], []
    for i, (x, y) in enumerate(dataloader):
        if i >= max_iters:
            break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        losses.append(loss.item())
        accs.append(compute_accuracy(logits, y))
    model.train()

    avg_loss = sum(losses) / max(1, len(losses))
    avg_acc = sum(accs) / max(1, len(accs))
    return {
        "loss": avg_loss,
        "perplexity": compute_perplexity(avg_loss),
        "accuracy": avg_acc,
    }
