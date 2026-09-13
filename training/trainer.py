"""
training/trainer.py — Loop training utama: iterasi batch, forward-backward
(dengan dukungan gradient accumulation), logging berkala (ke terminal + file
JSONL), evaluasi berkala di data validasi, dan checkpointing berkala.
Progress bar memakai tqdm.
"""

import json
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .optimizer import build_optimizer, build_lr_scheduler
from .checkpoint import save_checkpoint, load_checkpoint
from evaluation.metrics import evaluate


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = config.training.device
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            drop_last=True,
        )

        self.optimizer = build_optimizer(
            self.model, config.training.learning_rate, config.training.weight_decay
        )

        self.max_steps = config.training.max_steps or (
            len(self.train_loader) * config.training.epochs
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            max_steps=self.max_steps,
            max_lr=config.training.learning_rate,
            min_lr=config.training.min_learning_rate,
        )

        self.step = 0
        self.best_val_loss = float("inf")

        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        self.log_path = os.path.join(config.training.checkpoint_dir, "train_log.jsonl")

        if config.training.resume_from:
            state = load_checkpoint(
                config.training.resume_from, self.model, self.optimizer, self.device
            )
            self.step = state["step"]
            self.best_val_loss = state["best_val_loss"]
            tqdm.write(f"Resume dari step {self.step}, best_val_loss={self.best_val_loss:.4f}")

    def _infinite_loader(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def _log(self, record: dict) -> None:
        """Tulis satu baris JSON ke file log training (untuk dianalisis/diplot belakangan)."""
        record = {"time": time.time(), "step": self.step, **record}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _run_eval_and_maybe_save(self) -> None:
        cfg = self.config.training
        metrics = evaluate(self.model, self.val_loader, self.device, cfg.eval_iters)
        tqdm.write(
            f"  eval @ step {self.step}: loss {metrics['loss']:.4f} | "
            f"ppl {metrics['perplexity']:.2f} | acc {metrics['accuracy']:.3f}"
        )
        self._log({"event": "eval", **metrics})

        is_first_eval = self.best_val_loss == float("inf")
        if metrics["loss"] < self.best_val_loss or is_first_eval:
            self.best_val_loss = metrics["loss"]
            save_checkpoint(
                f"{cfg.checkpoint_dir}/best.pt",
                self.model, self.optimizer, self.step, self.best_val_loss,
            )

    def train(self):
        cfg = self.config.training
        data_iter = self._infinite_loader()
        t0 = time.time()
        accum_steps = max(1, cfg.grad_accum_steps)

        progress = tqdm(total=self.max_steps, initial=self.step, desc="training", unit="step")
        while self.step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Satu "step" logis = grad_accum_steps micro-batch, diakumulasi
            # sebelum optimizer.step(). Ini memperbesar batch efektif jadi
            # (batch_size * grad_accum_steps) tanpa menambah RAM per micro-batch.
            for _ in range(accum_steps):
                x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)

                _, loss, _ = self.model(x, y)
                (loss / accum_steps).backward()
                accum_loss += loss.item()

            avg_loss = accum_loss / accum_steps

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            is_last_step = self.step == self.max_steps - 1

            if self.step % cfg.log_every == 0:
                dt = time.time() - t0
                lr = self.scheduler.get_last_lr()[0]
                tqdm.write(f"step {self.step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
                self._log({"event": "train", "loss": avg_loss, "lr": lr})
                t0 = time.time()

            should_eval = (self.step > 0 and self.step % cfg.eval_every == 0) or is_last_step
            if should_eval:
                self._run_eval_and_maybe_save()

            if self.step > 0 and self.step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    f"{cfg.checkpoint_dir}/step_{self.step}.pt",
                    self.model, self.optimizer, self.step, self.best_val_loss,
                )

            self.step += 1
            progress.update(1)

        progress.close()

        save_checkpoint(
            f"{cfg.checkpoint_dir}/final.pt",
            self.model, self.optimizer, self.step, self.best_val_loss,
        )
        tqdm.write("Training selesai. Checkpoint akhir tersimpan (final.pt & best.pt).")