"""
config.py — Konfigurasi terpusat untuk my_ai.

Semua hyperparameter model, training, dan data diatur lewat dataclass di
sini supaya train.py dan inference.py cukup import satu sumber yang sama.
Config bisa disimpan/dimuat sebagai JSON (dipakai lagi saat inference).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os


@dataclass
class ModelConfig:
    vocab_size: int = 256       # diisi ulang otomatis setelah tokenizer di-fit
    d_model: int = 256          # dimensi embedding
    n_layers: int = 4           # jumlah transformer block
    n_heads: int = 4            # jumlah attention head
    d_ff: int = 1024            # dimensi hidden feed-forward (biasanya 4 * d_model)
    max_seq_len: int = 256      # panjang konteks maksimum
    dropout: float = 0.1
    tie_weights: bool = True    # ikat bobot embedding dengan output head


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 5
    max_steps: Optional[int] = None       # kalau diisi, override epochs
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float = 1.0
    grad_accum_steps: int = 1             # akumulasi gradien N micro-batch sebelum
                                           # optimizer.step() — batch efektif lebih
                                           # besar tanpa nambah RAM per langkah
    eval_every: int = 200
    eval_iters: int = 50
    checkpoint_every: int = 500
    log_every: int = 20
    device: str = "cpu"                   # "cpu", "cuda", atau "mps"
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    num_threads: Optional[int] = None      # None = biarkan PyTorch auto-detect;
                                            # isi manual (mis. 4 di RPi5) buat kontrol
                                            # eksplisit jumlah thread CPU intra-op


@dataclass
class DataConfig:
    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    tokenizer_path: str = "checkpoints/tokenizer.json"
    val_split: float = 0.1                # dipakai kalau val_path tidak ditemukan
    tokenizer: str = "bpe"                # "bpe" atau "char"
    bpe_vocab_size: int = 1024            # target ukuran vocab (dipakai kalau tokenizer="bpe")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "model": asdict(self.model),
                    "training": asdict(self.training),
                    "data": asdict(self.data),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
        )