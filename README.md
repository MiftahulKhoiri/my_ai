# my_ai

Framework kecil untuk melatih dan menjalankan model bahasa berbasis
transformer (gaya GPT, decoder-only) dari nol, memakai PyTorch murni.

## Struktur

```
my_ai/
├── data/
│   ├── tokenizer.py     # tokenizer level karakter, tanpa dependensi eksternal
│   └── dataset.py       # dataset PyTorch untuk next-token prediction
├── model/
│   ├── layers.py        # embedding token+posisi, feed-forward, init bobot
│   ├── attention.py     # causal multi-head self-attention
│   └── transformer.py   # TransformerBlock + model TransformerLM penuh
├── training/
│   ├── optimizer.py     # AdamW + scheduler warmup->cosine decay
│   ├── checkpoint.py    # simpan/muat checkpoint
│   └── trainer.py       # loop training: forward-backward, eval, checkpoint
├── evaluation/
│   └── metrics.py       # loss, perplexity, akurasi
├── config.py            # semua hyperparameter (dataclass, bisa disimpan JSON)
├── train.py             # entry point training
└── inference.py         # generate teks dari checkpoint
```

## Instalasi

```bash
pip install -r requirements.txt
```

Hanya butuh `torch` — sengaja diminimalkan supaya ringan dijalankan di
perangkat terbatas (mis. Raspberry Pi, Termux).

## Training

Sudah ada contoh corpus kecil di `data/train.txt` supaya pipeline-nya bisa
langsung dicoba:

```bash
python train.py
```

Untuk memakai data sendiri dan mengubah ukuran model:

```bash
python train.py \
  --train_path path/ke/corpus.txt \
  --epochs 10 \
  --batch_size 16 \
  --d_model 256 \
  --n_layers 4 \
  --n_heads 4 \
  --max_seq_len 256 \
  --device cpu
```

Checkpoint (`best.pt`, `final.pt`, `step_*.pt`), `tokenizer.json`, dan
`config.json` akan tersimpan di folder `checkpoints/`.

## Inference

```bash
python inference.py --checkpoint checkpoints/best.pt --prompt "Pada suatu hari"
```

## Catatan

- Tokenizer level karakter dipilih supaya tidak perlu library tambahan.
  Untuk vocab dan hasil yang lebih baik pada dataset besar, tokenizer ini
  bisa diganti dengan BPE tanpa mengubah bagian lain (dataset, model,
  training tidak bergantung pada detail internal tokenizer).
- Ukuran model default (`d_model=256`, `n_layers=4`, `n_heads=4`) sengaja
  kecil supaya training dengan CPU tetap terasa cepat sebagai contoh.
  Naikkan sesuai kebutuhan dan kekuatan perangkat.
- `data/train.txt` hanyalah contoh kecil untuk menguji pipeline — ganti
  dengan corpus asli sebelum melatih model yang benar-benar berguna.
