# my_ai — Panduan Lengkap

Framework untuk melatih model bahasa (language model) gaya GPT **dari nol**, dirancang untuk perangkat terbatas seperti **Raspberry Pi 5** atau **Termux Android**. Model belajar lewat *next-token prediction* dari teks yang kamu sediakan sendiri di `data/training/`.

## Daftar Isi

1. [Struktur Project](#struktur-project)
2. [Instalasi](#instalasi)
3. [Format Data Training](#format-data-training)
4. [Vocab Tetap (Lanjut Training Tanpa Ulang dari 0)](#vocab-tetap-lanjut-training-tanpa-ulang-dari-0)
5. [Cara Training](#cara-training)
6. [Cara Tes Model](#cara-tes-model)
7. [Catatan .gitignore](#catatan-gitignore)
8. [Troubleshooting](#troubleshooting)

---

## Struktur Project

```
my_ai/
├── config.py          # semua pengaturan model / training / data
├── train.py           # jalankan training
├── inference.py       # generate teks satu kali dari sebuah prompt
├── test_chatbot.py    # uji tanya-jawab (mode batch / interaktif)
├── data/
│   ├── training/       # taruh SEMUA file .txt data percakapan di sini
│   ├── tokenizer.py     # tokenizer level karakter
│   ├── bpe_tokenizer.py # tokenizer BPE byte-level (default)
│   └── dataset.py
├── model/              # arsitektur transformer
├── training/           # loop training, optimizer, checkpoint
├── evaluation/         # metrik evaluasi (loss, perplexity, accuracy)
└── checkpoints/        # hasil training: model, tokenizer, config, log
```

---

## Instalasi

Dibutuhkan **Python 3.9+**.

Disarankan pakai virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / Termux / Raspberry Pi
venv\Scripts\activate           # Windows
```

Install dependensi:

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:

```
torch>=2.0.0    # build CPU, pas untuk Raspberry Pi / Termux tanpa GPU
tqdm>=4.66.0
```

> Kalau instalasi torch lambat/gagal di Termux/Raspberry Pi, pastikan koneksi internet stabil dan ruang penyimpanan cukup — wheel torch versi CPU ukurannya cukup besar.

---

## Format Data Training

Semua file `.txt` di dalam `data/training/` otomatis digabung jadi satu korpus training (nama file bebas, boleh banyak file sekaligus).

Formatnya **harus** pola tanya-jawab berikut, supaya cocok dengan cara `test_chatbot.py` melakukan prompting ke model:

```
USER: <pertanyaan>
ASSISTANT: <jawaban>

USER: <pertanyaan lain>
ASSISTANT: <jawaban lain>
```

Beri satu baris kosong di antara tiap pasangan `USER`/`ASSISTANT`. Makin banyak dan makin beragam topiknya, makin bagus hasilnya — idealnya **ratusan sampai ribuan pasang**, bukan cuma belasan, supaya model tidak sekadar menghafal data yang sedikit (*overfitting*).

---

## Vocab Tetap (Lanjut Training Tanpa Ulang dari 0)

Tokenizer default project ini adalah **BPE byte-level**, jadi begitu vocab-nya "dikunci" sekali di awal, file data percakapan baru berikutnya bisa dipakai untuk **lanjut training** tanpa merusak checkpoint lama.

**1) Siapkan file referensi vocab**, misalnya `data/kamus.txt`, berisi kosakata/kalimat umum bahasa Indonesia yang luas.

> Penting: taruh file ini **di luar** `data/training/`, supaya isinya tidak ikut tercampur jadi data percakapan.

**2) Training pertama kali**, sekalian bangun vocab dari `kamus.txt`:

```bash
python train.py --vocab_path data/kamus.txt
```

Ini akan:
- fit tokenizer dari `data/kamus.txt`, simpan ke `checkpoints/tokenizer.json`
- melatih model dari **seluruh** isi `data/training/` seperti biasa
- menyimpan checkpoint ke `checkpoints/final.pt` & `checkpoints/best.pt`

**3) Nanti kalau mau nambah data baru:**

```bash
# taruh file .txt percakapan baru di data/training/, lalu:
python train.py --resume_from checkpoints/final.pt
```

Tanpa `--vocab_path` lagi — karena `checkpoints/tokenizer.json` sudah ada, `train.py` otomatis memakainya lagi (**tidak** di-fit ulang), sehingga vocab tetap identik dengan checkpoint lama → training lanjut dari bobot sebelumnya, bukan dari nol.

> Catatan: tiap resume, **seluruh** isi `data/training/` (lama + baru) tetap dipakai lagi bersamaan, supaya model tidak "lupa" pola dari data lama (mencegah *catastrophic forgetting*).

Kalau memang sengaja mau membangun ulang vocab dari nol (misal ganti total gaya/topik data):

```bash
python train.py --rebuild_tokenizer
```

⚠️ **Hati-hati:** ini membuat checkpoint lama **tidak kompatibel lagi** untuk dipakai bareng `--resume_from`.

---

## Cara Training

Training paling dasar (semua nilai default):

```bash
python train.py
```

Training dengan pengaturan umum:

```bash
python train.py --epochs 10 --batch_size 16 --device cpu
```

Khusus Raspberry Pi 5 (batasi thread CPU biar stabil):

```bash
python train.py --num_threads 4
```

### Opsi CLI yang sering dipakai

| Opsi | Fungsi | Default |
|---|---|---|
| `--train_path` | path folder/file data training | `data/training` |
| `--val_path` | path data validasi (opsional) | auto-split dari train |
| `--epochs` | jumlah putaran training | `5` |
| `--max_steps` | jumlah step eksplisit, override `--epochs` | – |
| `--batch_size` | ukuran batch | `32` |
| `--learning_rate` | laju belajar | `3e-4` |
| `--grad_accum_steps` | akumulasi gradien N micro-batch (batch efektif lebih besar tanpa nambah RAM) | `1` |
| `--num_threads` | jumlah thread CPU intra-op PyTorch (isi manual di Raspberry Pi) | auto |
| `--checkpoint_dir` | folder simpan checkpoint | `checkpoints` |
| `--resume_from` | lanjutkan training dari file checkpoint `.pt` | – |
| `--tokenizer` | `bpe` (default) atau `char` | `bpe` |
| `--bpe_vocab_size` | target ukuran vocab BPE | `1024` |
| `--vocab_path` | sumber teks buat fit vocab **pertama kali** (mis. `data/kamus.txt`) | `--train_path` |
| `--rebuild_tokenizer` | paksa fit ulang vocab dari nol (checkpoint lama jadi tidak kompatibel) | `False` |
| `--device` | `cpu` / `cuda` / `mps` | `cpu` |
| `--seed` | seed random | `1337` |

---

## Cara Tes Model

Mode batch — jalankan beberapa contoh pertanyaan bawaan:

```bash
python test_chatbot.py
```

Mode interaktif — ketik pertanyaan sendiri, ketik `exit` untuk keluar:

```bash
python test_chatbot.py --interactive
```

Pakai checkpoint tertentu (bukan default `checkpoints/best.pt`):

```bash
python test_chatbot.py --checkpoint checkpoints/final.pt
```

Generate teks bebas satu kali, tanpa format tanya-jawab:

```bash
python inference.py --checkpoint checkpoints/best.pt --prompt "Pada suatu hari"
```

---

## Catatan .gitignore

File hasil training (`checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`) dan log training sengaja di-*ignore* git karena ukurannya besar dan berubah tiap kali training. Tapi **semua** file di `data/training/*.txt` sengaja **tidak** di-*ignore*, supaya setiap perubahan atau tambahan data selalu kedeteksi git dan tetap bisa di-*add*/*commit*/*update* seperti biasa.

---

## Troubleshooting

**Error: `Dataset training kosong/lebih kecil dari batch_size`**
Data terlalu sedikit dibanding `batch_size`/`max_seq_len`. Tambah data training, atau perkecil `--batch_size` / `--max_seq_len`.

**Error `size mismatch` saat pakai `--resume_from`**
Vocab tokenizer berubah (biasanya karena `--rebuild_tokenizer` dipakai, atau file `checkpoints/tokenizer.json` terhapus/pindah). Pastikan `tokenizer.json` yang lama tetap ada dan tidak di-*fit* ulang.

**Metrik eval (loss/ppl/acc) selalu `0.0000`**
Data validasi lebih pendek dari `max_seq_len`. Perbesar `val_split` di `config.py`, atau sediakan file `data/val.txt` terpisah yang cukup panjang.

**Hasil generate model ngaco / ngasal**
Wajar kalau data training masih sedikit (model masih "polos"). Tambah jumlah dan variasi pasangan `USER:`/`ASSISTANT:`, lalu training lebih lama (naikkan `--epochs` atau `--max_steps`).

---

*Simpan file ini sebagai acuan tiap kali mau training atau nambah data baru ke project `my_ai`.*
