====================================================================
 my_ai — PANDUAN LENGKAP
 Framework training model bahasa transformer dari nol
 (khusus perangkat terbatas: Raspberry Pi 5 / Termux Android)
====================================================================

--------------------------------------------------------------------
1. TENTANG PROJECT
--------------------------------------------------------------------
my_ai adalah framework untuk melatih model bahasa (language model)
gaya GPT dari NOL — bukan fine-tune dari model pretrained. Model
belajar lewat next-token prediction dari teks yang kamu sediakan
sendiri di folder data/training/.

Struktur folder utama:
  config.py         -> semua pengaturan model / training / data
  train.py          -> jalankan training
  inference.py      -> generate teks satu kali dari sebuah prompt
  test_chatbot.py   -> uji tanya-jawab (mode batch / interaktif)
  data/             -> tokenizer & dataset
  model/            -> arsitektur transformer
  training/         -> loop training, optimizer, checkpoint
  evaluation/       -> metrik evaluasi (loss, perplexity, accuracy)
  data/training/    -> taruh SEMUA file .txt data percakapan di sini
  checkpoints/      -> hasil training (model, tokenizer, config, log)

--------------------------------------------------------------------
2. INSTALASI / DEPENDENSI
--------------------------------------------------------------------
Dibutuhkan Python 3.9 ke atas.

Disarankan pakai virtual environment:
  python -m venv venv
  source venv/bin/activate        (Linux / Termux / Raspberry Pi)
  venv\Scripts\activate           (Windows)

Install dependensi:
  pip install -r requirements.txt

Isi requirements.txt:
  torch>=2.0.0    (build CPU, pas untuk Raspberry Pi / Termux tanpa GPU)
  tqdm>=4.66.0

Catatan: kalau instalasi torch lambat/gagal di Termux/Raspberry Pi,
pastikan koneksi internet stabil dan ruang penyimpanan cukup —
wheel torch versi CPU ukurannya cukup besar.

--------------------------------------------------------------------
3. FORMAT DATA TRAINING
--------------------------------------------------------------------
Semua file .txt di dalam folder data/training/ otomatis digabung
jadi satu korpus training (nama file bebas, boleh banyak file
sekaligus).

Formatnya HARUS pola tanya-jawab seperti ini, supaya cocok dengan
cara test_chatbot.py melakukan prompting ke model:

  USER: <pertanyaan>
  ASSISTANT: <jawaban>

  USER: <pertanyaan lain>
  ASSISTANT: <jawaban lain>

Beri satu baris kosong di antara tiap pasangan USER/ASSISTANT.
Makin banyak dan makin beragam topiknya, makin bagus hasilnya —
idealnya ratusan sampai ribuan pasang, bukan cuma belasan, supaya
model tidak sekadar menghafal data yang sedikit (overfitting).

--------------------------------------------------------------------
4. VOCAB TETAP — SUPAYA BISA LANJUT TRAINING TANPA ULANG DARI 0
--------------------------------------------------------------------
Tokenizer default project ini adalah BPE byte-level, jadi begitu
vocab-nya "dikunci" sekali di awal, file data percakapan baru
berikutnya bisa dipakai untuk LANJUT training tanpa merusak
checkpoint lama.

Langkah-langkahnya:

  1) Siapkan file referensi vocab, misalnya data/kamus.txt, berisi
     kosakata/kalimat umum bahasa Indonesia yang luas.
     PENTING: taruh file ini DI LUAR folder data/training/, supaya
     isinya tidak ikut tercampur jadi data percakapan.

  2) Training PERTAMA KALI, sekalian bangun vocab dari kamus.txt:
       python train.py --vocab_path data/kamus.txt

     Ini akan:
       - fit tokenizer dari data/kamus.txt, simpan ke
         checkpoints/tokenizer.json
       - melatih model dari SELURUH isi data/training/ seperti biasa
       - menyimpan checkpoint ke checkpoints/final.pt & best.pt

  3) Nanti kalau mau nambah data baru:
       - taruh file .txt percakapan baru di data/training/
       - jalankan ulang TANPA --vocab_path, tambahkan --resume_from:
           python train.py --resume_from checkpoints/final.pt

     Karena checkpoints/tokenizer.json sudah ada, train.py otomatis
     memakainya lagi (TIDAK di-fit ulang), sehingga vocab tetap
     identik dengan yang dipakai checkpoint lama -> training lanjut
     dari bobot sebelumnya, bukan dari nol.

     Catatan: tiap resume, SELURUH isi data/training/ (lama + baru)
     tetap dipakai lagi bersamaan, supaya model tidak "lupa" pola
     dari data lama (mencegah catastrophic forgetting).

  Kalau memang sengaja mau membangun ulang vocab dari nol (misal
  ganti total gaya/topik data), pakai:
       python train.py --rebuild_tokenizer
  HATI-HATI: ini membuat checkpoint lama TIDAK kompatibel lagi
  untuk dipakai bareng --resume_from.

--------------------------------------------------------------------
5. CARA TRAINING — RINGKASAN PERINTAH
--------------------------------------------------------------------
Training paling dasar (pakai semua nilai default):
  python train.py

Training dengan pengaturan umum:
  python train.py --epochs 10 --batch_size 16 --device cpu

Khusus Raspberry Pi 5 (batasi thread CPU biar stabil):
  python train.py --num_threads 4

Opsi CLI yang paling sering dipakai:
  --train_path         path folder/file data training
                        (default: data/training)
  --val_path            path data validasi opsional
                        (default: otomatis split dari data training)
  --epochs               jumlah putaran training (default: 5)
  --max_steps             jumlah step eksplisit, override --epochs
  --batch_size             ukuran batch (default: 32)
  --learning_rate           laju belajar (default: 3e-4)
  --grad_accum_steps        akumulasi gradien N micro-batch
                            (batch efektif lebih besar tanpa nambah RAM)
  --num_threads              jumlah thread CPU intra-op PyTorch
                            (disarankan diisi manual di Raspberry Pi)
  --checkpoint_dir            folder simpan checkpoint (default: checkpoints)
  --resume_from                lanjutkan training dari file checkpoint .pt
  --tokenizer                   "bpe" (default) atau "char"
  --bpe_vocab_size               target ukuran vocab BPE (default: 1024)
  --vocab_path                    sumber teks buat fit vocab PERTAMA KALI
                                 (mis. data/kamus.txt)
  --rebuild_tokenizer               paksa fit ulang vocab dari nol
                                 (checkpoint lama jadi tidak kompatibel)
  --device                            cpu / cuda / mps
  --seed                                seed random (default: 1337)

--------------------------------------------------------------------
6. CARA TES MODEL SETELAH TRAINING
--------------------------------------------------------------------
Mode batch — jalankan beberapa contoh pertanyaan bawaan:
  python test_chatbot.py

Mode interaktif — ketik pertanyaan sendiri, ketik 'exit' untuk keluar:
  python test_chatbot.py --interactive

Pakai checkpoint tertentu (bukan default checkpoints/best.pt):
  python test_chatbot.py --checkpoint checkpoints/final.pt

Generate teks bebas satu kali, tanpa format tanya-jawab:
  python inference.py --checkpoint checkpoints/best.pt --prompt "Pada suatu hari"

--------------------------------------------------------------------
7. CATATAN .gitignore
--------------------------------------------------------------------
File hasil training (checkpoints/, *.pt, *.pth, *.ckpt) dan log
training sengaja di-ignore git karena ukurannya besar dan berubah
tiap kali training. Tapi SEMUA file di data/training/*.txt SENGAJA
TIDAK di-ignore, supaya setiap perubahan atau tambahan data selalu
kedeteksi git dan tetap bisa di-add/commit/update seperti biasa.

--------------------------------------------------------------------
8. TROUBLESHOOTING UMUM
--------------------------------------------------------------------
- Error: "Dataset training kosong/lebih kecil dari batch_size"
  -> Data terlalu sedikit dibanding batch_size/max_seq_len.
     Tambah data training, atau perkecil --batch_size / --max_seq_len.

- Error "size mismatch" saat pakai --resume_from
  -> Vocab tokenizer berubah (biasanya karena --rebuild_tokenizer
     dipakai, atau file checkpoints/tokenizer.json terhapus/pindah).
     Pastikan tokenizer.json yang lama tetap ada dan tidak di-fit ulang.

- Metrik eval (loss/ppl/acc) selalu 0.0000
  -> Data validasi lebih pendek dari max_seq_len. Perbesar val_split
     di config.py, atau sediakan file data/val.txt terpisah yang
     cukup panjang.

- Hasil generate model ngaco / ngasal
  -> Wajar kalau data training masih sedikit (model masih "polos").
     Tambah jumlah dan variasi pasangan USER:/ASSISTANT:, lalu
     training lebih lama (naikkan --epochs atau --max_steps).

====================================================================
 Selesai. Simpan file ini sebagai acuan tiap kali mau training atau
 nambah data baru ke project my_ai.
====================================================================
