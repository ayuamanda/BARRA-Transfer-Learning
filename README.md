# BARRA Transfer Learning: Klasifikasi Objek dengan MobileNetV2

## 1. Gambaran Proyek

BARRA (Barelang Articulate Robot Arm) adalah proyek Project Based Learning (PBL) robot arm.

BARRA (Barelang Articulate Robot Arm) adalah proyek robot lengan artikulasi yang dikembangkan
oleh tim BRAIL Polibatam sebagai bagian dari Project Based Learning (PBL). Proyek ini
menggabungkan aspek mekanik, elektronik, dan pemrograman untuk menghasilkan gerakan robot yang presisi.

Repositori ini berisi eksperimen transfer learning untuk mengklasifikasikan bentuk objek
(kubus/balok, silinder/kaleng, dan botol) yang berpotensi ditangani robot arm pada aplikasi pick-and-place.

## 2. Tujuan

Membangun model klasifikasi gambar dengan metode transfer learning (MobileNetV2 pretrained ImageNet)
untuk mengenali tiga kelas objek, lalu mengevaluasinya dengan data uji yang terpisah.

## 3. Dataset

> **PENTING: dataset ini sintetis (dibuat dengan AI), bukan foto asli.**
> Gambar dibuat dengan ChatGPT (GPT) pada Oktober 2026 dalam bentuk kolase foto realistis,
> lalu dipotong menjadi gambar satuan dengan `crop_collages.py`.

| Kelas | Gambar dihasilkan | Gambar dipakai |
|---|---:|---:|
| kubus (kotak/balok) | 32 | 32 |
| silinder (kaleng/tabung) | 32 | 31 |
| botol (botol plastik berleher) | 32 | 32 |
| **Total** | **96** | **95** |

Satu gambar silinder dibuang karena bentuknya ambigu (tumbler mirip botol).

**Pembagian data dilakukan per kolase**, bukan per gambar. Satu kolase (8 gambar) masuk ke
satu kelompok saja, supaya gambar yang mirip tidak bocor antara data latih dan data uji.

| Kelompok | Jumlah gambar |
|---|---:|
| Train | 47 |
| Validation | 24 |
| Test | 24 |

Struktur folder: `data/collages/<kelas>/` (kolase asli) dan `data/barra_dataset/{train,val,test}/<kelas>/` (hasil potongan).

## 4. Metode

- Model dasar: MobileNetV2 pretrained ImageNet (tanpa classification head), bobot dibekukan.
- Lapisan baru: GlobalAveragePooling2D, Dropout 0.2, dan Dense 3 kelas (softmax).
- Augmentasi saat training: flip horizontal, rotasi kecil, zoom.
- Input 224 x 224, batch size 16, optimizer Adam (learning rate 0.001), maksimum 20 epoch dengan EarlyStopping dan penyimpanan model terbaik berdasarkan `val_loss`.
- Hanya feature extraction (tanpa fine-tuning).
- Dijalankan di CPU (tanpa GPU), Ubuntu 24.04, Python 3.12.3, TensorFlow 2.21.0.

## 5. Hasil

Evaluasi pada data test (24 gambar, belum pernah dilihat model saat training):

- **Test accuracy: 91,67%** (22 dari 24 gambar benar)
- Test loss: 0,3788
- Validation accuracy akhir: 95,83% (dipakai untuk memantau training, bukan hasil utama)

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| botol | 0,889 | 1,000 | 0,941 | 8 |
| kubus | 0,889 | 1,000 | 0,941 | 8 |
| silinder | 1,000 | 0,750 | 0,857 | 8 |

Grafik dan detail ada di folder `results/`:

- `training_history.png`: grafik accuracy dan loss
- `confusion_matrix.png`: confusion matrix data test
- `classification_report.csv` dan `evaluation_metrics.json`: angka evaluasi
- `best_model.keras` dan `class_names.json`: model dan urutan kelas

Kesalahan model terjadi pada kelas silinder: dari 8 gambar silinder, 2 salah diprediksi
(satu sebagai botol dan satu sebagai kubus). Seluruh gambar botol dan kubus dikenali dengan benar.  

Contoh prediksi satu gambar:

```text
python predict.py data/barra_dataset/test/botol/botol_04_r1c1.png
Prediksi   : botol
Confidence : 98.97%
```

## 6. Keterbatasan

- Dataset **sintetis (buatan AI)**, sehingga hasil belum membuktikan model bekerja pada foto dari kamera sungguhan.
- Data test hanya 24 gambar. Satu kesalahan menggeser akurasi sekitar 4%, jadi angka ini kasar.
- Beberapa produk yang sama muncul di lebih dari satu kolase, sehingga model mungkin ikut mengenali produknya, bukan hanya bentuknya.
- Model hanya mengklasifikasikan gambar. Model tidak menghasilkan koordinat objek dan belum terhubung ke sistem kontrol robot.
- Hanya feature extraction yang dilakukan. Fine-tuning belum dicoba.

## 7. Cara Menjalankan

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python crop_collages.py   # potong kolase dan bagi ke train/val/test
python train.py           # latih model
python evaluate.py        # akurasi test dan classification report
python plot_results.py    # grafik training dan confusion matrix
python save_metadata.py   # simpan class_names.json dan evaluation_metrics.json
python predict.py data/barra_dataset/test/botol/botol_04_r1c1.png
```

`make_dummy.py` dan `cek_env.py` hanya dipakai untuk menguji lingkungan dan alur kode awal.

## 8. Pengembangan Lanjutan

- Uji dengan foto objek asli dari kamera.
- Coba fine-tuning dan tambah jumlah data.
- Evaluasi ketahanan terhadap pencahayaan dan latar yang berbeda.
- Kembangkan deteksi objek (lokasi) bila dibutuhkan untuk pengambilan oleh robot.

## 9. Referensi

- TensorFlow, Transfer Learning and Fine-Tuning: https://www.tensorflow.org/tutorials/images/transfer_learning