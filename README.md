# CNN Image Classification Application

### Deskripsi Proyek

Proyek ini merupakan aplikasi klasifikasi gambar berbasis Convolutional Neural Network (CNN) yang dikembangkan sebagai bagian dari tugas optimasi dan implementasi model ke dalam aplikasi sederhana.

Aplikasi ini dibuat menggunakan Streamlit sebagai antarmuka web sederhana. Pengguna dapat mengunggah gambar, kemudian sistem akan melakukan prediksi kelas gambar menggunakan model CNN yang telah dilatih sebelumnya.

### Tujuan Proyek

Tujuan dari proyek ini adalah:

1. Mengoptimalkan model CNN untuk meningkatkan performa klasifikasi gambar.
2. Mengimplementasikan model ke dalam aplikasi web sederhana.
3. Melakukan pengujian model menggunakan gambar baru yang belum ada di dataset training.
4. Mendokumentasikan hasil pengujian dan evaluasi akhir model.

### Fitur Aplikasi

- Upload gambar melalui antarmuka web.
- Menampilkan gambar input yang diunggah.
- Menampilkan hasil prediksi kelas gambar.
- Menampilkan nilai confidence dari hasil prediksi.
- Menampilkan tabel perbandingan performa model.
- Menampilkan grafik akurasi training dan validasi.
- Menampilkan analisis hasil optimasi model.

### Struktur File

```text
cnn-image-classification/
│
├── app.py              # File utama aplikasi Streamlit
├── model.keras         # Model CNN yang telah dilatih
├── requirements.txt    # Daftar library yang dibutuhkan
├── train_model.ipynb   # Notebook proses training model
└── readme.txt          # Dokumentasi proyek
