# Project: CNN Image Classification

## Requirements

Aplikasi ini direkomendasikan menggunakan:

- Python 3.11
- Streamlit
- TensorFlow CPU
- Pillow
- NumPy
- Pandas

Cara menjalankan:
1. Install library:
   python -m pip install -r requirements.txt

2. Jalankan:
   python -m streamlit run app.py

3. Buka aplikasi di browser:
   http://localhost:8501

### Struktur File

```text
cnn-image-classification/
│
├── app.py              # File utama aplikasi Streamlit
├── model.keras         # Model CNN yang telah dilatih
├── requirements.txt    # Daftar library yang dibutuhkan
├── runtime.txt         # Pengaturan versi Python untuk deployment
├── train_model.ipynb   # Notebook proses training model
└── readme.txt          # Dokumentasi proyek
