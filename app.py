# ===============================
# IMPORT LIBRARY
# ===============================
import streamlit as st
import requests
from PIL import Image
import urllib3
import pandas as pd

# Disable SSL warning (ngrok)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===============================
# CONFIG API
# ===============================
API_URL = "https://excluding-decaf-bruising.ngrok-free.dev/predict"

# ===============================
# UI HEADER
# ===============================
st.title("CNN Image Classification")

st.caption("TUGAS KELOMPOK 2 - OPTIMASI DAN IMPLEMENTASI DALAM APLIKASI SEDERHANA")

st.info("GROUP 2: INDRA KOESUMAH | INDHAH PUJIHASTUTI | ALVIYAN SYAFRIANSAH MATONDANG | INDRI TALITHA | MUHAMAD DEKY AKBAR")

# ===============================
# UPLOAD GAMBAR
# ===============================
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Input")

    files = {
        "file": uploaded_file.getvalue()
    }

    try:
        response = requests.post(API_URL, files=files, verify=False, timeout=10)
        response.raise_for_status()
        result = response.json()

        # ===============================
        # HASIL PREDIKSI
        # ===============================
        st.subheader("Hasil Prediksi")
        st.write("Kelas:", result["class"])
        st.write("Confidence:", round(result["confidence"] * 100, 2), "%")

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal konek ke API: {e}")

# ===============================
# TABEL PERBANDINGAN MODEL
# ===============================
st.subheader("📊 Perbandingan Optimasi Model CNN")

data = {
    "Model": [
        "Baseline CNN",
        "CNN + Dropout",
        "CNN + BatchNorm",
        "CNN + Dropout + BatchNorm",
        "CNN + Full Tuning"
    ],
    "Akurasi Train": [0.92, 0.88, 0.90, 0.89, 0.91],
    "Akurasi Validasi": [0.78, 0.83, 0.85, 0.87, 0.90],
    "Loss": ["Tinggi", "Sedang", "Lebih kecil", "Rendah", "Paling rendah"]
}

df = pd.DataFrame(data)

st.dataframe(df)

# ===============================
# GRAFIK
# ===============================
st.subheader("📈 Grafik Performa Model")
st.line_chart(df.set_index("Model")[["Akurasi Train", "Akurasi Validasi"]])

# ===============================
# ANALISIS
# ===============================
st.subheader("Analisa")

st.write("""
- Model baseline mengalami overfitting karena akurasi training tinggi tetapi validasi rendah.
- Dropout membantu mengurangi overfitting dengan cara menonaktifkan neuron secara acak saat training.
- Batch Normalization mempercepat training dan membuat model lebih stabil.
- Kombinasi Dropout dan BatchNorm memberikan hasil generalisasi yang lebih baik.
- Hyperparameter tuning menghasilkan performa terbaik dengan akurasi validasi tertinggi dan loss terendah.
""")

# ===============================
# FOOTER
# ===============================
st.success("Aplikasi berjalan dengan baik 🚀")
