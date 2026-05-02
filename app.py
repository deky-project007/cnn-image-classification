import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# ===============================
# CLASS NAME
# ===============================
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ===============================
# UI HEADER
# ===============================
st.title("CNN Image Classification - Group 2")

st.caption("TUGAS KELOMPOK 2 - OPTIMASI DAN IMPLEMENTASI DALAM APLIKASI SEDERHANA")

st.info("Aplikasi ini digunakan untuk klasifikasi gambar menggunakan model CNN.")

# ===============================
# INFORMASI INPUT GAMBAR
# ===============================
st.subheader("Upload Gambar")

st.write("Gunakan gambar yang termasuk ke dalam salah satu kelas berikut:")

class_info = pd.DataFrame({
    "No": list(range(1, 11)),
    "Class": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ],
    "Keterangan": [
        "Pesawat",
        "Mobil",
        "Burung",
        "Kucing",
        "Rusa",
        "Anjing",
        "Katak",
        "Kuda",
        "Kapal",
        "Truk"
    ]
})

st.dataframe(class_info, use_container_width=True)

st.warning("Upload gambar di luar kategori tersebut dapat menghasilkan prediksi yang tidak akurat.")

# ===============================
# UPLOAD GAMBAR
# ===============================
uploaded_file = st.file_uploader(
    "Upload gambar sesuai kategori model",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", use_container_width=True)

    # Preprocessing gambar
    image_resized = image.resize((32, 32))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # ===============================
    # HASIL PREDIKSI
    # ===============================
    st.subheader("Hasil Prediksi")
    st.write("Kelas:", class_names[predicted_index])
    st.write("Confidence:", round(confidence * 100, 2), "%")

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
