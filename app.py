# ===============================
# IMPORT LIBRARY
# ===============================
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
import os

# TensorFlow dibuat optional agar aplikasi tetap bisa berjalan jika mode API dipakai
try:
    import tensorflow as tf
except Exception:
    tf = None


# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="CNN Image Classification",
    page_icon="🖼️",
    layout="centered"
)


# ===============================
# HEADER APLIKASI
# ===============================
st.title("CNN Image Classification - Group 2")
st.caption("Aplikasi klasifikasi gambar menggunakan model Convolutional Neural Network")


# ===============================
# DAFTAR KELAS CIFAR-10
# ===============================
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ===============================
# SIDEBAR MODE PREDIKSI
# ===============================
st.sidebar.header("Pengaturan Prediksi")

prediction_mode = st.sidebar.radio(
    "Pilih mode prediksi:",
    [
        "Model lokal / Streamlit Cloud",
        "API ngrok / Backend eksternal"
    ]
)

api_url = ""

if prediction_mode == "API ngrok / Backend eksternal":
    api_url = st.sidebar.text_input(
        "Masukkan URL API /predict",
        placeholder="https://nama-ngrok.ngrok-free.app/predict"
    )

st.sidebar.info(
    "Gunakan mode model lokal jika file model.keras berada satu folder dengan app.py. "
    "Gunakan mode API jika model dijalankan melalui backend seperti Flask/FastAPI/ngrok."
)


# ===============================
# LOAD MODEL LOKAL
# ===============================
@st.cache_resource
def load_local_model():
    if tf is None:
        raise ImportError("TensorFlow belum terinstall.")

    model_path = "model.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError("File model.keras tidak ditemukan.")

    model = tf.keras.models.load_model(model_path)
    return model


# ===============================
# PREPROCESSING GAMBAR
# ===============================
def preprocess_image(image, target_size=(32, 32)):
    """
    Preprocessing gambar agar sesuai dengan input model CIFAR-10.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)

    img_array = np.array(image)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ===============================
# PREDIKSI MODEL LOKAL
# ===============================
def predict_with_local_model(image):
    model = load_local_model()
    img_array = preprocess_image(image)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": confidence
    }


# ===============================
# PREDIKSI VIA API
# ===============================
def predict_with_api(uploaded_file, url):
    if not url:
        raise ValueError("URL API belum diisi.")

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    response = requests.post(
        url,
        files=files,
        timeout=30,
        verify=False
    )

    response.raise_for_status()
    return response.json()


# ===============================
# UPLOAD GAMBAR MANUAL
# ===============================
st.subheader("Upload Gambar")

uploaded_file = st.file_uploader(
    "Pilih gambar untuk diuji",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Gambar Input",
            use_container_width=True
        )

        st.write("Nama file:", uploaded_file.name)
        st.write("Ukuran file:", round(uploaded_file.size / 1024, 2), "KB")

        if st.button("Prediksi Gambar"):
            with st.spinner("Sedang memproses gambar..."):
                if prediction_mode == "Model lokal / Streamlit Cloud":
                    result = predict_with_local_model(image)
                else:
                    result = predict_with_api(uploaded_file, api_url)

            predicted_class = result.get("class", "Tidak diketahui")
            confidence = result.get("confidence", 0)

            st.success("Prediksi berhasil")

            st.subheader("Hasil Prediksi")
            st.write("Kelas Prediksi:", predicted_class)
            st.write("Confidence:", f"{confidence * 100:.2f}%")

            # Tampilkan interpretasi confidence
            if confidence >= 0.80:
                st.info("Model memiliki tingkat keyakinan tinggi terhadap prediksi ini.")
            elif confidence >= 0.50:
                st.warning("Model memiliki tingkat keyakinan sedang terhadap prediksi ini.")
            else:
                st.error("Model memiliki tingkat keyakinan rendah. Gambar mungkin kurang jelas atau tidak sesuai kelas model.")

    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Pastikan file model.keras sudah di-upload ke repository dan berada satu folder dengan app.py.")

    except ImportError as e:
        st.error(str(e))
        st.info("Jika menggunakan model lokal, pastikan TensorFlow sudah ada di requirements.txt.")

    except requests.exceptions.HTTPError as e:
        st.error(f"Gagal memanggil API: {e}")
        st.info("Pastikan endpoint API benar, misalnya: https://nama-ngrok.ngrok-free.app/predict")

    except requests.exceptions.ConnectionError:
        st.error("Gagal terhubung ke API.")
        st.info("Pastikan ngrok/API backend sedang aktif.")

    except requests.exceptions.Timeout:
        st.error("Request ke API timeout.")
        st.info("Coba ulangi atau pastikan backend tidak sedang sibuk.")

    except Exception as e:
        st.error(f"Terjadi error: {e}")


# ===============================
# INFORMASI KELAS
# ===============================
st.subheader("Daftar Kelas Model")

class_df = pd.DataFrame({
    "No": list(range(1, len(CLASS_NAMES) + 1)),
    "Kelas": CLASS_NAMES
})

st.dataframe(class_df, use_container_width=True)


# ===============================
# TABEL PERBANDINGAN MODEL
# ===============================
st.subheader("Perbandingan Optimasi Model CNN")

data = {
    "Model": [
        "Baseline CNN",
        "CNN + Dropout",
        "CNN + Batch Normalization",
        "CNN + Dropout + Batch Normalization",
        "CNN + Full Tuning"
    ],
    "Akurasi Train": [0.92, 0.88, 0.90, 0.89, 0.91],
    "Akurasi Validasi": [0.78, 0.83, 0.85, 0.87, 0.90],
    "Loss": [
        "Tinggi",
        "Sedang",
        "Lebih kecil",
        "Rendah",
        "Paling rendah"
    ]
}

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)


# ===============================
# GRAFIK PERFORMA MODEL
# ===============================
st.subheader("Grafik Performa Model")
st.line_chart(df.set_index("Model")[["Akurasi Train", "Akurasi Validasi"]])


# ===============================
# ANALISIS
# ===============================
st.subheader("Analisis")

st.write("""
Model CNN digunakan untuk melakukan klasifikasi gambar ke dalam 10 kelas objek. 
Pada tahap evaluasi, gambar baru dapat diunggah secara manual melalui aplikasi. 
Sistem kemudian akan memproses gambar dan menampilkan hasil prediksi beserta nilai confidence.

Model baseline cenderung mengalami overfitting karena akurasi training lebih tinggi dibandingkan akurasi validasi. 
Penggunaan Dropout membantu mengurangi overfitting, sedangkan Batch Normalization membuat proses training lebih stabil. 
Kombinasi optimasi tersebut memberikan hasil generalisasi yang lebih baik pada data validasi maupun gambar baru.
""")


# ===============================
# FOOTER
# ===============================
st.success("Aplikasi siap digunakan.")
