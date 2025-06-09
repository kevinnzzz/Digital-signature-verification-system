import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
import cv2

# Fungsi testing
def testing(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray2, (3, 3), 0)

    _, otsu1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    close1 = cv2.morphologyEx(otsu1, cv2.MORPH_CLOSE, kernel)
    close2 = cv2.morphologyEx(otsu2, cv2.MORPH_CLOSE, kernel)

    resized1 = cv2.resize(close1, (497, 126))
    resized2 = cv2.resize(close2, (497, 126))

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(resized1, None)
    kp2, des2 = sift.detectAndCompute(resized2, None)

    if des1 is None or des2 is None:
        return [0, 9999, 0]

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    jumlah_match = len(good)
    rata_jarak = np.mean([m.distance for m in good]) if good else 9999
    std_jarak = np.std([m.distance for m in good]) if good else 0
    median_jarak = np.median([m.distance for m in good]) if good else 9999
    rasio = jumlah_match / min(len(des1), len(des2))
    rasio_kp1 = jumlah_match / len(des1)
    rasio_kp2 = jumlah_match / len(des2)
    jumlah_kp1 = len(kp1)
    jumlah_kp2 = len(kp2)
    total_match = len(matches)

    return [
        jumlah_match, rata_jarak, rasio,
        std_jarak, median_jarak,
        rasio_kp1, rasio_kp2,
        jumlah_kp1, jumlah_kp2, total_match
    ]

# Load model
model = joblib.load('model_knn_new.pkl')

# Contoh gambar
sample_options_1 = {
    "Asli 1": "01_068.png",
    "Asli 2": "05_068.png"
}

sample_options_2 = {
    "Asli": "07_068.png",
    "Palsu": "03_0124068.PNG"
}

st.title("Digital Signature Verification System")

# Kolom untuk dua gambar
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gambar 1 (Referensi)")

    # Tampilkan gambar default atau upload
    img1_option = st.radio("Pilih Gambar 1:", list(sample_options_1.keys()) + ["Upload sendiri"], key="img1_radio")

    if img1_option == "Upload sendiri":
        uploaded_img1 = st.file_uploader("Upload Gambar 1", type=["png", "jpg", "jpeg"], key="img1_upload")
        if uploaded_img1 is not None:
            img1_path = "temp_img1.png"
            with open(img1_path, "wb") as f:
                f.write(uploaded_img1.read())
            st.image(img1_path, caption="Gambar 1", use_container_width=True)
    else:
        img1_path = sample_options_1[img1_option]
        st.image(img1_path, caption="Gambar 1", use_container_width=True)

with col2:
    st.subheader("Gambar 2 (Pembanding)")

    # Tampilkan gambar default atau upload
    img2_option = st.radio("Pilih Gambar 2:", list(sample_options_2.keys()) + ["Upload sendiri"], key="img2_radio")

    if img2_option == "Upload sendiri":
        uploaded_img2 = st.file_uploader("Upload Gambar 2", type=["png", "jpg", "jpeg"], key="img2_upload")
        if uploaded_img2 is not None:
            img2_path = "temp_img2.png"
            with open(img2_path, "wb") as f:
                f.write(uploaded_img2.read())
            st.image(img2_path, caption="Gambar 2", use_container_width=True)
    else:
        img2_path = sample_options_2[img2_option]
        st.image(img2_path, caption="Gambar 2", use_container_width=True)

# Tombol Prediksi
if st.button("Predict"):
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        fitur = testing(img1_path, img2_path)
        hasil = model.predict([fitur])[0]
        if hasil == 0:
            st.success("Cocok")
        else:
            st.warning("Tidak Cocok")
    else:
        st.warning("Pastikan kedua gambar sudah dipilih atau diupload.")
