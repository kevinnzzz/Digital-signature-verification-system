import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image

# Fungsi testing (copy dari yang Anda miliki)
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
model = joblib.load('model_knn.pkl')

# Gambar contoh
sample_options = {
    "Genuine 1": "068/06_068.png",
    "Genuine 2": "068/07_068.png",
    "Forged 1": "068_forg/03_0113068.PNG"
}

st.title("Signature Verification System")

# Kolom untuk memilih gambar
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gambar 1 (Referensi)")
    img1_option = st.radio("Pilih atau Upload Gambar 1:", list(sample_options.keys()) + ["Upload sendiri"])
    if img1_option == "Upload sendiri":
        uploaded_img1 = st.file_uploader("Upload Gambar 1", key="img1")
        if uploaded_img1 is not None:
            img1_path = f"temp_img1.png"
            with open(img1_path, "wb") as f:
                f.write(uploaded_img1.read())
            img1_disp = Image.open(img1_path)
            st.image(img1_disp, caption="Gambar 1", use_column_width=True)
    else:
        img1_path = sample_options[img1_option]
        st.image(img1_path, caption="Gambar 1", use_column_width=True)

with col2:
    st.subheader("Gambar 2 (Pembanding)")
    img2_option = st.radio("Pilih atau Upload Gambar 2:", list(sample_options.keys()) + ["Upload sendiri"], key="img2")
    if img2_option == "Upload sendiri":
        uploaded_img2 = st.file_uploader("Upload Gambar 2", key="img2")
        if uploaded_img2 is not None:
            img2_path = f"temp_img2.png"
            with open(img2_path, "wb") as f:
                f.write(uploaded_img2.read())
            img2_disp = Image.open(img2_path)
            st.image(img2_disp, caption="Gambar 2", use_column_width=True)
    else:
        img2_path = sample_options[img2_option]
        st.image(img2_path, caption="Gambar 2", use_column_width=True)

# Tombol Prediksi
if st.button("Predict"):
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        fitur = testing(img1_path, img2_path)
        hasil = model.predict([fitur])[0]
        st.success("Matched!" if hasil == 0 else "Not Matched!")
    else:
        st.warning("Pastikan kedua gambar telah dipilih atau diupload dengan benar.")
