# app.py
import streamlit as st
import numpy as np
from scripts.decode_qr import decode_qr_to_features

st.set_page_config(page_title="✅ AI Cloth Verification", layout="centered")

st.title("👕 AI Cloth Verification System")

uploaded_file = st.file_uploader("Upload a QR Code Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded QR Code", use_container_width=True)

    with open("temp_qr_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = decode_qr_to_features("temp_qr_image.png")

    if features is None:
        st.error("⚠️ QR decoding not available or failed due to missing dependencies.")
        st.info("ℹ️ Try reloading or ensure required libraries (OpenCV, pyzbar) are installed.")
    else:
        st.success("✅ QR decoded successfully!")
        st.write("Extracted Features (sample):")
        st.write(features[:10])
else:
    st.info("📤 Please upload a QR code image to start.")
