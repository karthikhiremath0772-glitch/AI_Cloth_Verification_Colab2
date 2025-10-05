from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product as generate_qr
from scripts.verify_return import verify_returned_product


import streamlit as st
from PIL import Image
import numpy as np
import os

# Import your AI scripts
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product as generate_qr   # ✅ fixed import
from scripts.verify_return import verify_returned_product

st.title("✅ AI Cloth Verification")
st.write("Upload a cloth image to verify the product using AI")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Image uploaded successfully ✅")

    # Step 1: Extract Features
    st.write("Processing the image with AI...")
    features = extract_features(np.array(img))
    st.write("Features extracted successfully ✅")

    # Step 2: Generate QR
    st.subheader("Generated QR Code")
    qr_path = generate_qr("sample_product", "data/products")
    qr_img = Image.open(qr_path)
    st.image(qr_img, caption="QR Code", use_container_width=True)
    st.write("QR code generated ✅")

    # Step 3: Verify Return
    result, score = verify_returned_product(features, features)  # Dummy check with same features
    st.subheader("Verification Result")
    st.write(f"Result: **{result}**")
    st.write(f"Similarity Score: {score:.2f}")
