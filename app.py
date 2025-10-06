# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os

# Import your AI scripts
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr
from scripts.verify_return import verify_returned_product

st.title("✅ AI Cloth Verification")

uploaded_file = st.file_uploader("Upload a cloth image to verify the product using AI", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Simulate AI processing
    st.write("Processing the image with AI...")
    features = extract_features(image)
    st.success("Features extracted successfully ✅")

    # Generate QR code
    qr_code = generate_qr(image)
    st.image(qr_code, caption="Generated QR Code", use_container_width=True)
    st.success("QR code generated ✅")

    # Verify the product
    result, similarity = verify_returned_product(features)
    st.write(f"Verification Result: {result}")
    st.write(f"Similarity Score: {similarity}")
