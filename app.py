# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io

# Import your AI scripts
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr
from scripts.verify_return import verify_returned_product

st.title("✅ AI Cloth Verification")

# Step 1: Upload image
uploaded_file = st.file_uploader("Upload Cloth Image for Verification", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing the image with AI...")

    # Step 2: Extract features using your AI script
    features = extract_features(image)
    st.success("Features extracted successfully ✅")

    # Step 3: Generate QR code using your script
    qr_code_image = generate_qr(features)
    st.image(qr_code_image, caption="Generated QR Code", use_column_width=False)
    st.success("QR code generated ✅")

    # Step 4: Verify returned product
    result, similarity = verify_returned_product(features)
    st.write(f"Verification Result: {result}")
    st.write(f"Similarity Score: {similarity:.2f}")
