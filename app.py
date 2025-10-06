# Force redeploy comment
import streamlit as st
from PIL import Image
import numpy as np

from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.verify_return import verify_returned_product


st.set_page_config(page_title="✅ AI Cloth Verification", layout="centered")
st.title("👕 AI Cloth Verification System")

# --- Step 1: Upload Original Cloth ---
uploaded_file = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])
features_str = None

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", width=250)

    st.write("Processing the image with AI...")
    features = extract_features(img)
    st.success("Features extracted successfully ✅")

    st.write("Generating QR code...")
    qr_img, features_str = generate_qr_for_product(features)
    st.image(qr_img, caption="QR Code", width=250)
    st.success("QR code generated ✅")

# --- Step 2: Verify Returned Product ---
st.subheader("Verify Returned Product")
returned_file = st.file_uploader("Upload returned product image", type=["jpg", "jpeg", "png"], key="returned")

if returned_file and features_str:
    returned_img = Image.open(returned_file).convert('RGB')
    returned_features = extract_features(returned_img)

    # Verification
    result, score = verify_returned_product(list(map(float, features_str.split(','))), returned_features)
    st.write(f"Verification Result: **{result}**")
    st.write(f"Similarity Score: {score:.2f}")
