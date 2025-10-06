import streamlit as st
from PIL import Image
import numpy as np
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.decode_qr import decode_qr
from scripts.verify_return import verify_returned_product

st.set_page_config(page_title="✅ AI Cloth Verification", layout="centered")
st.title("👕 AI Cloth Verification System")

# --- Step 1: Upload Cloth Image ---
uploaded_file = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])
qr_path = None

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)

    # --- Step 2: Extract Features ---
    st.write("Processing the image with AI...")
    features = extract_features(img)
    st.success("Features extracted successfully ✅")

    # --- Step 3: Generate QR ---
    qr_img = generate_qr_for_product(features)
    qr_path = "qr_code.png"
    qr_img.save(qr_path)
    st.image(qr_img, caption="QR Code", width=250)
    st.success("QR code generated ✅")

# --- Step 4: Verify Returned Product ---
st.subheader("Verify Returned Product")
returned_file = st.file_uploader("Upload returned product image", type=["jpg", "jpeg", "png"], key="returned")
if returned_file and qr_path:
    returned_img = Image.open(returned_file).convert("RGB")
    st.image(returned_img, caption="Returned Product", width=250)

    st.write("Extracting features from returned product...")
    returned_features = extract_features(returned_img)

    # Decode original features from QR
    original_features = decode_qr(qr_path)

    if original_features is not None:
        # Verify
        result, score = verify_returned_product(original_features, returned_features)
        st.write(f"Verification Result: **{result}**")
        st.write(f"Similarity Score: {score:.2f}")
    else:
        st.warning("⚠️ Could not verify. QR decoding failed.")
