# app.py
import streamlit as st
from PIL import Image
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.decode_qr import decode_qr

st.set_page_config(page_title="AI Cloth Verification", layout="centered")
st.title("✅ AI Cloth Verification")
st.write("Upload a cloth image to verify the product using AI")

# --- Step 0: Upload Image ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image as PIL and convert to RGB
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing the image with AI...")

    # --- Step 1: Extract Features ---
    try:
        features = extract_features(img)  # Pass PIL Image directly
        st.success("Features extracted successfully ✅")
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        st.stop()

    # --- Step 2: Generate QR Code ---
    try:
        qr_img = generate_qr_for_product(features)
        st.image(qr_img, caption="QR Code", width=250)
        st.success("QR code generated ✅")
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        st.stop()

    # --- Step 3: Optional Verification ---
    try:
        decoded_features = decode_qr(qr_img)
        st.write("Decoded features from QR:")
        st.write(decoded_features[:10], "...")  # show first 10 values
        st.success("QR code verified successfully ✅")
    except Exception as e:
        st.error(f"Error decoding QR code: {e}")
