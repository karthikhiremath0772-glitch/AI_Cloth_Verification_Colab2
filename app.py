import streamlit as st
from PIL import Image
import numpy as np

# Import your modules
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.decode_qr import decode_qr
from scripts.verify_return import verify_returned_product

# --- App Title ---
st.title("✅ AI Cloth Verification")

# --- Step 1: Upload Cloth Image ---
uploaded_file = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)

    # --- Step 2: Extract Features ---
    st.write("Processing the image with AI...")
    features = extract_features(img)
    st.success("Features extracted successfully ✅")

    # --- Step 3: Generate QR Code ---
    st.subheader("Generated QR Code")
    qr_img = generate_qr_for_product(features)

    # Ensure it's a proper image object
    if hasattr(qr_img, "save"):
        qr_path = "qr_code.png"
        qr_img.save(qr_path)
        st.image(qr_img, caption="QR Code", width=250)
        st.success("QR code generated ✅")
    else:
        st.error("Failed to generate QR code image. Please check generate_qr.py")

    # --- Step 4: Verify Returned Product ---
    st.subheader("Verify Returned Product")
    returned_file = st.file_uploader("Upload returned product image", type=["jpg", "jpeg", "png"], key="returned")

    if returned_file:
        returned_img = Image.open(returned_file).convert("RGB")
        st.image(returned_img, caption="Returned Product", width=250)

        st.write("Extracting features from returned product...")
        returned_features = extract_features(returned_img)

        # Decode original features from QR
        try:
            original_features = decode_qr(qr_path)
        except Exception as e:
            st.error(f"Error decoding QR: {e}")
            original_features = None

        if original_features is not None:
            # Verify similarity
            result, score = verify_returned_product(original_features, returned_features)
            st.write(f"### 🧠 Verification Result: **{result}**")
            st.write(f"🔢 Similarity Score: `{score:.2f}`")
        else:
            st.warning("⚠️ Could not verify. QR decoding failed.")
