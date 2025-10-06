import streamlit as st
from PIL import Image
import numpy as np
import io

from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.decode_qr import decode_qr
from scripts.verify_return import verify_returned_product

st.set_page_config(page_title="AI Cloth Verification", layout="centered")
st.title("✅ AI Cloth Verification")

st.markdown("Upload a product image to generate and verify its QR code.")


# -------------------------------
# STEP 1: Upload Original Cloth Image
# -------------------------------
uploaded_file = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)

    # Extract features
    st.write("Processing the image with AI...")
    try:
        features = extract_features(img)
        st.success("Features extracted successfully ✅")
    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        st.stop()

    # Generate QR code
    st.subheader("Generate Product QR Code")
    try:
        qr_img = generate_qr_for_product(features)

        if qr_img is not None:
            qr_path = "qr_code.png"
            qr_img.save(qr_path)
            st.image(qr_img, caption="Generated QR Code", width=250)
            st.success("QR code generated ✅")
        else:
            st.error("❌ QR generation failed. Invalid image format.")
            st.stop()
    except Exception as e:
        st.error(f"❌ QR generation failed: {e}")
        st.stop()

    # -------------------------------
    # STEP 2: Verify Returned Product
    # -------------------------------
    st.subheader("Verify Returned Product")
    returned_file = st.file_uploader("Upload returned product image", type=["jpg", "jpeg", "png"])

    if returned_file:
        returned_img = Image.open(returned_file).convert("RGB")
        st.image(returned_img, caption="Returned Product", width=250)
        st.write("Extracting features from returned product...")

        try:
            returned_features = extract_features(returned_img)
        except Exception as e:
            st.error(f"❌ Failed to extract returned features: {e}")
            st.stop()

        # Decode original QR features
        try:
            original_features = decode_qr(qr_path)
        except Exception as e:
            st.error(f"Error decoding QR: {e}")
            st.stop()

        if original_features is None:
            st.warning("⚠️ QR code not found or not generated.")
            st.stop()

        # Verify
        try:
            result, score = verify_returned_product(original_features, returned_features)
            st.write(f"### Verification Result: **{result}**")
            st.write(f"Similarity Score: `{score:.2f}`")
        except Exception as e:
            st.error(f"⚠️ Could not verify. {e}")
