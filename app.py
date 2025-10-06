import streamlit as st
from PIL import Image
import numpy as np
import os

from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.decode_qr import decode_qr
from scripts.verify_return import verify_returned_product

# Streamlit app title
st.title("✅ AI Cloth Verification")

# --- Step 1: Upload Cloth Image ---
uploaded_file = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)

    # --- Step 2: Extract Features ---
    st.write("Processing the image with AI...")
    features = extract_features(img)
    if features is None:
        st.error("❌ Failed to extract features.")
    else:
        st.success("Features extracted successfully ✅")

        # --- Step 3: Generate QR ---
        st.subheader("Generate Product QR Code")
        qr_img = generate_qr_for_product(features)

        if qr_img is not None:
            qr_path = "qr_code.png"
            qr_img.save(qr_path)
            st.image(np.array(qr_img), caption="QR Code", width=250)
            st.success("QR code generated ✅")
        else:
            st.error("❌ QR generation failed. Invalid image format.")
            qr_path = None

        # --- Step 4: Verify Returned Product ---
        st.subheader("Verify Returned Product")
        returned_file = st.file_uploader("Upload returned product image", type=["jpg", "jpeg", "png"], key="return_upload")

        if returned_file:
            returned_img = Image.open(returned_file)
            st.image(returned_img, caption="Returned Product", width=250)

            st.write("Extracting features from returned product...")
            returned_features = extract_features(returned_img)

            if qr_path and os.path.exists(qr_path):
                try:
                    original_features = decode_qr(qr_path)
                    if original_features is None:
                        st.warning("⚠️ Could not decode features from QR code.")
                    else:
                        result, score = verify_returned_product(original_features, returned_features)
                        st.write(f"Verification Result: **{result}**")
                        st.write(f"Similarity Score: {score:.2f}")
                except Exception as e:
                    st.error(f"⚠️ Could not verify. QR decoding failed.\nError: {e}")
            else:
                st.warning("⚠️ QR code not found or not generated.")
