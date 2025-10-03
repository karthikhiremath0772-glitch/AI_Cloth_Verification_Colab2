import streamlit as st
import os
import numpy as np
from PIL import Image
import qrcode
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import json
from datetime import datetime

# ===============================
# Paths
# ===============================
BASE_DIR = "/content/drive/MyDrive/AI_Cloth_Verification_Colab/data"
PRODUCTS_FOLDER = os.path.join(BASE_DIR, "products")
RETURNS_FOLDER = os.path.join(BASE_DIR, "returns")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

os.makedirs(PRODUCTS_FOLDER, exist_ok=True)
os.makedirs(RETURNS_FOLDER, exist_ok=True)

# ===============================
# History Manager
# ===============================
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"products": [], "verifications": []}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

history = load_history()

# ===============================
# Feature Extractor (ResNet18)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(image: Image.Image):
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_t).cpu().numpy().flatten()
    return features

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ===============================
# Save product (features + QR)
# ===============================
def save_product(uploaded_file, product_id):
    img = Image.open(uploaded_file).convert("RGB")

    features = extract_features(img)
    feature_path = os.path.join(PRODUCTS_FOLDER, f"{product_id}_features.npy")
    np.save(feature_path, features)

    qr = qrcode.make(product_id)
    qr_path = os.path.join(PRODUCTS_FOLDER, f"{product_id}_qr.png")
    qr.save(qr_path)

    # Save to history
    history["products"].append({
        "product_id": product_id,
        "feature_file": feature_path,
        "qr_file": qr_path,
        "uploaded_at": str(datetime.now())
    })
    save_history(history)

    return feature_path, qr_path

# ===============================
# Verify return
# ===============================
def verify_return(returned_file, product_id, threshold=0.75):
    img = Image.open(returned_file).convert("RGB")
    returned_features = extract_features(img)

    feature_file = os.path.join(PRODUCTS_FOLDER, f"{product_id}_features.npy")
    if not os.path.exists(feature_file):
        return False, 0.0

    product_features = np.load(feature_file)
    similarity = cosine_similarity(returned_features, product_features)

    is_verified = similarity >= threshold

    # Save verification attempt
    history["verifications"].append({
        "product_id": product_id,
        "returned_file": returned_file,
        "similarity": float(similarity),
        "verified": bool(is_verified),
        "verified_at": str(datetime.now())
    })
    save_history(history)

    return is_verified, similarity

# ===============================
# Decode QR using OpenCV
# ===============================
def decode_qr(qr_file):
    file_bytes = np.frombuffer(qr_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    return data if data else None

# ===============================
# Streamlit UI
# ===============================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["📦 Upload Product", "✅ Verify Return", "📜 History"])

# --- Upload Product Page ---
if page == "📦 Upload Product":
    st.title("📦 Upload New Product")
    uploaded_product = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded_product is not None:
        product_id = os.path.splitext(uploaded_product.name)[0].replace(" ", "_").replace("-", "_")
        feature_path, qr_path = save_product(uploaded_product, product_id)

        st.success(f"✅ Product uploaded and processed: {product_id}")
        st.write(f"Feature file: {feature_path}")
        st.write(f"QR code file: {qr_path}")

        st.image(qr_path, caption="Generated QR Code", use_container_width=True)

        with open(qr_path, "rb") as qr_file:
            st.download_button(
                label="⬇️ Download QR Code",
                data=qr_file,
                file_name=f"{product_id}_qr.png",
                mime="image/png"
            )

# --- Verify Return Page ---
elif page == "✅ Verify Return":
    st.title("✅ Verify Returned Product")

    uploaded_qr = st.file_uploader("Upload QR Code of the product", type=["png", "jpg", "jpeg"], key="qr_upload")
    uploaded_return = st.file_uploader("Upload Returned Product Image", type=["jpg", "jpeg", "png"], key="return_upload")

    if uploaded_qr is not None and uploaded_return is not None:
        product_id = decode_qr(uploaded_qr)

        if product_id:
            st.success(f"✅ QR Code decoded: {product_id}")

            return_id = os.path.splitext(uploaded_return.name)[0].replace(" ", "_").replace("-", "_")
            return_path = os.path.join(RETURNS_FOLDER, f"{return_id}_return.jpg")
            with open(return_path, "wb") as f:
                f.write(uploaded_return.read())

            st.success(f"Returned image saved: {return_path}")

            is_verified, similarity = verify_return(return_path, product_id)

            if is_verified:
                st.success(f"✅ Return Verified! Product matches ID: {product_id}")
            else:
                st.error(f"❌ Return Rejected. Product does not match ID: {product_id}")

            st.write(f"Similarity Score: {similarity:.2f}")
        else:
            st.error("❌ Could not decode QR code")

# --- History Page ---
elif page == "📜 History":
    st.title("📜 Verification & Product History")

    st.subheader("Uploaded Products")
    if history["products"]:
        for prod in history["products"]:
            st.markdown(f"- **{prod['product_id']}** uploaded at {prod['uploaded_at']}")
            st.image(prod["qr_file"], caption=f"QR Code for {prod['product_id']}", width=150)
    else:
        st.info("No products uploaded yet.")

    st.subheader("Verification Attempts")
    if history["verifications"]:
        for v in history["verifications"]:
            result = "✅ Verified" if v["verified"] else "❌ Rejected"
            st.markdown(f"- **{v['product_id']}** | {result} | Similarity: {v['similarity']:.2f} | At: {v['verified_at']}")
            st.image(v["returned_file"], caption=f"Return Attempt for {v['product_id']}", width=150)
    else:
        st.info("No verifications done yet.")
