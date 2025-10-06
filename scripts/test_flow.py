from PIL import Image
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.decode_qr import decode_qr
from scripts.verify_return import verify_returned_product
import os

# --- Test Images ---
original_img_path = "test_images/original.jpg"  # put a sample image here
returned_img_path = "test_images/returned.jpg"  # put a returned product image here
qr_path = "test_qr.png"

# Load original image
original_img = Image.open(original_img_path).convert("RGB")

# Extract features
original_features = extract_features(original_img)
print("Original features extracted ✅")

# Generate QR
qr_img = generate_qr_for_product(original_features)
qr_img.save(qr_path)
print(f"QR code saved as {qr_path} ✅")

# Decode QR
decoded_features = decode_qr(qr_path)
if decoded_features is not None:
    print("QR code decoded successfully ✅")
else:
    print("⚠️ QR decoding failed")

# Load returned product image
returned_img = Image.open(returned_img_path).convert("RGB")
returned_features = extract_features(returned_img)
print("Returned product features extracted ✅")

# Verify
result, score = verify_returned_product(decoded_features, returned_features)
print(f"Verification Result: {result}")
print(f"Similarity Score: {score:.2f}")

# Cleanup
if os.path.exists(qr_path):
    os.remove(qr_path)
