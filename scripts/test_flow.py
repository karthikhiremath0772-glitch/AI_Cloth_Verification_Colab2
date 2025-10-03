import os
from PIL import Image
import numpy as np
from scripts.ai_feature_extractor import extract_features
from scripts.generate_qr import generate_qr_for_product
from scripts.verify_return import verify_return

# Paths (update if your structure is different)
products_folder = os.path.join(os.getcwd(), 'data', 'products')
returns_folder = os.path.join(os.getcwd(), 'data', 'returns')

# ===== Step 1: Add a new product =====
product_id = 'sample_product'  # You can change this name

# Upload your product image to data/products/ folder manually
img_path = os.path.join(products_folder, f'{product_id}.jpg')
img = Image.open(img_path)

# Extract features
features = extract_features(img)
np.save(os.path.join(products_folder, f"{product_id}_features.npy"), features)

# Generate QR
qr_path = generate_qr_for_product(product_id, products_folder)

print("✅ Product added")
print("Features saved at:", os.path.join(products_folder, f"{product_id}_features.npy"))
print("QR code saved at:", qr_path)

# ===== Step 2: Verify returned product =====
# Upload returned image manually to data/returns/ folder
return_image_path = os.path.join(returns_folder, f"{product_id}_return.jpg")

is_verified, similarity = verify_return(return_image_path, products_folder)
print("\n✅ Returned product verification")
print(f"Verified: {is_verified}")
print(f"Similarity Score: {similarity:.2f}")
