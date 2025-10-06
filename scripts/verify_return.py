# scripts/verify_return.py
import os
import numpy as np
from PIL import Image
from scripts.ai_feature_extractor import extract_features

def verify_returned_product(return_img_path, product_folder):
    """
    Verify a returned product against stored product features.

    Args:
        return_img_path (str): Path to the returned product image.
        product_folder (str): Path to folder where product features are stored.

    Returns:
        tuple: (is_verified (bool), similarity_score (float))
    """
    # Extract ID from returned image filename
    returned_id = os.path.splitext(os.path.basename(return_img_path))[0].replace('_return', '')

    # Construct path to feature file
    feature_file = os.path.join(product_folder, f"{returned_id}_features.npy")

    if not os.path.exists(feature_file):
        return False, 0.0  # Cannot verify if features missing

    # Load product features
    product_features = np.load(feature_file)

    # Extract features from returned image using AI model
    img = Image.open(return_img_path).convert("RGB")  # Ensure RGB mode
    returned_features = extract_features(np.array(img))

    # Compute cosine similarity
    sim = np.dot(returned_features, product_features.T) / (
        np.linalg.norm(returned_features) * np.linalg.norm(product_features)
    )

    is_verified = sim > 0.9  # Threshold for verification
    return is_verified, float(sim)
