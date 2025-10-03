import os
import numpy as np

def verify_return(return_img_path, product_folder):
    """
    Verify a returned product against stored product features.
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
    from scripts.ai_feature_extractor import extract_features
    from PIL import Image
    img = Image.open(return_img_path)
    returned_features = extract_features(img)
    
    # Compute cosine similarity
    sim = np.dot(returned_features, product_features.T) / (
        np.linalg.norm(returned_features) * np.linalg.norm(product_features)
    )
    
    is_verified = sim > 0.9  # Threshold
    return is_verified, float(sim)