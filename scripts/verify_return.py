import numpy as np

def verify_returned_product(original_features, returned_features):
    """
    Compare original and returned features using cosine similarity
    """
    if original_features is None or returned_features is None:
        return "Failed", 0.0

    # Convert to numpy arrays
    orig = np.array(original_features)
    ret = np.array(returned_features)

    # Normalize
    orig_norm = orig / (np.linalg.norm(orig) + 1e-10)
    ret_norm = ret / (np.linalg.norm(ret) + 1e-10)

    # Cosine similarity
    similarity = float(np.dot(orig_norm, ret_norm))
    result = "Match" if similarity > 0.95 else "Mismatch"
    return result, similarity
