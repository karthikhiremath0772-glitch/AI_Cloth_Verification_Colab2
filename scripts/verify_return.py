import numpy as np

def verify_returned_product(original_features: np.ndarray, returned_features: np.ndarray, threshold: float = 0.9):
    """
    Compare original product features with returned product features.

    Args:
        original_features (np.ndarray): Features stored in QR
        returned_features (np.ndarray): Features extracted from returned image
        threshold (float): Cosine similarity threshold

    Returns:
        tuple: (is_verified: bool, similarity_score: float)
    """
    # Cosine similarity
    sim = np.dot(original_features, returned_features) / (
        np.linalg.norm(original_features) * np.linalg.norm(returned_features)
    )
    is_verified = sim >= threshold
    return is_verified, float(sim)
