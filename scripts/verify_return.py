from scipy.spatial.distance import cosine

def verify_returned_product(original_features, returned_features, threshold=0.85):
    # Cosine similarity
    score = 1 - cosine(original_features, returned_features)
    result = "✅ Match" if score >= threshold else "❌ Not Match"
    return result, score
