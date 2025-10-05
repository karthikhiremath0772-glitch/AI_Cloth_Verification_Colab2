# scripts/decode_qr.py
import cv2
import base64
import numpy as np
from typing import Optional

def decode_qr_to_features(image_path: str) -> Optional[np.ndarray]:
    """
    Decodes a QR code image and extracts embedded feature data (base64-encoded).

    Args:
        image_path (str): Path to the QR image.

    Returns:
        np.ndarray or None: Decoded 512-dimensional feature vector, or None if decoding fails.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None

    # Initialize OpenCV QRCode detector
    detector = cv2.QRCodeDetector()

    # Detect and decode the QR
    data, bbox, _ = detector.detectAndDecode(img)

    if not data:
        print("⚠️ No QR code detected.")
        return None

    try:
        # Decode base64 -> float32 vector
        decoded_bytes = base64.b64decode(data)
        features = np.frombuffer(decoded_bytes, dtype=np.float32)

        # Confirm expected shape
        if features.size == 512:
            print("✅ Successfully decoded 512-D features from QR")
            return features
        else:
            print(f"⚠️ Decoded vector length = {features.size} (expected 512)")
            return features
    except Exception as e:
        print(f"❌ Error decoding QR data: {e}")
        return None
