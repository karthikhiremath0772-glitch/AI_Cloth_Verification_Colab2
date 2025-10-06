# scripts/decode_qr.py
import cv2
import base64
import numpy as np
from typing import Optional

def decode_qr(image_path: str) -> Optional[np.ndarray]:
    """
    Decodes a QR code from an image file and returns the decoded features (numpy array).
    Returns None if decoding fails.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("⚠️ Failed to read image for decoding.")
            return None

        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img)

        if not data:
            print("⚠️ No QR code detected.")
            return None

        # Convert from base64 back to numpy array
        decoded_bytes = base64.b64decode(data)
        arr = np.frombuffer(decoded_bytes, dtype=np.float32)
        return arr

    except Exception as e:
        print(f"QR Decoding Error: {e}")
        return None
