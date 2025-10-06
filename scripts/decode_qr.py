import cv2
import numpy as np
import base64

def decode_qr(qr_img_path: str) -> np.ndarray:
    """
    Decode a QR code image back into product features.

    Args:
        qr_img_path (str): Path to QR code image

    Returns:
        np.ndarray: Original feature vector
    """
    # Load QR code image
    img = cv2.imread(qr_img_path)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)
    if not data:
        raise ValueError("QR code could not be decoded")
    
    # Decode base64 to bytes
    features_bytes = base64.b64decode(data)
    
    # Convert bytes back to NumPy array (float32)
    features = np.frombuffer(features_bytes, dtype=np.float32)
    return features
