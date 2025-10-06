import cv2
import numpy as np

def decode_qr_opencv(qr_path):
    # Load image
    img = cv2.imread(qr_path)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if not data:
        raise ValueError("QR code not found")
    # Convert string back to features (assume comma-separated floats)
    features = np.array([float(x) for x in data.split(',')])
    return features
