# decode_qr.py
import cv2
import numpy as np

def decode_qr(image_path):
    """Decode QR code from the given image path using OpenCV."""
    try:
        img = cv2.imread(image_path)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img)
        if data:
            print("Decoded data:", data)
            return data
        else:
            print("⚠️ No QR code found in the image.")
            return None
    except Exception as e:
        print("⚠️ Error decoding QR:", e)
        return None

