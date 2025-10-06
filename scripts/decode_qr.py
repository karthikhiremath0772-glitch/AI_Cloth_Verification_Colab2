# scripts/decode_qr.py
import cv2

def decode_qr(image_path):
    """
    Decode QR code using OpenCV (works on Streamlit Cloud, Colab, or local)
    """
    try:
        detector = cv2.QRCodeDetector()
        img = cv2.imread(image_path)
        if img is None:
            print("⚠️ Image not found or unreadable:", image_path)
            return None

        data, points, _ = detector.detectAndDecode(img)
        if data:
            print("✅ QR Code decoded successfully:", data)
            return data
        else:
            print("⚠️ No QR code found in the image.")
            return None
    except Exception as e:
        print("⚠️ Error decoding QR:", e)
        return None
