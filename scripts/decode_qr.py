import cv2

def decode_qr(image_path):
    """
    Decode QR code using OpenCV
    """
    try:
        detector = cv2.QRCodeDetector()
        img = cv2.imread(image_path)
        if img is None:
            print("⚠️ Image not found or unreadable:", image_path)
            return None

        data, points, _ = detector.detectAndDecode(img)
        if data:
            # Convert back to numpy array
            features_list = list(map(float, data.split(',')))
            return features_list
        else:
            print("⚠️ No QR code found in the image.")
            return None
    except Exception as e:
        print("⚠️ Error decoding QR:", e)
        return None
