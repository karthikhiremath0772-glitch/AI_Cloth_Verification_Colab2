# scripts/generate_qr.py
import qrcode
from PIL import Image
import base64
import numpy as np
import io

def generate_qr_for_product(features):
    """
    Generate a QR code encoding the product features as base64.
    Returns a valid PIL Image.
    """
    try:
        # Convert features (numpy array) to base64
        if isinstance(features, np.ndarray):
            features_str = base64.b64encode(features.tobytes()).decode("utf-8")
        else:
            features_str = str(features)

        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            box_size=10,
            border=4
        )
        qr.add_data(features_str)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Ensure output is a proper PIL Image
        if not isinstance(img, Image.Image):
            img = img.convert("RGB")

        return img

    except Exception as e:
        print(f"QR Generation Error: {e}")
        return None
