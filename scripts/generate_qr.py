# scripts/generate_qr.py
import qrcode
from PIL import Image
import io
import base64
import numpy as np

def generate_qr_for_product(features):
    """
    Generates a QR code from the product's extracted features (list or np.array).
    Returns a PIL Image object of the QR code.
    """
    # Convert numpy array to base64 for compact encoding
    if isinstance(features, np.ndarray):
        features_str = base64.b64encode(features.tobytes()).decode('utf-8')
    else:
        features_str = str(features)

    # Generate the QR code
    qr = qrcode.QRCode(
        version=1,
        box_size=10,
        border=4
    )
    qr.add_data(features_str)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Ensure the result is a proper PIL image
    if not isinstance(img, Image.Image):
        img = img.convert("RGB")

    return img
