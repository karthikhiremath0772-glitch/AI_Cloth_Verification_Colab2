import qrcode
import io
import numpy as np
import base64
from PIL import Image

def generate_qr_for_product(features: np.ndarray) -> Image.Image:
    """
    Generates a QR code image for given product features.
    Encodes features using base64 to reduce size.

    Args:
        features (np.ndarray): 1D NumPy array of product features

    Returns:
        PIL.Image: QR code image
    """
    # Convert features to bytes
    features_bytes = features.tobytes()

    # Encode as base64 string
    features_b64 = base64.b64encode(features_bytes).decode('utf-8')

    # Create QR code
    qr = qrcode.QRCode(
        version=None,  # Let qrcode lib choose the minimal version
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(features_b64)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")
    return qr_img
