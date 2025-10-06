import qrcode
import numpy as np
from PIL import Image

def generate_qr_for_product(features):
    """
    Generate a QR code from the feature vector
    """
    # Convert numpy array to string
    features_str = ','.join(map(str, features.tolist()))
    
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(features_str)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    return img
