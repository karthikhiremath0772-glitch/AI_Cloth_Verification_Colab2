# scripts/decode_qr.py

from PIL import Image
from pyzbar.pyzbar import decode
import base64
import numpy as np
from typing import Optional

def decode_qr_to_features(image_path: str) -> Optional[np.ndarray]:
    """
    Decode a QR code from an image and return the features as a numpy array.

    Args:
        image_path (str): Path to the QR code image.

    Returns:
        np.ndarray or None: Decoded feature vector if QR code is valid, else None.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Decode QR code
        qr_codes = decode(img)
        if not qr_codes:
            print("No QR code found in the image.")
            return None
        
        # Take the first QR code found
        qr_data = qr_codes[0].data.decode('utf-8')
        
        # Convert base64 string back to numpy array
        feature_bytes = base64.b64decode(qr_data)
        features = np.frombuffer(feature_bytes, dtype=np.float32)
        
        return features
    
    except Exception as e:
        print(f"Error decoding QR code: {e}")
        return None
