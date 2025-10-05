# scripts/decode_qr.py
"""
Safe QR decoder — won't crash even if OpenCV or pyzbar aren't available.
"""

import base64
import numpy as np
from typing import Optional
from PIL import Image

# Try importing optional libraries
try:
    import cv2
except ImportError:
    cv2 = None
    print("⚠️ OpenCV (cv2) not installed — QR decoding will be disabled.")

try:
    from pyzbar.pyzbar import decode
except ImportError:
    decode = None
    print("⚠️ pyzbar not installed — QR decoding will be disabled.")


def decode_qr_to_features(image_path: str) -> Optional[np.ndarray]:
    """
    Attempts to decode QR code and extract feature data from it.
    Returns numpy array or None if not possible.
    """
    # Check dependencies
    if cv2 is None or decode is None:
        print("❌ Missing dependencies. Cannot decode QR in this environment.")
        return None

    try:
        img = cv2.imread(image_path)
        qr_data = decode(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

        if not qr_data:
            print("⚠️ No QR code detected in the image.")
            return None

        qr_text = qr_data[0].data.decode("utf-8")
        arr = np.frombuffer(base64.b64decode(qr_text), dtype=np.float32)
        return arr

    except Exception as e:
        print(f"❌ QR decoding failed: {e}")
        return None
