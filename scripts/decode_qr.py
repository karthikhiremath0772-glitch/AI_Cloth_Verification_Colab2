
"""
decode_qr.py
Utility to decode QR codes from images.

Functions:
- decode_qr_text(path) -> str | None       : returns decoded text or None
- decode_qr_to_features(path) -> np.ndarray|None : tries base64->float32 array (if encoded)
"""
from PIL import Image
from pyzbar.pyzbar import decode
import base64
import numpy as np
from typing import Optional

def decode_qr_text(path: str) -> Optional[str]:
    """Return the decoded QR text from image file path, or None if no QR found."""
    img = Image.open(path).convert('RGB')
    decoded = decode(img)
    if not decoded:
        return None
    return decoded[0].data.decode('utf-8')

def decode_qr_to_features(path: str) -> Optional[np.ndarray]:
    """
    If the QR contains base64-encoded float32 feature bytes, decode to numpy array.
    Returns the numpy array or None if decoding not possible.
    """
    txt = decode_qr_text(path)
    if txt is None:
        return None
    # try base64 -> numpy
    try:
        b = base64.b64decode(txt)
        arr = np.frombuffer(b, dtype=np.float32)
        return arr
    except Exception:
        return None

__all__ = ['decode_qr_text', 'decode_qr_to_features']
