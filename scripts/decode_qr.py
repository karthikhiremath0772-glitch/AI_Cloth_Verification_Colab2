from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np

def decode_qr(qr_path):
    try:
        img = Image.open(qr_path).convert("RGB")
        decoded = decode(img)
        if not decoded:
            print("No QR code found.")
            return None

        data = decoded[0].data.decode("utf-8")
        # Convert string data back to numpy array
        features = np.fromstring(data.replace("[", "").replace("]", ""), sep=",")
        return features
    except Exception as e:
        print("Decode error:", e)
        return None
