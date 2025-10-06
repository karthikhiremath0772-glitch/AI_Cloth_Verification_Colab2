import qrcode
from PIL import Image
import io
import numpy as np

def generate_qr_for_product(features):
    try:
        if isinstance(features, np.ndarray):
            data_str = np.array2string(features.flatten(), precision=4, separator=",")
        else:
            data_str = str(features)

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data_str)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        return qr_img
    except Exception as e:
        print("QR Generation Error:", e)
        return None
