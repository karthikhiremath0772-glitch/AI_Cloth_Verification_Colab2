import qrcode
import os

def generate_qr_for_product(product_id, save_folder):
    """
    Generates a QR code for a product and saves it in the product folder.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    qr_data = f"Product ID: {product_id}"
    qr_img = qrcode.make(qr_data)
    
    qr_path = os.path.join(save_folder, f"{product_id}_qr.png")
    qr_img.save(qr_path)
    
    return qr_path
