import qrcode

def generate_qr_for_product(features):
    # Convert features list to string
    features_str = ','.join([str(f) for f in features])
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(features_str)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    return img, features_str
