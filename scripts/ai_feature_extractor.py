# ai_feature_extractor.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load pretrained ResNet18 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # Remove final classification layer
resnet = resnet.to(device)
resnet.eval()

# Transform function for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image: Image.Image) -> np.ndarray:
    """
    Extract feature vector from an image.
    
    Args:
        image (PIL.Image.Image): Input image of a cloth.
        
    Returns:
        np.ndarray: 512-dimensional feature vector.
    """
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_t)
    return features.cpu().numpy().flatten()
