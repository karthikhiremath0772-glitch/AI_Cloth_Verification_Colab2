# Use PyTorch ResNet18 for feature extraction
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Load pre-trained model
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()  # Set to evaluation mode

# Remove the final classification layer
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Transform function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img: Image.Image):
    """
    Extract deep features from image using ResNet18
    """
    img_t = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = feature_extractor(img_t)
    features = features.flatten().numpy()
    return features
