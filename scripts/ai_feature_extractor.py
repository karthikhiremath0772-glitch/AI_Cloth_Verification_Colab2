# Simple AI feature extractor using PyTorch pretrained model (CPU only)
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load ResNet18 pretrained (CPU)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # evaluation mode

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img: Image.Image):
    img_t = transform(img).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        features = model(img_t)
    return features.squeeze().tolist()  # return as list of floats
