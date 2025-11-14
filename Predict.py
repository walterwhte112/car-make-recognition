from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# load models
pca = joblib.load("models/car_pca_model.pkl")
knn_pca = joblib.load("models/car_knn_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# load backbone
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Identity()
model = model.to(device)
model.eval()

def predict_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_t).cpu().numpy()
        feat_pca = pca.transform(feat)

    pred_encoded = knn_pca.predict(feat_pca)[0]
    return le.inverse_transform([pred_encoded])[0]
