import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models as models
from pathlib import Path
from PIL import Image

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
artifacts_path = current_dir / "artifacts"

class CarClassifierRestNet(nn.Module):
    def __init__(self, num_classes):
        super(CarClassifierRestNet, self).__init__()
        self.model = models.resnet50(weights = "DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False   ## Freeze all the previous layers

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
trained_model = None
classes = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']
num_classes = len(classes)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_transforms = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.255])
    ])
    image_tensor = image_transforms(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = CarClassifierRestNet(num_classes)
        trained_model.load_state_dict(torch.load(artifacts_path/"saved_model.pth"))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, prediction = torch.max(output, 1)

    return classes[prediction.item()] 

