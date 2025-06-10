[![GitHub repo](https://img.shields.io/badge/Repo-Car_Damage_Classifier-blue?logo=github)](https://github.com/Mullaivendan9894/Deep-Learning/tree/master/car_damage_classifier)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://car-damage-classifier.streamlit.app/)

# 🚗💥 Car Damage Classifier

A deep learning application that classifies car damage into six distinct categories using **transfer learning with ResNet50**. This project provides predictions via a user-friendly **Streamlit web interface**.

---

## 📌 Project Overview

This computer vision system automatically classifies car damage into one of the following six categories:

* **Front Breakage**
* **Front Crushed**
* **Front Normal**
* **Rear Breakage**
* **Rear Crushed**
* **Rear Normal**

The system is built using **PyTorch** for model development and **Streamlit** for deployment, offering an intuitive way to get damage predictions.

---

## ✨ Features

* **Model Architecture Exploration**: We tested four different model architectures:
    * Plain CNN
    * CNN with Regularization (BatchNorm + Dropout)
    * EfficientNetB0 (Transfer Learning)
    * **ResNet50 (Transfer Learning - Selected as the Best Model)**
* **High Accuracy**: The final model achieves a **77.9% validation accuracy**.
* **Automated Preprocessing**: Includes fully automated preprocessing and normalization for seamless integration.
* **User-Friendly Interface**: Provides a simple and interactive web interface for making predictions.

---

## 🧠 Model Architecture

The best-performing model leverages **ResNet50** with a strategy of selective fine-tuning. Below is the core architecture:

<pre>
import torch.nn as nn
import torchvision.models as models

class CarClassifierRestNet(nn.Module):
    def __init__(self, num_classes):
        super(CarClassifierRestNet, self).__init__()
        self.model = models.resnet50(weights="DEFAULT")
        
        # Freeze all layers except layer4
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final classifier
        in_features = self.model.fc.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x) </pre>

---

## 📊 Model Comparison

| Model | Epochs | Avg Loss | Validation Accuracy |
| :------------------------ | :----- | :------- | :------------------ |
| Plain CNN | 5 | 0.8685 | ~57.9% |
| CNN + Regularization | 10 | 1.0227 | ~54.3% |
| EfficientNet-B0 (Frozen) | 10 | 0.8811 | ~64.4% |
| **ResNet-50 (Fine-tuned)** | **10** | **0.0788** | **77.9%** |

**✔️ Final model accuracy: Approximately 78%**

---

## 📂 Project Structure
<pre>
├── artifacts/               # Saved models and related files
│   └── saved_model.pth
├── prediction_helper.py     # Inference helper for saved model
├── app.py                   # Streamlit app for live prediction
├── requirements.txt         # Dependencies
└── README.md                # This file
</pre>


## ⚙️ Install Dependencies
<pre> pip install -r requirements.txt </pre>

## 🚀 Run Streamlit App
<pre> streamlit run app.py </pre>
