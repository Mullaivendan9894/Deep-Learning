import numpy as np
import cv2
from tensorflow.keras.models import load_model
from pathlib import Path
import matplotlib.pyplot as plt

import os

print("Current path:", artifacts_path)
print("Files in artifacts folder:", os.listdir(artifacts_path))


# Constants
CLASSES = {
    0: "Early blight",
    1: "Late blight", 
    2: "Healthy"
}

COLORS = {
    "Healthy": "#4CAF50",
    "Early blight": "#FF9800",
    "Late blight": "#F44336"
}

# Model Loading
def load_potato_model():
    """Load and cache the trained model"""
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    artifacts_path = current_dir / "artifacts"
    return load_model(artifacts_path / "cnn_model_version_1.keras")

# Image Processing
def preprocess_image(image):
    """Convert PIL image to model input format"""
    image = np.array(image)
    if image is None:
        raise ValueError("Invalid image")
    image_resized = cv2.resize(image, (250, 250))
    scaled_image = image_resized / 255.0
    return np.expand_dims(scaled_image, axis=0)

# Prediction Logic
def predict_potato_disease(model, image):
    """Make prediction on input image"""
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    
    predicted_class = np.argmax(prediction)
    result = CLASSES[predicted_class]
    
    probabilities = {
        cls: float(prediction[0][i]) 
        for i, cls in CLASSES.items()
    }
    
    return {
        "class": result,
        "probabilities": probabilities,
        "color": COLORS[result]
    }

# Visualization
def plot_probabilities(probabilities):
    """Generate probability distribution plot"""
    fig, ax = plt.subplots()
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [COLORS[cls] for cls in classes]
    
    ax.barh(classes, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Disease Probability Distribution")
    
    for i, v in enumerate(values):
        ax.text(v + 0.03, i, f"{v:.1%}", color='black', fontweight='bold')
    
    return fig

# Recommendation Generator
def get_recommendation(prediction_class):
    """Generate treatment recommendations"""
    recommendations = {
        "Healthy": ("✅ No action needed - plant appears healthy!", "success"),
        "Early blight": ("⚠️ Apply copper-based fungicides and remove affected leaves", "warning"),
        "Late blight": ("🚨 Urgent: Isolate plant and apply fungicide immediately", "error")
    }
    return recommendations.get(prediction_class, ("ℹ️ Consult an agricultural expert", "info"))
