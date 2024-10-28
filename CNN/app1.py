import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("Final_model.h5")

# Function to preprocess image
def preprocess_image(image):
    # Convert the uploaded PIL image to an OpenCV image (NumPy array)
    image = np.array(image)
    
    # Check if the image is valid
    if image is None:
        st.error("Invalid image. Please upload a valid image file.")
        return None
    
    # Resize the image to the required input size (150x150)
    image_resized = cv2.resize(image, (150, 150))
    
    # Scale the image to the range [0, 1]
    scaled_image = image_resized / 255.0
    
    # Expand dimensions to match the input shape of the model
    return np.expand_dims(scaled_image, axis=0)  # Shape: (1, 150, 150, 3)

# Streamlit app setup
st.title("Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    if preprocessed_image is not None:
        # Make prediction
        cnn_pred = model.predict(preprocessed_image)
        predicted_class = np.argmax(cnn_pred)
        
        # Map prediction to label
        label_map = {0: "Cat", 1: "Dog"}
        result = label_map[predicted_class]
        
        # Display result
        st.write(f"Prediction: {result}")
