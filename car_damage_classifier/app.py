import streamlit as st
from PIL import Image
from prediction_helper import predict
import os

# Set page configuration
st.set_page_config(
    page_title="Car Damage Classifier",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stFileUploader>div>div>div>div {
        color: #4CAF50;
    }
    .title {
        color: #2c3e50;
        text-align: center;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        border-radius: 5px;
        background-color: #e8f5e9;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='title'>🚗 Car Damage Classifier</h1>", unsafe_allow_html=True)
st.markdown("""
    This app classifies car damage into one of the following categories:
    - Front Breakage
    - Front Crushed
    - Front Normal
    - Rear Breakage
    - Rear Crushed
    - Rear Normal
            
    Upload an image of a car to get started!
    """)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width = True)
    
    # Create a temporary file to save the uploaded image
    temp_file_path = "temp_uploaded_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Make prediction when button is clicked
    if st.button("Classify Damage"):
        with st.spinner('Analyzing the image...'):
            # Call your prediction function
            prediction = predict(temp_file_path)
            
            # Display the result
            st.markdown(f"<div class='result'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        
        # Remove the temporary file
        os.remove(temp_file_path)

# Add some footer information
st.markdown("---")
st.markdown("""
    *Note: This is a Deep learning model for demonstration purposes. 
    The accuracy may vary based on image quality and angle.*
    """)