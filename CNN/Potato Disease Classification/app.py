import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("cnn_model_version_1.keras")

## Function to preprocess the image
def preprocess_image(image):
    ## Convert the uploaded PIL image to an OpenCV image(Numpy array)
    image = np.array(image)

    # Check if the image is valid
    if image is None:
        st.error("Invalid image. please upload a valid image file")
        return None
    

    ## Resize the image to th e required input size (250,250)
    image_resized = cv2.resize(image, (250,250))

    ## Scale the image to the range [0, 1]
    scaled_image = image_resized / 255.0


    ## Expan dimesion to match the input shape of the model
    return np.expand_dims(scaled_image, axis = 0)

# Streamlit app setup
st.title("Potato Disease Classification Using Deep Learning")

# Uplod Image
upload_file = st.file_uploader("Upload an image of a potato leaf showing Early Blight, Late Blight, or Healthy condition.", type = ["jpg", "png", "jpeg"])

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption = "Uploaded Image", use_column_width = True)

    ## Preprocess the image
    preprocessed_image = preprocess_image(image)

    if preprocessed_image is not None:
        cnn_model = model.predict(preprocessed_image)
        predicted_class = np.argmax(cnn_model)
        confidence = [round(100*(np.max(i)),2) for i in cnn_model]

        ## Map predicted to label
        classes = {0:'Early blight', 1:'Late blight', 2:'Healthy'}
        result = classes[predicted_class]

        ## DIsplay result
        st.write(f"Predition: {result}")
        st.write(f"Confidence: {confidence[0]}%")