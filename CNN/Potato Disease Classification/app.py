import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🥔",
    layout="wide"
)

# --- Style Constants ---
COLORS = {
    "Healthy": "#4CAF50",  # Green
    "Early blight": "#FF9800",  # Orange
    "Late blight": "#F44336"  # Red
}

# --- Model Loading ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
artifacts_path = current_dir / "artifacts"
model = load_model(artifacts_path / "cnn_model_version_1.keras")

# --- Image Processing ---
def preprocess_image(image):
    image = np.array(image)
    if image is None:
        st.error("Invalid image. Please upload a valid image file")
        return None
    image_resized = cv2.resize(image, (250, 250))
    scaled_image = image_resized / 255.0
    return np.expand_dims(scaled_image, axis=0)

# --- UI Components ---
def display_result(prediction, probabilities):
    st.subheader("Diagnosis Result")
    
    # Color-coded result badge
    color = COLORS.get(prediction, "#2196F3")
    st.markdown(
        f"""<div style="background-color:{color};padding:10px;border-radius:5px;color:white;text-align:center">
        <h3>{prediction}</h3>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Probability visualization
    st.subheader("Confidence Levels")
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
    
    st.pyplot(fig)

# --- Main App ---
st.title("🥔 Potato Disease Classifier")
st.markdown("""
    Upload an image of a potato leaf to detect Early Blight, Late Blight, or Healthy conditions.
    The model achieves **95.8% accuracy** on test data.
""")

with st.expander("ℹ️ About this tool"):
    st.write("""
    This deep learning system helps farmers quickly identify potato diseases for early intervention.
    - **Early Blight**: Orange-brown lesions with concentric rings
    - **Late Blight**: Water-soaked lesions with white mold
    - **Healthy**: Uniform green color with no spots
    """)

# --- File Upload ---
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear photo of a potato leaf"
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True,
            output_format="PNG"
        )
    
    with st.spinner("Analyzing the leaf..."):
        preprocessed_image = preprocess_image(image)
        
        if preprocessed_image is not None:
            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction)
            
            classes = {
                0: "Early blight",
                1: "Late blight", 
                2: "Healthy"
            }
            
            probabilities = {
                "Early blight": prediction[0][0],
                "Late blight": prediction[0][1],
                "Healthy": prediction[0][2]
            }
            
            result = classes[predicted_class]
            
            with col2:
                display_result(result, probabilities)
                
                # Additional recommendations
                if result == "Healthy":
                    st.success("✅ No action needed - plant appears healthy!")
                elif result == "Early blight":
                    st.warning("⚠️ Recommended: Apply copper-based fungicides and remove affected leaves")
                else:
                    st.error("🚨 Urgent: Isolate plant and apply appropriate fungicide immediately")

# --- Footer ---
st.markdown("---")
st.caption("""
    *Note: This tool provides preliminary diagnosis only. \
    For critical decisions, consult with agricultural experts.*
""")