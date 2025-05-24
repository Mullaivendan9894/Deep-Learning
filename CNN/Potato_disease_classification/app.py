import streamlit as st
from PIL import Image
from prediction_helper import (
    load_potato_model,
    predict_potato_disease,
    plot_probabilities,
    get_recommendation
)

# Configure app
st.set_page_config(
    page_title="🥔 Potato Disease Detector",
    page_icon="🥔",
    layout="wide"
)
# Debugging info for Streamlit Cloud (optional: remove in production)
st.write("📁 Working directory:", os.getcwd())
st.write("📄 Files here:", os.listdir())
artifacts_path = Path("artifacts")
if artifacts_path.exists():
    st.write("📦 Artifacts folder contents:", os.listdir(artifacts_path))
else:
    st.warning("⚠️ 'artifacts' folder not found!")

try:
    import cv2
    st.success(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    st.error(f"Failed to import OpenCV: {e}")

# Initialize model (cached)
@st.cache_resource
def load_model():
    return load_potato_model()



# UI Components
def display_result(prediction, col):
    """Display all results in specified column"""
    with col:
        # Result badge
        st.subheader("Diagnosis Result")
        st.markdown(
            f"""<div style="background-color:{prediction['color']};padding:12px;
                 border-radius:8px;color:white;text-align:center;margin-bottom:20px">
                 <h3 style="margin:0;">{prediction['class']}</h3>
            </div>""",
            unsafe_allow_html=True
        )
        
        # Probability plot
        st.subheader("Confidence Levels")
        st.pyplot(plot_probabilities(prediction['probabilities']))
        
        # Recommendation
        message, message_type = get_recommendation(prediction['class'])
        if message_type == "success":
            st.success(message)
        elif message_type == "warning":
            st.warning(message)
        else:
            st.error(message)

# Main App
model = load_model()

st.title("🥔 Potato Disease Classifier")
st.markdown("Upload an image of a potato leaf for disease detection")

with st.expander("ℹ️ About this tool"):
    st.write("""
    - **Early Blight**: Circular brown lesions with target-like rings
    - **Late Blight**: Irregular water-soaked lesions with white mold
    - **Healthy**: Uniform green coloration
    """)

# File Upload and Results
col1, col2 = st.columns([1, 1])
uploaded_file = col1.file_uploader(
    "Choose leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    col1.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        try:
            prediction = predict_potato_disease(model, image)
            display_result(prediction, col2)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("For diagnosis confirmation, consult a plant pathologist")
