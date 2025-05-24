# 🥔 Potato Disease Classification using Deep Learning

[![GitHub repo](https://img.shields.io/badge/Repo-Potato_Disease_Classification-blue?logo=github)](https://github.com/Mullaivendan9894/Deep-Learning/tree/master/potato_disease_classification)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://potato-disease-classification-dl.streamlit.app/)


A production-ready CNN system that classifies potato leaf diseases with **95.8% accuracy**, deployed as a farmer-friendly web application.


## 🌟 Key Features
- **High-Precision CNN**: 6-layer architecture achieving 99% precision on diseased leaves
- **Field-Ready Performance**: <1s inference speed on low-end devices
- **Robust Training**: Advanced augmentation (rotation/flipping/zooming)
- **Intuitive Interface**: Color-coded results with confidence scoring
- **Real-World Ready**: Handles field conditions (shadows, dirt, etc.)

## 🧰 Tech Stack
| Component           | Technology Stack             |
|---------------------|------------------------------|
| **Framework**       | TensorFlow/Keras             |
| **Frontend**        | Streamlit                    | 
| **Image Processing**| OpenCV, PIL                  |
| **Optimization**    | Adam, Batch Normalization    |
| **Deployment**      | Streamlit Sharing            |

## 📊 Performance Metrics
| Metric                  | Value       | Significance                     |
|-------------------------|-------------|----------------------------------|
| **Test Accuracy**       | 95.8%       | Overall model correctness        |
| **Precision (Diseased)**| 99%         | Few false positives              | 
| **Recall (Healthy)**    | 93%         | Reliable healthy identification  |
| **Inference Speed**     | <1s         | Real-time field use              |

## 🗂 Dataset
[PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) from Kaggle:
- **Total Images**: 2,152 high-quality leaf images
- **Classes**: 
  - Early Blight (1,000 images)
  - Late Blight (1,000 images) 
  - Healthy (152 images)
- **Augmented**: 5x synthetic samples via transformations



## 📦 Project Structure
<pre> Potato_disease_classification/
├── artifacts/                # Trained model
├── app.py                    # Streamlit application
├── prediction_helper.py      # Prediction functions
├── README.md                 # This file
├── requirements.txt          # Dependencies </pre>

## 🚀 Quick Start
### Install dependencies:
<pre> pip install -r requirements.txt </pre>

### Run the Streamlit app:
<pre> streamlit run app.py </pre>
