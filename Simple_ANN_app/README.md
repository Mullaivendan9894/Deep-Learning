# Customer Churn Prediction
## Overview
This project aims to build a customer churn prediction model using machine learning techniques. The model predicts whether a customer is likely to leave (churn) based on their attributes. A Streamlit web application is provided for users to input customer data and receive predictions in real-time.

Technologies Used
Python: The primary programming language for this project.
Pandas: Used for data manipulation and analysis.
NumPy: Used for numerical computations.
Scikit-learn: For data preprocessing, including label encoding and scaling.
TensorFlow: For building and training the deep learning model.
Pickle: For saving and loading the trained model and preprocessing objects.
Streamlit: To create an interactive web application for the model.
Files
model.h5: The trained TensorFlow model for predicting customer churn.
label_encoder_gender.pkl: A Pickle file containing the label encoder for the gender feature.
one_hot_encoder_geo.pkl: A Pickle file containing the one-hot encoder for the geography feature.
standard_scaler.pkl: A Pickle file containing the standard scaler used to standardize the feature values.
app.py: The main Streamlit application that allows users to input customer data and receive predictions.
How It Works
Model Loading:

The pre-trained TensorFlow model is loaded along with the encoders and scaler using the Pickle module.
User Input:

The Streamlit app prompts users to enter customer details, including:
Geography (selected from a dropdown)
Gender (selected from a dropdown)
Age (via a slider)
Balance (numeric input)
Credit Score (numeric input)
Estimated Salary (numeric input)
Tenure (via a slider)
Number of Products (via a slider)
Has Credit Card (binary choice)
Is Active Member (binary choice)
Data Preprocessing:

The user input is transformed into a suitable format for prediction:
Gender is label encoded.
Geography is one-hot encoded.
The entire input dataset is scaled using the standard scaler.
Prediction:

The processed input data is fed into the model to predict the likelihood of customer churn.
The prediction probability is then displayed, indicating whether the customer is likely to churn or not.


Here's a sample README text for your customer churn prediction project. You can modify it as needed to fit your specific context or requirements.

Customer Churn Prediction
Overview
This project aims to build a customer churn prediction model using machine learning techniques. The model predicts whether a customer is likely to leave (churn) based on their attributes. A Streamlit web application is provided for users to input customer data and receive predictions in real-time.

Technologies Used
Python: The primary programming language for this project.
Pandas: Used for data manipulation and analysis.
NumPy: Used for numerical computations.
Scikit-learn: For data preprocessing, including label encoding and scaling.
TensorFlow: For building and training the deep learning model.
Pickle: For saving and loading the trained model and preprocessing objects.
Streamlit: To create an interactive web application for the model.
Files
model.h5: The trained TensorFlow model for predicting customer churn.
label_encoder_gender.pkl: A Pickle file containing the label encoder for the gender feature.
one_hot_encoder_geo.pkl: A Pickle file containing the one-hot encoder for the geography feature.
standard_scaler.pkl: A Pickle file containing the standard scaler used to standardize the feature values.
app.py: The main Streamlit application that allows users to input customer data and receive predictions.
How It Works
Model Loading:

The pre-trained TensorFlow model is loaded along with the encoders and scaler using the Pickle module.
User Input:

The Streamlit app prompts users to enter customer details, including:
Geography (selected from a dropdown)
Gender (selected from a dropdown)
Age (via a slider)
Balance (numeric input)
Credit Score (numeric input)
Estimated Salary (numeric input)
Tenure (via a slider)
Number of Products (via a slider)
Has Credit Card (binary choice)
Is Active Member (binary choice)
Data Preprocessing:

The user input is transformed into a suitable format for prediction:
Gender is label encoded.
Geography is one-hot encoded.
The entire input dataset is scaled using the standard scaler.
Prediction:

The processed input data is fed into the model to predict the likelihood of customer churn.
The prediction probability is then displayed, indicating whether the customer is likely to churn or not.
Running the Application
To run the Streamlit application, follow these steps:

Ensure you have the necessary libraries installed. You can install them using:

bash
Copy code
pip install pandas numpy scikit-learn tensorflow streamlit
Place the model and encoder files in the same directory as app.py.

Run the Streamlit application with the following command:

bash
Copy code
streamlit run app.py
Open the URL provided by Streamlit in your web browser to interact with the application.

Conclusion
This project demonstrates the application of machine learning techniques in predicting customer behavior, specifically churn. By using this model, businesses can take proactive measures to retain their customers, ultimately improving their bottom line.
