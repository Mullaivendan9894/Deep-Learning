## Step 1 Import Libraries and load the model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model

## Load the IMDB Dataset word index

word_idx = imdb.get_word_index()
reversed_word_index = {index: word for word, index in word_idx.items()}

## Load the pre_trained model with ReLU

model = load_model("simple_rnn_imdb.h5")


## Step 2 Helper Function
## FUnction to decode reviews
def decode_review(encoded_review):
    return " ".join([reversed_word_index.get(i-3, "j")for i in encoded_review])


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_idx.get(word,2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen = 500)
    return padded_review

 ## Streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Entre a movie review to Classify it as positive or Negative")

 ## User Input
user_input = st.text_area("Moview_review")

if st.button("Classify"):
    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)

    if prediction[0][0] > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

     ## Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")

else:
    st.write("Please Entre a Movie review.")




