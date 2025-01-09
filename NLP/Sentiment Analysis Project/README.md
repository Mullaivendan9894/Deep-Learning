# Sentiment Analysis Project (Natural Language Processing)

## Overview

This project focuses on analyzing the sentiment of tweets using Natural Language Processing (NLP) techniques and Deep Learning models. The objective is to classify the sentiment of tweets into three categories: **Negative**, **Neutral**, and **Positive**.

## Steps and Workflow

### 1. Data Loading and Initial Analysis
- Load the dataset from a CSV file using `pandas`.
- Retain the relevant columns: `candidate`, `sentiment`, and `text`.
- Perform an initial inspection of the dataset:
  - Check column data types.
  - Analyze the distribution of sentiment values.

### 2. Preprocessing

#### a. Text Cleaning
- Strip HTML tags using `BeautifulSoup`.
- Replace contractions using the `contractions` library.
- Remove retweets, non-alphabetical characters, and convert text to lowercase.
- Remove numbers.

#### b. Tokenization, Stopword Removal, and Lemmatization
- Tokenize text using `nltk`.
- Remove stopwords using the NLTK corpus.
- Lemmatize words using `WordNetLemmatizer`.

### 3. Label Encoding
- Map sentiment labels (`Negative`, `Neutral`, `Positive`) to numeric values (`0`, `1`, `2`).
- One-hot encode the numeric labels.

### 4. Tokenization and Padding
- Tokenize preprocessed text using `Tokenizer` from Keras.
- Convert text to sequences of integers.
- Pad sequences to ensure uniform length.

### 5. Train-Test Split
- Split the data into training and testing sets (80% train, 20% test).

### 6. Word2Vec Embedding
- Train a Word2Vec model on tokenized text to create word embeddings.
- Map the embeddings to a matrix to initialize the Embedding layer.

### 7. Model Architecture
- Use a sequential model with the following layers:
  1. **Embedding layer** (pre-trained Word2Vec embeddings, non-trainable).
  2. **SpatialDropout1D layer**.
  3. **LSTM layers**.
  4. **Dense output layer** with softmax activation.

### 8. Training and Validation
- Compile the model with Adam optimizer and categorical crossentropy loss.
- Train the model using the training data and validate it with a split of the training set.

### 9. Evaluation
- Plot training vs validation accuracy and loss curves.
- Evaluate the model on the test set to calculate loss and accuracy.

### 10. Predictions and Classification Report
- Make predictions on the test set.
- Generate a classification report using `sklearn`.
