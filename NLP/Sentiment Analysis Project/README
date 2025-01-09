# Sentiment Analysis Project (Natural Language Processing)

## Overview
This project focuses on analyzing the sentiment of tweets using Natural Language Processing (NLP) techniques and Deep Learning models. The objective is to classify the sentiment of tweets into three categories: **Negative**, **Neutral**, and **Positive**.

---

## Steps and Workflow

### 1. Data Loading and Initial Analysis
- **Load the dataset** from a CSV file using `pandas`.
- **Retain relevant columns**: `candidate`, `sentiment`, and `text`.
- Perform an initial inspection of the dataset:
  - Check column data types.
  - Analyze the distribution of sentiment values.

---

### 2. Data Preprocessing

#### Text Cleaning Steps:
1. **HTML Stripping**: Remove HTML tags from the text.
2. **Expand Contractions**: Convert contractions like "don't" to "do not".
3. **Remove Punctuation and Numbers**: Keep only alphabetic characters.
4. **Tokenization**: Split the text into individual words.
5. **Stopword Removal**: Remove common words like "is" and "the" that don't add significant meaning.
6. **Lemmatization**: Reduce words to their base form.

#### Mapping Sentiments and Encoding:
- Map sentiment values (`Negative`, `Neutral`, `Positive`) to numerical values (`0`, `1`, `2`).
- Convert sentiment labels into one-hot encoding for training the model.

---

### 3. Text Vectorization
- **Tokenizer**: Convert words into sequences of integers.
- **Padding**: Ensure all sequences have the same length.

---

### 4. Model Development

#### Model Architecture:
1. **Embedding Layer**: Converts words into dense vectors.
2. **SpatialDropout1D**: Reduces overfitting by randomly setting input elements to zero.
3. **LSTM Layers**: Capture sequence dependencies in text data.
4. **Dense Layer**: Outputs class probabilities.

#### Optimizations:
- **Callbacks**:
  - `EarlyStopping`: Prevent overfitting by stopping training early if validation loss stops improving.
  - `ReduceLROnPlateau`: Dynamically adjust the learning rate based on validation loss.
- **Regularization**:
  - Added L2 regularization to LSTM layers.
  - Incorporated dropout layers to improve generalization.

---

### 5. Model Evaluation
- **Evaluate the trained model** using test data.
- **Generate predictions** and analyze model performance using metrics like precision, recall, and F1-score.
