import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
def load_data():
    dataset = pd.read_csv("twitter.csv")
    dataset["labels"] = dataset["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
    return dataset[["tweet", "labels"]]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Train model
def train_model(data):
    X = data["tweet"].apply(preprocess_text)
    y = data["labels"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)

# Streamlit UI
st.title("Hate Speech Detection")

data = load_data()
model, accuracy, report = train_model(data)

st.write(f"### Model Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(report)

text_input = st.text_area("Enter a tweet to analyze:")
if st.button("Analyze"):
    prediction = model.predict([preprocess_text(text_input)])[0]
    st.write(f"### Prediction: {prediction}")
