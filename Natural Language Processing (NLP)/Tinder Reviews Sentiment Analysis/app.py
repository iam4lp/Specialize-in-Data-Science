import streamlit as st
import joblib
import re

# Load
model = joblib.load("src/sentiment_model.pkl")
vectorizer = joblib.load("src/tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("🔥 Tinder Reviews Sentiment Analysis")
st.write("Analyze sentiment of Tinder app reviews")

user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "Positive":
            st.success("😊 Positive Review")
        elif prediction == "Negative":
            st.error("😡 Negative Review")
        else:
            st.warning("😐 Neutral Review")