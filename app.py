import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = text.replace("subject:", "")
    text = re.sub(r'[^a-z\s]', '', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

model = joblib.load("spam_model.pkl")

st.title("Email Spam Detector")
email = st.text_area("Paste an email here:", height=200)

if st.button("Check"):
    if email.strip():
        result = model.predict([clean_text(email)])[0]
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
    else:
        st.warning("Please paste some text first.")