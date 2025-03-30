import streamlit as st
import joblib
import helper
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Emotion Detection Using Text")

# load the model
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

user_input = st.text_area("Enter the text")
if st.button("Predict"):
    user_input = helper.clean_text(user_input)
    user_input = [helper.lemmatization(user_input)]
    user_input = vectorizer.transform(user_input)
    result = model.predict(user_input)
    st.markdown(f'## Emotion : {result[0]}')