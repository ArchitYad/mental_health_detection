import streamlit as st
import numpy as np
import tensorflow as tf
import re
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

with open('label_map.json', 'r') as f:
    label_map = json.load(f)

label_decoder = {v: k for k, v in label_map.items()}

model = load_model('model.h5', custom_objects={'AttentionLayer': AttentionLayer})

# Assuming you saved and reloaded tokenizer — otherwise re-fit on same dataset
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])  # or load tokenizer

max_len = 30

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("Mental Health Text Classifier")

user_input = st.text_area("Enter your mental health-related text")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded)
    predicted_label_index = np.argmax(prediction)
    predicted_label = label_decoder[str(predicted_label_index)]  # str because keys are string in JSON

    st.write(f"### Predicted Label: `{predicted_label}`")
