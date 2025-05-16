import streamlit as st
import numpy as np
import tensorflow as tf
import re
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

with open('label_map.json', 'r') as f:
    label_map = json.load(f)

label_decoder = {v: k for k, v in label_map.items()}

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


model = load_model('model.h5', custom_objects={'AttentionLayer': AttentionLayer})


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
    
st.title("🧠 Mental Health Text Classifier")

user_input = st.text_area("Enter your mental health-related text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=30, padding='post')
        prediction = model.predict(padded)
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_decoder[str(predicted_label_index)]  # label_map keys are strings
        st.success(f"### Predicted Label: `{predicted_label}`")
