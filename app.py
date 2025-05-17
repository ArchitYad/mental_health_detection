import streamlit as st
import numpy as np
import tensorflow as tf
import re
import json
import pickle
import gc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

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

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("🧠 Mental Health Text Classifier")
user_input = st.text_area("Enter your mental health-related text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Loading model and tokenizer..."):
            try:
                # Load tokenizer and label map
                with open('tokenizer.pkl', 'rb') as f:
                    tokenizer = pickle.load(f)
                with open('label_map.json', 'r') as f:
                    label_map = json.load(f)

                # Load model
                model = load_model('models/model.h5', custom_objects={'AttentionLayer': AttentionLayer})

                # Prepare input
                cleaned = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=30, padding='post')

                # Predict
                prediction = model.predict(padded)
                predicted_index = np.argmax(prediction)
                predicted_label = label_map.get(str(predicted_index), "Will use for training and tell you later.")

                st.success(f"Predicted Label: `{predicted_label}`")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

            finally:
                # Clean up to reduce memory usage
                del model
                del tokenizer
                del label_map
                gc.collect()
