import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

MODEL_PATH = Path("ann_spam_model.keras")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")

st.set_page_config(
  page_title="ANN Spam Detector",
  layout="centered"
)

st.title("Spam Message Detector")
st.write("Check weather a message is Spam or Ham(Not Spam)")

# Check weather the model files exists
if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
  st.error(
    "Model files not found"
    "Please run experiment.ipynb first"
  )
  st.stop()
  
def load_artifacts():
  loaded_model = tf.keras.models.load_model(
    MODEL_PATH
  )
  
  with open(VECTORIZER_PATH,"rb") as file:
    loaded_vectorizer = pickle.load(file)
    
  return loaded_model,loaded_vectorizer

model,vectorizer = load_artifacts()

# User input

message = st.text_area(
  "Enter your message",
  placeholder="Example: Congratulations! Claim your free prize.",
  height=150
)

#Predict when button is clicked

if st.button(
  "Check Message",
  type="primary",
  use_container_width=True
):
  if not message.strip():
    st.warning("Please enter a message")
    
  else:
    input_feature = vectorizer.transform(
      [message]
    ).toarray().astype(np.float32)
    
    # ANN Prediction
    prediction = model.predict(
      input_feature,
      verbose=0
    )

    spam_probability=float(prediction[0][0])
    st.write("probability:", spam_probability)

    # Classification
    if spam_probability >= 0.5:
      st.error("Prediction: Spam")

    else:
      st.success("Prediction: Not spam")