import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="😺 MeowMeow Silly Classifier!", page_icon="😺")
st.title("😺 MeowMeow Silly Classifier!")
st.write("Upload an image and let the AI decide: Is it Silly or Not Silly?")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/meowmeow.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    st.write("Analyzing...")

    img_resized = image.resize((model.input_shape[1], model.input_shape[2]))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictons 
    pred = model.predict(img_array)[0][0]
    silly_prob = float(pred)
    silly_label = "😹 Silly!" if silly_prob < 0.5 else "😼 Not Silly!"
    st.markdown(f"## Prediction: {silly_label}")
    st.progress(silly_prob if silly_prob > 0.5 else 1-silly_prob)
    st.write(f"Confidence: {silly_prob*100:.2f}% Silly" if silly_prob < 0.5 else f"Confidence: {(1-silly_prob)*100:.2f}% Not Silly")
    st.balloons() if silly_prob < 0.5 else st.snow()
else:
    st.info("Please upload an image to get started!")
