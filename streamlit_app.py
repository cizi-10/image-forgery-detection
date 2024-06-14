import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# Function to load the model
def load_model(model_dir):
    return tf.saved_model.load(model_dir)

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to make predictions
def predict(image_path, model):
    image = preprocess_image(image_path)
    predictions = model(image)
    predictions = predictions['output_0'].numpy()  # Adjust this line based on your model's output
    return predictions

# Streamlit app
st.title("Image Forgery Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model
model_path = "model"  # Change this to the directory containing saved_model.pb
try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Display and predict
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    predictions = predict(uploaded_file, model)
    st.image(predictions[0], caption='Predicted Manipulated Regions', use_column_width=True)
