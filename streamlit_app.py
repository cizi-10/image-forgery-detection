import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import gdown
import os

# Function to download the model from Google Drive
def download_model(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

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
    predictions = predictions.numpy()  # Adjust based on model's output
    return predictions

# Streamlit app
st.title("Image Forgery Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Google Drive link to the model and output path
drive_url = 'https://drive.google.com/drive/folders/1KUzAPhuWDkSGVBwnGRIucoe34c-6bqWA?usp=sharing'
model_dir = 'SavedModel'
output_path = f'{model_dir}.zip'

# Download and extract model if not already present
if not os.path.exists(model_dir):
    try:
        download_model(drive_url, output_path)
        tf.keras.utils.get_file(fname=model_dir, origin=f'file://{output_path}', untar=True)
        st.success("Model downloaded and extracted successfully!")
    except Exception as e:
        st.error(f"Error downloading the model: {e}")

# Load model
try:
    saved_model = tf.saved_model.load(model_dir)
    infer = saved_model.signatures['serving_default']  # Load the serving signature
    model = tf.keras.layers.Lambda(lambda x: infer(tf.constant(x))['output_0'])  # Wrap in a Keras layer
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Display and predict
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    try:
        predictions = predict(uploaded_file, model)
        st.image(predictions[0], caption='Predicted Manipulated Regions', use_column_width=True)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
