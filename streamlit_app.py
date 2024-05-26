import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops
import numpy as np

# Load the trained model
model_path = 'path_to_your_model/VGG16_417418.h5'  # Replace with the path to your model
model = load_model(model_path)

# Function to apply ELA
def ELA(img_path, quality=90, threshold=60):
    TEMP = 'ela_temp.jpg'
    SCALE = 10
    original = Image.open(img_path).convert('RGB')
    original.save(TEMP, quality=quality)
    temporary = Image.open(TEMP)
    diff = ImageChops.difference(original, temporary)
    
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    
    for x in range(WIDTH):
        for y in range(HEIGHT):
            r, g, b = d[x, y]
            modified_intensity = int(0.2989 * r + 0.587 * g + 0.114 * b)
            d[x, y] = modified_intensity * SCALE, modified_intensity * SCALE, modified_intensity * SCALE
    
    binary_mask = diff.point(lambda p: 255 if p > threshold else 0)
    return binary_mask

def preprocess_image(image):
    image = ELA(image)
    image = image.resize((100, 100))
    image = np.array(image) / 255.0
    image = image.reshape(1, 100, 100, 3)
    return image

# Streamlit app
st.title('Image Forgery Detection using ELA and VGG16')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    if np.argmax(prediction) == 0:
        st.write("The image is classified as: Real")
    else:
        st.write("The image is classified as: Fake")
