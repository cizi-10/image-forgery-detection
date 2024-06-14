import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import streamlit as st

# Load the saved model
model_path = 'saved_model.pb'  # Update this with your actual model directory
try:
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path)
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to post-process the predicted mask
def postprocess_mask(mask, original_size):
    mask = np.squeeze(mask)  # Remove batch dimension
    mask = (mask > 0.5).astype(np.uint8)  # Binarize
    mask = Image.fromarray(mask * 255)  # Convert to PIL Image
    mask = mask.resize(original_size, Image.NEAREST)  # Resize to original size
    return mask

# Function to overlay mask on the original image
def overlay_mask(image_path, mask):
    original_image = Image.open(image_path)
    mask = mask.convert("L")  # Convert mask to grayscale
    mask = ImageOps.colorize(mask, black="black", white="red")  # Colorize mask (red for manipulated regions)
    overlayed_image = Image.blend(original_image, mask, alpha=0.5)  # Blend original image and mask
    return overlayed_image

# Function to predict manipulated regions
def predict_manipulated_regions(image_path):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)

    # Predict the mask
    input_tensor = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)
    output_dict = infer(input_tensor)
    predicted_mask = output_dict['output_0'].numpy()

    # Post-process the predicted mask
    original_image = Image.open(image_path)
    mask = postprocess_mask(predicted_mask, original_image.size)

    # Overlay mask on the original image
    overlayed_image = overlay_mask(image_path, mask)

    return overlayed_image

# Streamlit app
st.title("Image Forgery Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    try:
        overlayed_image = predict_manipulated_regions("temp_image.png")
        st.image(overlayed_image, caption="Manipulated Regions Highlighted", use_column_width=True)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
