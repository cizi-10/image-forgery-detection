import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model_path = 'saved_model.pb'  # Update this with your actual model path
model = load_model(model_path, custom_objects={'loss': 'binary_crossentropy', 'f1_score': 'binary_accuracy'})  # Update custom_objects as needed

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path)
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
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
    predicted_mask = model.predict(preprocessed_image)

    # Post-process the predicted mask
    original_image = Image.open(image_path)
    mask = postprocess_mask(predicted_mask, original_image.size)

    # Overlay mask on the original image
    overlayed_image = overlay_mask(image_path, mask)

    return overlayed_image

# Test the function with an input image
input_image_path = 'Tp_D_CND_S_N_ani00073_ani00068_00193.png'  # Update this with the actual image path
overlayed_image = predict_manipulated_regions(input_image_path)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(overlayed_image)
plt.axis('off')
plt.title('Manipulated Regions Highlighted')
plt.show()
