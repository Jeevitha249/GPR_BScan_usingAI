import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model_path = r"C:\Users\i5 7TH GEN 8GB 256GB\Desktop\gpr_classification_model.h5"  # Path to your saved model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['metal_pipe', 'metal_rod', 'pvc_pipe']  # Adjust as per your training classes

# Function to preprocess the input image
def preprocess_image(image, input_shape):
    img = image.resize(input_shape[:2])  # Resize image
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit app
st.title("GPR Object Classification")
st.write("Upload an image to predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image and make prediction
    input_shape = (128, 128, 3)  # Input shape used during training
    preprocessed_image = preprocess_image(image, input_shape)
    predictions = model.predict(preprocessed_image)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display the predicted class
    st.write(f"Predicted Class: **{predicted_class}**")

    # Optional: Overlay label on the image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 10), predicted_class, fill="red", font=font)

    # Display image with label
    st.image(image, caption="Predicted Image", use_container_width=True)

