import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow model
model_path = 'modelmobilenetv3.h5'
model = tf.keras.models.load_model(model_path)

# Set the image transformation
image_size = (224, 224)

# Preprocess function
def preprocess_image(image):
    img = image.resize(image_size)
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img

# Set title and header
st.title('Pharmaceutical Drugs Classification')
st.header('Please upload an image')

# Image upload and prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        # Preprocess the image
        img = preprocess_image(image)
        
        # Get prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        class_names = ['Alaxan', 'Bactidol', 'Decolgen', 'Fish_Oil', 'Kremil_S']  # Update with your class names

        # Display prediction
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Probability: {predictions[0][predicted_class]*100:.2f}%")
