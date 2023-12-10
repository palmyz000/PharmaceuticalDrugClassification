import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Load the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('modelmobilenetv3.h5', map_location=device)
model.eval()

# Set the image transformation
image_size = (224, 224)  # You can adjust the size as needed
image_transform = T.Compose([
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set title and header
st.title('Pharmaceutical Drugs Classification')
st.header('Please upload an image')

# Image upload and prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with torch.no_grad():
            # Preprocess the image
            img_tensor = image_transform(image).unsqueeze(0)
            img_tensor = img_tensor.to(device)

            # Get prediction
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            class_names = ['Alaxan', 'Bactidol', 'Decolgen', 'Fish_Oil', 'Kremil_S']  # Update with your class names

            # Display prediction
            st.write(f"Predicted Class: {class_names[predicted_class]}")
            st.write(f"Probability: {probabilities[predicted_class].item()*100:.2f}%")
