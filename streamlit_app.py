import streamlit as st 
import tensorflow as tf
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title 
st.title('Pharmaceutical Drugs Classification')

# Set Header 
st.header('Please upload a picture')

# Load Model 
model = tf.keras.models.load_model('saved_model.h5')

# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['Alaxan', 'Bactidol', 'Decolgen', 'Fish_Oil', 'Kremil_S']

    if st.button('Prediction'):
        # Resize and preprocess the image for prediction
        img_array = np.array(image.resize((224, 224)))
        img_array = img_array / 255.0  # Normalize the image
        
        # Make prediction
        predictions = model.predict(np.expand_dims(img_array, axis=0))
        probli = predictions.tolist()[0]
        
        st.write("## Prediction Result")
        # Get the index of the maximum value in probli
        max_index = np.argmax(probli)

        # Iterate over the class_name and probli lists
        for i in range(len(class_name)):
            # Set the color to blue if it's the maximum value, otherwise use the default color
            color = "blue" if i == max_index else None
            st.write(f"## <span style='color:{color}'>{class_name[i]} : {probli[i]*100:.2f}%</span>", unsafe_allow_html=True)
