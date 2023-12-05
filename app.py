import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# โหลดโมเดล
model = tf.keras.models.load_model('mobilenetv3_large_100_checkpoint_fold2.pt')  # แทนที่ด้วยตำแหน่งที่เก็บโมเดลของคุณ

st.title('CNN Image Classification')
st.text('Upload an image for classification')

# ฟังก์ชันสำหรับประมวลผลภาพ
def process_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0  # ปรับสเกลให้อยู่ในช่วง 0-1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# โค้ดของแอป
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # ประมวลผลภาพเมื่อกดปุ่ม
    if st.button('Classify'):
        processed_img = process_image(image)
        prediction = model.predict(processed_img)
        classes = ['Alaxan', 'Bactidol', 'Decolgen', 'Fish_Oil', 'Kremil_S']  # แทนที่ด้วยชื่อคลาสของคุณ
        class_idx = np.argmax(prediction)
        st.write(f'Prediction: {classes[class_idx]}')
        st.write(f'Probability: {prediction[0][class_idx]:.2%}')
