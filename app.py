import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# โหลดโมเดล
model = torch.load('mobilenetv3_large_100_checkpoint_fold2.pt', map_location=torch.device('cpu'))  # แทนที่ด้วยตำแหน่งที่เก็บโมเดลของคุณ

st.title('CNN Image Classification')
st.text('Upload an image for classification')

# ฟังก์ชันสำหรับประมวลผลภาพ
def process_image(image):
    size = 224
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(image)
    img = img.unsqueeze(0)
    return img

# โค้ดของแอป
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # ประมวลผลภาพเมื่อกดปุ่ม
    if st.button('Classify'):
        processed_img = process_image(image)
        with torch.no_grad():
            prediction = model(processed_img)
            probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
            top_prob, top_class = probabilities.topk(1)
            classes = ['Alaxan', 'Bactidol', 'Decolgen', 'Fish_Oil', 'Kremil_S']  # แทนที่ด้วยชื่อคลาสของคุณ
            st.write(f'Prediction: {classes[top_class]}')
            st.write(f'Probability: {top_prob.item():.2%}')
