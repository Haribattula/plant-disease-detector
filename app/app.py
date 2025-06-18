import os
os.environ["STREAMLIT_HOME"] = "/tmp"
os.environ["XDG_CONFIG_HOME"] = "/tmp"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_USAGE_STATS"] = "false"

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
from utils.model import ResNet9
from utils.disease import disease_dic

st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection App")

# Disease classes
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                   'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                   'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                   'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                   'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                   'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# Load model
disease_model_path = 'Code/models/plant_disease_model.pth'
model = ResNet9(3, len(disease_classes))
model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
model.eval()

# Prediction function
def predict(img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image).unsqueeze(0)
    output = model(img_t)
    _, pred = torch.max(output, 1)
    return disease_classes[pred.item()]

# Upload and predict
uploaded_file = st.file_uploader("Upload a plant leaf image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded Leaf Image", use_column_width=True)
    with st.spinner("Predicting..."):
        label = predict(image_bytes)
        description = disease_dic.get(label, "No description available.")
        st.success(f"**Prediction:** {label}")
        st.markdown(f"**Details:** {description}")
