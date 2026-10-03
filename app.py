import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ==========================================
# 1. Page Configuration & Model Setup
# ==========================================
st.set_page_config(page_title="Smart Trash Classifier", page_icon="♻️")

CLASS_NAMES = ["Organic", "Recyclable", "Residual"]
SAVE_PATH = "best_mobilenet_waste.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
    
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. Web Interface Layout
# ==========================================
st.title("♻️ Smart Trash Classifier")
st.write("Upload an image or snap a picture using your camera to identify the trash category.")

option = st.radio("Choose Input Method:", ("Camera Capture", "File Upload"))

uploaded_image = None

if option == "Camera Capture":
    uploaded_image = st.camera_input("Take a photo of the item")
else:
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])

# ==========================================
# 3. Inference Logic
# ==========================================
if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="Target Image", width="stretch")
        
    with col2:
        tensor = test_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            
        top_prob, top_idx = torch.max(probabilities, 0)
        prediction = CLASS_NAMES[top_idx]
        confidence = top_prob.item() * 100
        
        st.success(f"**Prediction:** {prediction.upper()}")
        st.metric(label="Confidence Level", value=f"{confidence:.1f}%")

    st.subheader("Confidence Breakdown")
    for name, prob in zip(CLASS_NAMES, probabilities):
        st.progress(float(prob), text=f"{name.capitalize()}: {prob.item()*100:.1f}%")