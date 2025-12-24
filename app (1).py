import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

# 🧠 Load Model
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    return model, idx_to_class

# 🖼️ Image Preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # add batch dimension

# 🔍 Prediction
def predict(image, model, idx_to_class):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = idx_to_class[predicted.item()]
    return class_name

# 🖥️ Streamlit UI
st.title("🦠 Virus Image Classifier")

uploaded_file = st.file_uploader("Upload a TEM virus image (.tif or .png)", type=["tif", "png", "jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model, idx_to_class = load_model("virus_deit_model.pth")

    with st.spinner("Predicting..."):
        prediction = predict(image, model, idx_to_class)
    st.success(f"🧬 Predicted Virus Type: **{prediction}**")
