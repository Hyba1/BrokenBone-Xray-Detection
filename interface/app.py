import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image,ImageOps

st.set_page_config(layout="centered", page_title="Bone Fracture Detector")

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("./model/resnet50_epoch15.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

def predict(image):
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        return prob.argmax().item(), prob.max().item()
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None, None

with st.sidebar:
    st.title("X-ray Analysis")
    st.markdown("---")
    upload = st.file_uploader("Choose X-ray", type=["jpg", "png", "jpeg"])
    st.markdown("---")
    analyze_btn = False
    if upload:
        analyze_btn = st.button("Analyze", type="primary")

st.header("Bone Fracture Detector", divider="rainbow")

if upload:
    try:
        img = Image.open(upload)
        img = ImageOps.fit(img, (300, 400))  
        st.image(img, width=300)  

        # st.image(img, caption="Uploaded X-ray",  width=300)
        
        
        if analyze_btn:
          
            st.info(f"Image mode: {img.mode}")  
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            class_idx, confidence = predict(img)
            
            if confidence is not None:  
                st.subheader("Results")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", ["Normal", "Abnormal"][class_idx])
                col2.metric("Confidence", f"{confidence:.1%}")
                
                if class_idx == 1:
                    st.error("Clinical recommendation: Immediate specialist consultation")
                else:
                    st.success("No fracture detected")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
else:
    st.warning("Please upload an X-ray image")
