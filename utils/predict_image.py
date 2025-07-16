from PIL import Image, ImageEnhance
import torch
from torchvision import transforms

def predict_image(path, model, device):
    model.eval()

    image = Image.open(path).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
        class_name = "Abnormal" if pred.item() == 1 else "Normal"
        print(f"Prediction: {class_name}")
        return class_name
