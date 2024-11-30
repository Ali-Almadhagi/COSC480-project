import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import os
from facenet_pytorch import MTCNN
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np


# Load the pre-trained ResNet model with a modified final layer
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model_path = "models/drowsiness_detection_model_resnet18.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Initialize the MTCNN face detector
mtcnn = MTCNN(keep_all=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# Function to detect and crop face using MTCNN
def detect_and_crop_face(image_path):
    # Open the image using PIL
    img = Image.open(image_path).convert("RGB")

    # Detect face using MTCNN
    face = mtcnn(img)
    # If no face is detected, return the original image and save it to desktop
    if face is None:
        print("No face detected. Using the original image.")

        # Define the path to save the image (change 'your_username' to your actual desktop path if needed)
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "no_face_detected_image.jpg")

        # Save the image to the desktop
        img.save(desktop_path)
        print(f"Image saved to: {desktop_path}")

        return img
    else:
        print("Face detected and cropped successfully.")

        #Rescale tensor values from [-1, 1] to [0, 1]
        cropped_face = face.squeeze(0)  # Remove batch dimension
        cropped_face = (cropped_face + 1) / 2  # Rescale to [0, 1]

        cropped_face_pil = transforms.ToPILImage()(cropped_face)
        cropped_face_pil.show()

    return cropped_face_pil

# Prediction function using ResNet model
def predict_drowsiness(image_path, model):
    # Detect and crop the face from the image
    img = detect_and_crop_face(image_path)

    # Apply transformations
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()

    # Map the predicted class index to class name
    class_names = ['Non-Drowsy', 'Drowsy']
    result = class_names[class_index]
    return result

# Load the saved model
model_path = 'models/drowsiness_detection_model_resnet18.pth'
model = load_model()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def main():
    # Example usage for a single image
    prediction = predict_drowsiness('path/to/your_image.jpg', model, device)
    print(f"Prediction:{prediction}")

if __name__ == '__main__':
    main()
