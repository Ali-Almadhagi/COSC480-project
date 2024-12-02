import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import os
from facenet_pytorch import MTCNN
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np


# Load the pre-trained ResNet model with a modified final layer
def load_model():
    # Load ResNet18 model with pretrained weights
    # Load EfficientNet B0 model with pretrained weights
    model = efficientnet_b0(weights=None)

    # Modify the final fully connected layer for binary classification
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 2)  # Binary classification layer
    )
    model_path = "models/drowsiness_detection_efficientnet_b0(2).pth"
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
def detect_and_crop_face(image_path, zoom_out_factor=1.2):
    """
    Detect and crop the face from the image, with a zoom-out factor to include the whole head.

    Args:
        image_path (str): Path to the input image.
        zoom_out_factor (float): Factor by which to expand the bounding box (default is 1.2 for 20% zoom out).

    Returns:
        PIL.Image: The cropped image, or the original image if no face is detected.
    """
    # Open the image using PIL
    img = Image.open(image_path).convert("RGB")

    # Detect face using MTCNN
    boxes, _ = mtcnn.detect(img)

    # If no face is detected, return the original image
    if boxes is None:
        print("No face detected. Using the original image.")

        # Save the image to the desktop for debugging purposes
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "no_face_detected_image.jpg")
        img.save(desktop_path)
        print(f"Image saved to: {desktop_path}")

        return img
    else:
        print("Face detected. Cropping the image.")

        # Get the first detected bounding box
        box = boxes[0]
        x1, y1, x2, y2 = box

        # Calculate the width and height of the bounding box
        width = x2 - x1
        height = y2 - y1

        # Expand the bounding box dimensions for zoom-out effect
        x1 = max(0, x1 - (width * (zoom_out_factor - 1) / 2))
        y1 = max(0, y1 - (height * (zoom_out_factor - 1) / 2))
        x2 = min(img.width, x2 + (width * (zoom_out_factor - 1) / 2))
        y2 = min(img.height, y2 + (height * (zoom_out_factor - 1) / 2))

        # Crop the image using the expanded bounding box
        cropped_face = img.crop((x1, y1, x2, y2))
        #cropped_face.show()

        return cropped_face


# Prediction function using ResNet model
def predict_drowsiness(image_path, model):
    # Detect and crop the face from the image
    img = detect_and_crop_face(image_path)

    # Apply transformations
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        print("Raw outputs (logits):", outputs)  # Debug: Log raw outputs
        print("Probabilities:", probabilities)  # Debug: Log probabilities

        _, predicted = torch.max(probabilities, 1)  # Select the class with highest probability
        class_index = predicted.item()

    # Map the predicted class index to class name
    class_names = ['Drowsy', 'Non Drowsy']
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
