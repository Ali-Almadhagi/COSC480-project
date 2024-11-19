import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from facenet_pytorch import MTCNN
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt


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
# def detect_and_crop_face(image_path):
#     # Open the image using PIL
#     img = Image.open(image_path).convert("RGB")

#     # Detect face using MTCNN and get the cropped face
#     face = mtcnn(img)

#     # If no face is detected, return the original image
#     if face is None:
#         print("No face detected. Using the original image.")
#         return img
#     else:
#         print("Face detected and cropped successfully.")
#         # Convert the tensor output from MTCNN to a PIL image
#         face_pil = transforms.ToPILImage()(face.squeeze(0))
#         plt.imshow(face_pil)
#         plt.title("Detected Face")
#         plt.axis('off')
#         plt.show()
#         return face_pil

# Prediction function using ResNet model
def predict_drowsiness(image_path, model):
    # Detect and crop the face from the image
    # img = detect_and_crop_face(image_path)
    img= Image.open(image_path)

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
