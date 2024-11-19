import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from facenet_pytorch import MTCNN
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


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

def calculate_average_brightness(image):
    # Convert image to grayscale to calculate brightness
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)

    # Calculate average brightness (mean pixel value)
    avg_brightness = np.mean(np_image)
    return avg_brightness

# Function to detect and crop face using MTCNN
def detect_and_crop_face(image_path):
    # Open the image using PIL
    img = Image.open(image_path).convert("RGB")

    # Find the avg brightness
    avg_brightness = calculate_average_brightness(img)
    print(f"Average brightness: {avg_brightness}")

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
        # Convert the tensor output from MTCNN to a PIL image
        cropped_face = transforms.ToPILImage()(face.squeeze(0))


    # Calculate the average brightness of the image
    avg_brightness = calculate_average_brightness(cropped_face)
    print(f"Average brightness: {avg_brightness}")


    # We will enhance the brightness if the average is below a certain level
    if avg_brightness < 30:
        brightness_factor = 5.0
    elif avg_brightness > 50:
        brightness_factor = 2.0
    else:
        brightness_factor = 3.0

    # Increase the brightness of the image
    enhancer = ImageEnhance.Brightness(cropped_face)
    cropped_face = enhancer.enhance(brightness_factor)

    # Show the image to verify it's loaded correctly
    cropped_face.show()

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
