import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Parameters
    batch_size = 64
    num_epochs = 10
    data_dir = '/content/Driver Drowsiness Dataset (DDD)'

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    # Basic transformation for validation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

     # Load dataset and apply transformations
    dataset = ImageFolder(root=data_dir)

    # Separate indices by class
    drowsy_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == dataset.class_to_idx['Drowsy']]
    non_drowsy_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == dataset.class_to_idx['Non Drowsy']]

    # Split each class into 95% training and 5% validation
    drowsy_train_size = int(0.95 * len(drowsy_indices))
    non_drowsy_train_size = int(0.95 * len(non_drowsy_indices))

    # Split the indices for each class
    drowsy_train_indices = drowsy_indices[:drowsy_train_size]
    drowsy_val_indices = drowsy_indices[drowsy_train_size:]
    non_drowsy_train_indices = non_drowsy_indices[:non_drowsy_train_size]
    non_drowsy_val_indices = non_drowsy_indices[non_drowsy_train_size:]

    # Combine indices for balanced train and validation sets
    train_indices = drowsy_train_indices + non_drowsy_train_indices
    val_indices = drowsy_val_indices + non_drowsy_val_indices

    # Create subsets using the specific indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Apply the correct transformations to each subset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Initialize resnet18 model
    model = resnet18(pretrained=True)
    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final layer for binary classification (drowsy and non-drowsy)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics for the epoch
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, '
              f'F1-Score: {f1:.2f}')

    # Validation phase
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate and print validation metrics
    val_accuracy = accuracy_score(val_labels, val_preds) * 100
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    # Plot confusion matrix for validation set
    plot_confusion_matrix(val_labels, val_preds, classes=['Non-Drowsy', 'Drowsy'])

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'drowsiness_detection_model_resnet18.pth')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()
