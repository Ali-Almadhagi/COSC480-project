import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress bar

# Define a CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Adjusted for the correct input size
        self.fc2 = nn.Linear(128, 2)  # Adjusted for 2 classes: drowsy and non-drowsy
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Flatten the feature maps correctly
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

# Initialize the CNN model
cnn_model = CNN()
criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

batch_size = 64  # You can adjust the batch size if necessary
num_epochs = 10  # You can change the number of epochs as needed
train_test_split_ratio = 0.8  # 80% training, 20% testing

# Define transformations for preprocessing the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Set the path to the dataset
data_dir = '../Driver Drowsiness Dataset (DDD)'  # Replace with the correct path

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and testing sets (80% training, 20% testing)
train_size = int(train_test_split_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check if data was indexed properly by printing class-to-index mapping
print("Class to index mapping:", dataset.class_to_idx)

for epoch in range(num_epochs):
    # Train and evaluate the CNN model
    train_loss, train_acc = train(cnn_model, train_loader, criterion, cnn_optimizer)
    test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# Final evaluation on test set
cnn_test_loss, cnn_test_acc = evaluate(cnn_model, test_loader, criterion)
print(f"Final Test Loss: {cnn_test_loss:.4f}, Final Test Accuracy: {cnn_test_acc:.2f}%")
