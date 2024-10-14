#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# Define a CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Adjusted for 2 classes: drowsy and non-drowsy
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load and preprocess data
#transform tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Load the training dataset
#file path
train_dir = '../Driver Drowsiness Dataset (DDD)'

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)

#set batch size
batch_size = 64
train_test_split_ratio = 0.8

# Split the dataset into training and testing sets (80% training, 20% testing)
train_size = int(train_test_split_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


#load train and test data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#check if data was indexed properly
print(dataset.class_to_idx)


# Train the model

#train and eval functions
#
#
#
#


num_epochs = 5
# opt to use gpu if device has it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#loop through epochs
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)


cnn_test_loss, cnn_test_acc = evaluate(model, test_loader, criterion)
print(f"cnn Test Loss: {cnn_test_loss:.4f}, Test Accuracy: {cnn_test_acc:2f}%")
