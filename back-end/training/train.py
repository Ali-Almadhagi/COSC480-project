# Load and preprocess data
#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#transform tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Load the training dataset
#file path 
train_dir = 'ExamplePath'

#load train and test folder 
train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)

#set batch size
batch_size = 64

#load train and test data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#check if data was indexed properly 
print(train_dataset.class_to_idx)

# Build a simple CNN model for demonstration purposes


# Train the model
