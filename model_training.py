# importing libraries
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
# 1.1 Load data, visualize data
# First lets define the transform for the dataset to resize image and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Download the dataset required
data_dir = './data/food-101'
original_dataset = datasets.Food101(data_dir, download=True, transform=transform)

# Then lets make some initial steps to choose 15 classes from 101 class
# Here is original dataset
classes = original_dataset.classes
class_to_idx = original_dataset.class_to_idx

# Then select only 15 classes
selected_classes = random.sample(classes, 15)
selected_class_indices = [class_to_idx[class_name] for class_name in selected_classes]

# Next a new class-to-index mapping
new_class_to_idx = {selected_class_indices[i]: i for i in range(len(selected_class_indices))}

# Here is subset containing only the selected 15 classes
subset_indices = [i for i, (img, label) in enumerate(original_dataset) if label in selected_class_indices]
subset = Subset(original_dataset, subset_indices)

# Remapping of the labels in the subset
def remap_labels(subset, new_class_to_idx):
    remapped_subset = []
    for img, label in subset:
        new_label = new_class_to_idx[label]
        remapped_subset.append((img, new_label))
    return remapped_subset

remapped_subset = remap_labels(subset, new_class_to_idx)

# 1.2 Train-test split
# train-test split
labels = [label for _, label in remapped_subset]
train_indices, test_indices = train_test_split(list(range(len(remapped_subset))),
                                               test_size=0.3, random_state=42, stratify=labels)

train_val_set = Subset(remapped_subset, train_indices)
test_set = Subset(remapped_subset, test_indices)

# train-validation split
train_labels = [label for _, label in train_val_set]
train_indices, val_indices = train_test_split(list(range(len(train_val_set))),
                                              test_size=0.1, random_state=42, stratify=train_labels)

train_set = Subset(remapped_subset, train_indices)
val_set = Subset(remapped_subset, val_indices)

# 2.1-2.3
# Load pre-trained model resnet34 model
model = models.resnet34(pretrained=True)

# Modify the final layer for our specific number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(selected_classes))

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set,batch_size=32, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Lets proceed to training
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {running_loss / len(train_loader):.4f}, '
          f'Validation Loss: {val_loss / len(val_loader):.4f}, '
          f'Validation Accuracy: {correct / total:.4f}')