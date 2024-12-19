import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from copy import deepcopy
from collections import Counter

# Parameters
classes_per_task = 10  # Number of classes per task
batch_size = 128
learning_rate = 0.001
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),           # Random crop
    transforms.RandomHorizontalFlip(),              # Horizontal flip
    transforms.RandomRotation(15),                 # Random rotation
    transforms.ToTensor(),                         # Convert to tensor
    transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),  # Normalize
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
])

cache_dir = "/cluster/scratch/rrigoni/.cache/huggingface"
train_dataset = datasets.CIFAR100(root=cache_dir, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR100(root=cache_dir, train=False, download=True, transform=test_transform)

# Group classes into tasks
num_classes = 100
all_classes = list(range(num_classes))

task_groups = [all_classes[i : min(i + classes_per_task, len(all_classes))] for i in range(0, len(all_classes), classes_per_task)]
num_tasks = len(task_groups)
print(f"num_tasks: {num_tasks}, task_groups: {task_groups}")

# Count samples per class in the dataset
class_counts = Counter(test_dataset.targets)

# Aggregate counts based on task groups
task_group_counts = {}
for task_id, task_classes in enumerate(task_groups):
    task_count = sum(class_counts[class_label] for class_label in task_classes)
    task_group_counts[f"Task {task_id + 1}"] = task_count

# Print results
print("\nNumber of samples per task group:")
for task, count in task_group_counts.items():
    print(f"{task}: {count}")

def remap_labels(dataset, task_classes):
    """
    Remap the labels in a deep copy of the dataset to fit within the range [0, classes_per_task-1].

    Parameters:
        dataset (Dataset): The dataset to remap labels for.
        task_classes (list): List of class labels for the current task.

    Returns:
        Subset: A subset of the copied dataset with labels remapped to the range [0, classes_per_task-1].
    """
    # Create a deep copy of the dataset
    dataset_copy = deepcopy(dataset)

    # Create a mapping from the original labels to new labels
    class_mapping = {original: new for new, original in enumerate(task_classes)}

    # Create a list of indices and remap the labels in the copied dataset
    indices = []
    for i, (image, label) in enumerate(dataset_copy):
        if label in class_mapping:
            indices.append(i)
            dataset_copy.targets[i] = class_mapping[label]  # Update the label in the copy

    # Return the subset of the copied dataset with remapped labels
    return Subset(dataset_copy, indices)

# Define the task-aware ResNet model
from models.model_loader import ModuleLoader

loader = ModuleLoader()
model = loader.load_model(f'task_aware_resnet-18-{classes_per_task}-{num_tasks}')
model = model.to(device)
# print(f"model: {model}")

# Training and evaluation
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for task_id, task_classes in enumerate(task_groups):
    print(f"\nTraining on Task {task_id + 1}/{num_tasks} with classes {task_classes}")

    # Remap train and test datasets for the current task
    task_train_dataset = remap_labels(train_dataset, task_classes)
    # task_test_dataset = remap_labels(test_dataset, task_classes)

    print(f"len(task_train_dataset): {len(task_train_dataset)}")

    train_loader = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(task_test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model on the current task
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            # print(f"task_classes: {task_classes}, labels: {labels}")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, task_label=task_id)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Validate on all tasks seen so far
    print(f"\nValidating on all tasks...")
    model.eval()
    for eval_task_id, eval_task_classes in enumerate(task_groups):
        print(f"eval_task_id: {eval_task_id}, eval_task_classes: {eval_task_classes}")
        eval_test_dataset = remap_labels(test_dataset, eval_task_classes)
        print(f"len(eval_test_dataset): {len(eval_test_dataset)}")
        eval_loader = DataLoader(eval_test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in eval_loader:
                # print(f"task_classes: {eval_task_classes}, labels: {labels}")
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_label=eval_task_id)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Task {eval_task_id + 1} Test Accuracy: {accuracy:.2f}%")
