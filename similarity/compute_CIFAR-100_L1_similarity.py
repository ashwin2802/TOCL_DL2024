import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from itertools import combinations
import random

# Parameters
n = 100  # Number of random image pairs to sample
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-100 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))  # CIFAR-100 statistics
])

test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Group images by class
class_indices = {i: [] for i in range(100)}
for idx, (_, label) in enumerate(test_dataset):
    class_indices[label].append(idx)

# Function to compute average L1 distance between n random image pairs
def average_l1_distance(dataset, indices1, indices2, n):
    distances = []
    for _ in range(n):
        idx1 = random.choice(indices1)
        idx2 = random.choice(indices2)
        
        # Load images
        img1 = dataset[idx1][0].to(device)
        img2 = dataset[idx2][0].to(device)
        
        # Compute L1 distance
        l1_distance = torch.abs(img1 - img2).mean().item()
        distances.append(l1_distance)
    
    return np.mean(distances)

# Compute the similarity matrix
similarity_matrix = np.zeros((100, 100))

print("Computing similarity matrix...")
for i, j in combinations(range(100), 2):  # Iterate over class pairs
    print(f"Computing distance between class {i} and class {j}...")
    avg_distance = average_l1_distance(test_dataset, class_indices[i], class_indices[j], n)
    similarity_matrix[i, j] = avg_distance
    similarity_matrix[j, i] = avg_distance

# Save the similarity matrix
np.save("similarity_matrix.npy", similarity_matrix)

print("Similarity matrix computation completed and saved!")
