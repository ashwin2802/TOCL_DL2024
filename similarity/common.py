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

import argparse
from utils.compute_metrics import *
from torch.autograd import grad

from common import *

# Define the task-aware ResNet model
from models.model_loader import ModuleLoader

def get_non_task_head_parameters(model):
    """
    Filter out the parameters associated with `self.task_heads`.
    """
    return [p for name, p in model.named_parameters() if "task_heads" not in name]

def hessian_vector_product(loss, model, vector):
    # Get the non-task-head parameters
    non_task_head_params = get_non_task_head_parameters(model)

    # Compute first-order gradients
    grads = grad(loss, non_task_head_params, create_graph=True)

    # Split the vector into the same shapes as the non-task-head parameters
    vector_split = []
    pointer = 0
    for param in non_task_head_params:
        num_param = param.numel()
        vector_split.append(vector[pointer:pointer + num_param].reshape_as(param))
        pointer += num_param

    # Compute Hessian-vector product
    grads_vector = grad(
        grads, non_task_head_params, grad_outputs=vector_split, retain_graph=True
    )
    # Flatten the resulting gradients
    return torch.cat([g.reshape(-1) for g in grads_vector])

def generate_random_images(batch_size, image_shape):
    """
    Generate a batch of random images with the given shape.
    """
    return torch.rand(batch_size, *image_shape)

# Training loop with batch augmentation
def train_model_with_augmentation(model, train_loader, optimizer, criterion, device, args, task_id):
    """
    Train the model with augmented batches (original + random images with label 1).
    """
    model.train()
    for epoch in range(args['train_epochs']):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            # Move data to the appropriate device
            images, labels = images.to(device), labels.to(device)
            
            # Generate random images and assign label 1
            random_images = generate_random_images(images.size(0), images.shape[1:]).to(device)
            random_labels = torch.ones(images.size(0), dtype=torch.long).to(device)

            # Augment the batch
            augmented_images = torch.cat((images, random_images), dim=0)
            augmented_labels = torch.cat((labels, random_labels), dim=0)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(augmented_images, task_label=task_id)

            # Compute loss
            loss = criterion(outputs, augmented_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{args['train_epochs']}, Loss: {running_loss / len(train_loader)}")
    
    