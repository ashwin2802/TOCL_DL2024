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

import argparse
from utils.compute_metrics import *
from torch.autograd import grad

from common import *
from pipeline.common import *

from models.model_loader import ModuleLoader

def compute_curvature(
    model, Tj_loader, Tk_loader, criterion, device, task_label_j, task_label_k
):
    # Get the non-task-head parameters
    non_task_head_params = get_non_task_head_parameters(model)

    def augment_batch(images, labels):
        """
        Augment a batch of images and labels with random images assigned label 1.
        """
        batch_size = images.size(0)
        random_images = generate_random_images(batch_size, images.shape[1:]).to(device)
        random_labels = torch.ones(batch_size, dtype=torch.long).to(device)
        augmented_images = torch.cat((images, random_images), dim=0)
        augmented_labels = torch.cat((labels, random_labels), dim=0)
        return augmented_images, augmented_labels

    # Compute gradients g_k for task Tk
    model.eval()
    Tk_iter = iter(Tk_loader)
    Xk, yk = next(Tk_iter)
    Xk, yk = Xk.to(device), yk.to(device)

    # Augment Tk batch
    Xk_aug, yk_aug = augment_batch(Xk, yk)

    output_k = model(Xk_aug, task_label=task_label_k)
    loss_k = criterion(output_k, yk_aug)
    grads_k = grad(loss_k, non_task_head_params, create_graph=True)
    g_k = torch.cat([g.reshape(-1) for g in grads_k])

    # Compute squared L2 norm of g_k
    g_k_squared_norm = g_k.norm(p=2).pow(2)  # Compute L2 norm and square it

    # Compute Hessian-vector product H_j g_k
    Tj_iter = iter(Tj_loader)
    Xj, yj = next(Tj_iter)
    Xj, yj = Xj.to(device), yj.to(device)

    # Augment Tj batch
    Xj_aug, yj_aug = augment_batch(Xj, yj)

    output_j = model(Xj_aug, task_label=task_label_j)
    loss_j = criterion(output_j, yj_aug)

    # Compute gradients g_j for task Tj
    grads_j = grad(loss_j, non_task_head_params, create_graph=True)
    g_j = torch.cat([g.reshape(-1) for g in grads_j])

    # Normalize gradients to avoid scaling issues
    g_k_normalized = g_k / (g_k.norm(p=2) + 1e-8)  # Normalize g_k
    g_j_normalized = g_j / (g_j.norm(p=2) + 1e-8)  # Normalize g_j

    # Compute the inner product of normalized gradients
    normalized_inner_product = g_k_normalized.dot(g_j_normalized)

    # Compute Hessian-vector product H_j g_k
    Hv = hessian_vector_product(loss_j, model, g_k)

    # Compute curvature c(j, k) and normalize by the squared L2 norm of g_k
    curvature = g_k.dot(Hv)
    normalized_curvature = curvature / (g_k_squared_norm + 1e-8)  # Add small epsilon for numerical stability

    # Calculate order of magnitude for normalized_curvature
    avg_curvature_magnitude = normalized_curvature.abs().mean()
    order_of_magnitude = torch.log10(avg_curvature_magnitude + 1e-8).floor() + 1
    scale_factor = 10 ** order_of_magnitude

    # Scale down normalized_inner_product
    scaled_inner_product = normalized_inner_product * scale_factor
    # Take the negative because larger values mean higher heterogeneity
    scaled_inner_product = - scaled_inner_product

    # Compute the final score using the scaled inner product
    final_score = normalized_curvature + scaled_inner_product

    # Return all metrics
    return final_score.item()
    # return {
    #     "final_score": final_score.item(),
    #     "normalized_curvature": normalized_curvature.item(),
    #     "scaled_inner_product": scaled_inner_product.item(),
    #     "order_of_magnitude": order_of_magnitude.item(),
    # }

def validation_with_gradients(model, task_groups, test_dataset, args, device, criterion, task_id):
    results = []
    model.eval()
    
    # Sample a batch from task k
    task_j_classes = task_groups[task_id]
    task_j_dataset = remap_labels(test_dataset, task_j_classes)
    task_j_loader = DataLoader(task_j_dataset, batch_size=args['test_mb_size'], shuffle=True)
    
    for eval_task_id, eval_task_classes in tqdm(enumerate(task_groups)):
        eval_test_dataset = remap_labels(test_dataset, eval_task_classes)
        task_k_loader = DataLoader(eval_test_dataset, batch_size=args['test_mb_size'], shuffle=True)

        curvature = compute_curvature(model, task_j_loader, task_k_loader, nn.CrossEntropyLoss(), device, task_id, eval_task_id)
        results.append(curvature)

    return results

def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes_per_task = 2 # 0 -> real images, 1 -> random noise

    train_dataset, test_dataset = get_dataset('mnist-10')

    num_classes = 10
    task_groups = [[i] for i in range(num_classes)]
    num_tasks = len(task_groups)
    print(f"num_tasks: {num_tasks}, task_groups: {task_groups}")

    results = []
    for task_id, task_classes in enumerate(task_groups):
        print(f"Reinstantiating model for task_id: {task_id + 1}/{len(task_groups)}")
        loader = ModuleLoader()
        model = loader.load_model(args['model_name'].format(classes_per_task, num_tasks))
        model = model.to(device)

        # Training and evaluation
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)

        # Remap train and test datasets for the current task
        task_train_dataset = remap_labels(train_dataset, task_classes)
        train_loader = DataLoader(task_train_dataset, batch_size=args['train_mb_size'], shuffle=True)

        # Train the model on the current task
        model.train()
        train_model_with_augmentation(model, train_loader, optimizer, criterion, device, args, task_id)

        model.eval()
        print(f"Computing gradient scores...")
        gradient_results = validation_with_gradients(model, task_groups, test_dataset, args, device, criterion, task_id)
        print(f"gradient_results: {gradient_results}")
        results.append(gradient_results)

    file_path = args['res_file_template'].format(args['model_name'].format(classes_per_task, num_tasks), args['train_epochs'])
    with open(file_path, 'w') as f: 
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help=""
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f: 
        config = json.load(f)

    main(config)
