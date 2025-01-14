import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from copy import deepcopy
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle

import argparse
from utils.compute_metrics import *

# Define the task-aware ResNet model
from models.model_loader import ModuleLoader

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

def validation(model, task_groups, test_dataset, args, device): 
    results = []
    intermediate_representations = []
    
    # Validate on all tasks seen so far
    print(f"\nValidating on all tasks...")
    model.eval()
    for eval_task_id, eval_task_classes in enumerate(task_groups):
        print(f"eval_task_id: {eval_task_id}, eval_task_classes: {eval_task_classes}", flush=True)
        eval_test_dataset = remap_labels(test_dataset, eval_task_classes)
        print(f"len(eval_test_dataset): {len(eval_test_dataset)}", flush=True)
        eval_loader = DataLoader(eval_test_dataset, batch_size=args['test_mb_size'], shuffle=False)

        correct = 0
        total = 0
        feature_sums = None
        feature_count = 0

        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_label=eval_task_id)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Extract intermediate features and compute the running sum
                features = model.forward_feature_extraction(images)
                if feature_sums is None:
                    feature_sums = torch.zeros_like(features.sum(dim=0)).to(device)
                feature_sums += features.sum(dim=0)
                feature_count += features.size(0)

        # Compute average intermediate representation for the task
        avg_features = feature_sums / feature_count
        intermediate_representations.append(avg_features.cpu())  # Store as PyTorch tensor on CPU

        accuracy = 100 * correct / total
        print(f"Task {eval_task_id + 1} Test Accuracy: {accuracy:.2f}%", flush=True)
        results.append(accuracy)

    # Stack intermediate representations into a 2D tensor
    intermediate_rep_matrix = torch.stack(intermediate_representations)

    return results, intermediate_rep_matrix

# import torch
# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from itertools import cycle
# import os

# def plot_averaged_representations(averaged_representations, output_dir):
#     """
#     Plots the PCA of concatenated intermediate representations in 2D with detailed debug information.

#     Parameters:
#         averaged_representations (list): List of `num_tasks` many `[num_tasks, hidden_dim]` tensors.
#         output_dir (str): Directory to save the PCA plot.
#     """
#     print(f"Received {len(averaged_representations)} tensors in averaged_representations.", flush=True)

#     # Step 1: Concatenate tensors along the first dimension
#     concatenated_representations = torch.cat(averaged_representations, dim=0)  # Shape: [num_tasks * len(averaged_representations), hidden_dim]
#     print(f"Concatenated tensor shape: {concatenated_representations.shape}", flush=True)

#     # Step 2: Convert to numpy for PCA
#     all_representations = concatenated_representations.numpy()  # Convert PyTorch tensor to NumPy array
#     print(f"Converted to numpy. Shape: {all_representations.shape}", flush=True)

#     # Step 3: Apply PCA
#     print("Applying PCA...", flush=True)
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(all_representations)  # Shape: [num_tasks * len(averaged_representations), 2]
#     print(f"PCA completed. Result shape: {pca_result.shape}", flush=True)

#     # Prepare symbols and colors for plotting
#     num_tasks = averaged_representations[0].shape[0]  # Extract number of tasks from the first tensor
#     print(f"Number of tasks: {num_tasks}", flush=True)

#     symbols = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']  # Define a list of symbols
#     symbol_cycle = cycle(symbols[:num_tasks])  # Cycle through symbols for tasks
#     colors = plt.cm.viridis(np.linspace(0, 1, len(averaged_representations)))  # Progressive colormap
#     print(f"Prepared {len(symbols[:num_tasks])} symbols and {len(colors)} colors.", flush=True)

#     # Plot PCA results
#     print("Starting PCA plotting...", flush=True)
#     plt.figure(figsize=(10, 8))

#     # Total number of tensors (runs) used in averaged_representations
#     num_runs = len(averaged_representations)

#     # Flattened number of points per run (num_tasks * num_tasks)
#     num_points_per_run = num_tasks

#     for i, task_symbol in enumerate(symbols[:num_tasks]):  # Assign each task a unique symbol
#         print(f"Processing task {i + 1} with symbol '{task_symbol}'.", flush=True)
#         for run_idx, color in enumerate(colors):  # Iterate over runs for progressive coloring
#             # Compute indices for the current task in the concatenated PCA result
#             task_indices = [j * num_points_per_run + i for j in range(num_runs)]
#             print(f"Task {i + 1}, Run {run_idx + 1}: Indices -> {task_indices}", flush=True)

#             # Extract points for the current task and run
#             task_points = pca_result[task_indices]
#             print(f"Task {i + 1}, Run {run_idx + 1}: Points shape -> {task_points.shape}", flush=True)

#             # Scatter plot for the current task and run
#             plt.scatter(
#                 task_points[:, 0],
#                 task_points[:, 1],
#                 marker=task_symbol,
#                 color=color,
#                 alpha=0.7,
#                 label=f"Task {i + 1}, Run {run_idx + 1}" if run_idx == 0 else None  # Label only once per task
#             )

#     # Customize the plot
#     print("Finalizing the plot...", flush=True)
#     plt.title("PCA of Concatenated Intermediate Representations")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.grid(True)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot

#     # Save the plot
#     plot_path = os.path.join(output_dir, "pca_averaged_intermediate_representations.png")
#     plt.savefig(plot_path, bbox_inches="tight")
#     print(f"PCA plot saved to {plot_path}", flush=True)
#     plt.close()


def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes_per_task = args['classes_per_task']  # Number of classes per task

    # Transforms for MNIST
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),          # Random crop (MNIST images are 28x28)
        transforms.RandomHorizontalFlip(),             # Horizontal flip (optional, although MNIST is digits)
        transforms.RandomRotation(15),                # Random rotation
        transforms.ToTensor(),                         # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,)),    # Normalize with MNIST's mean and std
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))     # Same normalization for test set
    ])

    # MNIST Datasets
    train_dataset = datasets.MNIST(root=args['data_dir'], train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=args['data_dir'], train=False, download=True, transform=test_transform)
    
    # Group classes into tasks
    if args['path_to_task_groups'] is not None:
        with open(args['path_to_task_groups'], 'r') as f: 
            task_groups = json.load(f)
    else: 
        num_classes = args.get('num_classes', 10)
        all_classes = list(range(num_classes))

        random.shuffle(all_classes)

        task_groups = [all_classes[i : min(i + classes_per_task, len(all_classes))] for i in range(0, len(all_classes), classes_per_task)]
    
    num_tasks = len(task_groups)
    print(f"num_tasks: {num_tasks}, task_groups: {task_groups}", flush=True)

    # Count samples per class in the dataset
    class_counts = Counter(test_dataset.targets.tolist())

    # Aggregate counts based on task groups
    task_group_counts = {}
    for task_id, task_classes in enumerate(task_groups):
        task_count = sum(class_counts[class_label] for class_label in task_classes)
        task_group_counts[f"Task {task_id + 1}"] = task_count

    # Print results
    print("\nNumber of samples per task group:")
    for task, count in task_group_counts.items():
        print(f"{task}: {count}", flush=True)

    ci_results = {
        "average_accuracy": [],
        "average_incremental_accuracy": [],
        "forgetting_measure": [],
        "backward_transfer": [],
        "intermediate_representations": []
    }

    for ci_iteration in range(args['ci_iterations']): 
        loader = ModuleLoader()
        model = loader.load_model(args['model_name'].format(max(classes_per_task, 2), num_tasks))
        model = model.to(device)

        # Training and evaluation
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)

        print(f"ci_iteration: {ci_iteration + 1}/{args['ci_iterations']}", flush=True)
        results = []
        curr_iteration_intermediate_representations = []

        # randomly shuffle the task_groups -> isolate the effect of task grouping and not ordering
        if args['path_to_task_groups'] is not None and args['shuffle_task_groups']:
            random.shuffle(task_groups) 

        if args['path_to_task_groups'] is None:
            random.shuffle(all_classes)
            task_groups = [all_classes[i : min(i + classes_per_task, len(all_classes))] for i in range(0, len(all_classes), classes_per_task)]

        print(f"ci_iteration: {ci_iteration}, task_groups: {task_groups}")
        for task_id, task_classes in enumerate(task_groups):
            print(f"\nTraining on Task {task_id + 1}/{num_tasks} with classes {task_classes}", flush=True)

            # Remap train and test datasets for the current task
            task_train_dataset = remap_labels(train_dataset, task_classes)
            print(f"len(task_train_dataset): {len(task_train_dataset)}", flush=True)

            train_loader = DataLoader(task_train_dataset, batch_size=args['train_mb_size'], shuffle=True)

            # Train the model on the current task
            model.train()
            for epoch in range(args['train_epochs']):
                running_loss = 0.0

                for images, labels in train_loader:
                    # Dynamically augment batch if classes_per_task == 1
                    if len(task_classes) == 1:
                        # print("Adding random images to the batch for classes_per_task == 1", flush=True)

                        # Determine number of random samples to add
                        num_samples_to_add = len(images)

                        # Generate random MNIST-like images
                        random_images = torch.randint(
                            0, 256, (num_samples_to_add, 1, 28, 28), dtype=torch.uint8
                        ).float() / 255.0
                        random_labels = torch.ones(num_samples_to_add, dtype=torch.long)  # Labels are all 1

                        # Concatenate random images and labels to the batch
                        images = torch.cat((images, random_images), dim=0)
                        labels = torch.cat((labels, random_labels), dim=0)

                    # Move images and labels to the device
                    images, labels = images.to(device), labels.to(device)

                    # Training step
                    optimizer.zero_grad()
                    outputs = model(images, task_label=task_id)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                print(f"Epoch {epoch + 1}/{args['train_epochs']}, Loss: {running_loss / len(train_loader)}", flush=True)

            # Validation and analytics
            validation_scores, intermediate_representations = validation(model, task_groups, test_dataset, args, device)

            validation_scores_dict = {}
            template = "Top1_Acc_Exp/Exp{:03d}"
            for idx, score in enumerate(validation_scores):
                validation_scores_dict[template.format(idx)] = score

            print(f"validation_scores_dict: {validation_scores_dict}", flush=True)
            results.append(validation_scores_dict)

            print(f"intermediate_representations.shape: {intermediate_representations.shape}")
            curr_iteration_intermediate_representations.append(intermediate_representations)

        # Compute metrics for this iteration
        accuracy_matrix = compute_accuracy_matrix(results, num_tasks)
        average_accuracy = compute_average_accuracy_matrix(accuracy_matrix)
        average_incremental_accuracy = compute_average_incremental_accuracy_matrix(average_accuracy)
        forgetting_measure = compute_forgetting_matrix(accuracy_matrix)
        backward_transfer = compute_backward_transfer_matrix(accuracy_matrix)

        # Collect metrics for this iteration
        ci_results["average_accuracy"].append(average_accuracy)
        ci_results["average_incremental_accuracy"].append(average_incremental_accuracy)
        ci_results["forgetting_measure"].append(forgetting_measure)
        ci_results["backward_transfer"].append(backward_transfer)
        ci_results["intermediate_representations"].append(curr_iteration_intermediate_representations)

    metrics_stats = {}
    last_elements = torch.tensor([t[-1].item() for t in ci_results["average_accuracy"]])
    last_elements = torch.stack([t[-1] for t in ci_results["average_accuracy"]])

    metrics_stats["average_final_accuracy"] = last_elements

    for metric_name, values in ci_results.items():
        if metric_name == "intermediate_representations":
            pass
        else:
            # Process other metrics as before
            metric_tensor = torch.stack(values)
            metrics_stats[f"{metric_name}_mean"] = torch.mean(metric_tensor, dim=0)
            if args['ci_iterations'] > 1:
                metrics_stats[f"{metric_name}_std"] = torch.std(metric_tensor, dim=0)

    if args['ci_iterations'] == 1:  
        metrics_stats["average_accuracy_std"] = torch.zeros_like(torch.tensor(metrics_stats["average_accuracy_mean"]))
        metrics_stats["average_incremental_accuracy_std"] = torch.zeros_like(torch.tensor(metrics_stats["average_incremental_accuracy_mean"]))
        metrics_stats["forgetting_measure_std"] = torch.zeros_like(torch.tensor(metrics_stats["forgetting_measure_mean"]))
        metrics_stats["backward_transfer_std"] = torch.zeros_like(torch.tensor(metrics_stats["backward_transfer_mean"]))
        
    file_path = args['res_file_template'].format(args['model_name'].format(classes_per_task, num_tasks), args['classes_per_task'], args['similarity_metric'], args['grouping'], args['ordering'], args['train_epochs'])

    save_ci_results_to_file(
        file_path,
        metrics_stats["average_accuracy_mean"], metrics_stats.get("average_accuracy_std", None),
        metrics_stats["average_incremental_accuracy_mean"], metrics_stats.get("average_incremental_accuracy_std", None),
        metrics_stats["forgetting_measure_mean"], metrics_stats.get("forgetting_measure_std", None),
        metrics_stats["backward_transfer_mean"], metrics_stats.get("backward_transfer_std", None),
        metrics_stats["average_final_accuracy"],
        ci_results["intermediate_representations"]
    )

    # output_dir = os.path.dirname(file_path)
    # plot_averaged_representations(metrics_stats["intermediate_representations_mean"], output_dir)

    print(f"Final aggregated results saved to {file_path}", flush=True)

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
