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
    # Validate on all tasks seen so far
    print(f"\nValidating on all tasks...")
    model.eval()
    for eval_task_id, eval_task_classes in enumerate(task_groups):
        print(f"eval_task_id: {eval_task_id}, eval_task_classes: {eval_task_classes}")
        eval_test_dataset = remap_labels(test_dataset, eval_task_classes)
        print(f"len(eval_test_dataset): {len(eval_test_dataset)}")
        eval_loader = DataLoader(eval_test_dataset, batch_size=args['test_mb_size'], shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_label=eval_task_id)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Task {eval_task_id + 1} Test Accuracy: {accuracy:.2f}%")
        results.append(accuracy)

    return results
    
def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes_per_task = args['classes_per_task']  # Number of classes per task

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

    train_dataset = datasets.CIFAR100(root=args['data_dir'], train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=args['data_dir'], train=False, download=True, transform=test_transform)

    # Group classes into tasks
    if args['path_to_task_groups'] is not None:
        with open(args['path_to_task_groups'], 'r') as f: 
            task_groups = json.load(f)
    else: 
        num_classes = args.get('num_classes', 100)
        all_classes = list(range(num_classes))

        random.shuffle(all_classes)

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

    ci_results = {
        "average_accuracy": [],
        "average_incremental_accuracy": [],
        "forgetting_measure": [],
        "backward_transfer": []
    }

    for ci_iteration in range(args['ci_iterations']): 
        loader = ModuleLoader()
        model = loader.load_model(args['model_name'].format(classes_per_task, num_tasks))
        model = model.to(device)

        # Training and evaluation
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)

        print(f"ci_iteration: {ci_iteration + 1}/{args['ci_iterations']}")
        results = []

        # randomly shuffle the task_groups -> isolate the effect of task grouping and not ordering
        if args['path_to_task_groups'] is not None:
            random.shuffle(task_groups) 
        else: 
            random.shuffle(all_classes)
            task_groups = [all_classes[i : min(i + classes_per_task, len(all_classes))] for i in range(0, len(all_classes), classes_per_task)]

        for task_id, task_classes in enumerate(task_groups):
            print(f"\nTraining on Task {task_id + 1}/{num_tasks} with classes {task_classes}")

            # Remap train and test datasets for the current task
            task_train_dataset = remap_labels(train_dataset, task_classes)
            print(f"len(task_train_dataset): {len(task_train_dataset)}")

            train_loader = DataLoader(task_train_dataset, batch_size=args['train_mb_size'], shuffle=True)

            # Train the model on the current task
            model.train()
            for epoch in range(args['train_epochs']):
                running_loss = 0.0
                for images, labels in train_loader:
                    # print(f"task_classes: {task_classes}, labels: {labels}")
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images, task_label=task_id)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch {epoch + 1}/{args['train_epochs']}, Loss: {running_loss / len(train_loader)}")

            validation_scores = validation(model, task_groups, test_dataset, args, device)
            validation_scores_dict = {}
            template = "Top1_Acc_Exp/Exp{:03d}"
            for idx, score in enumerate(validation_scores): 
                validation_scores_dict[template.format(idx)] = score

            print(f"validation_scores_dict: {validation_scores_dict}")
            results.append(validation_scores_dict)

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

    metrics_stats = {}
    for metric_name, values in ci_results.items():
        # metric_tensor = torch.tensor(values)
        # Stack the list of tensors into a single tensor
        metric_tensor = torch.stack(values)  # This assumes all tensors have the same shape

        metrics_stats[f"{metric_name}_mean"] = torch.mean(metric_tensor, dim=0)  # Mean across iterations
        metrics_stats[f"{metric_name}_std"] = torch.std(metric_tensor, dim=0)    # Std across iterations
        print(f"{metric_name}: {values}")
        print(f"{metric_name}_mean: {metrics_stats[f'{metric_name}_mean']}")

    # Save aggregated results
    file_path = args['res_file_template'].format(args['model_name'].format(classes_per_task, num_tasks), args['classes_per_task'], args['similarity_metric'], args['grouping'], args['train_epochs'])
    save_ci_results_to_file(
        file_path,
        metrics_stats["average_accuracy_mean"], metrics_stats["average_accuracy_std"],
        metrics_stats["average_incremental_accuracy_mean"], metrics_stats["average_incremental_accuracy_std"],
        metrics_stats["forgetting_measure_mean"], metrics_stats["forgetting_measure_std"],
        metrics_stats["backward_transfer_mean"], metrics_stats["backward_transfer_std"]
    )

    print(f"Final aggregated results saved to {file_path}")

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
