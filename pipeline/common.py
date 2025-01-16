import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torch.nn as nn

def get_dataset(dataset_name, data_dir='~/.torch/datasets'):
    """
    Returns train and test datasets for MNIST, CIFAR-10, or CIFAR-100 based on input.
    
    Parameters:
        dataset_name (str): "mnist-10", "cifar-10", or "cifar-100".
        data_dir (str): Directory where the datasets will be stored/downloaded.
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    if dataset_name == "cifar-100":
        # Transforms for CIFAR-100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ])

        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "mnist-10":
        # Transforms for MNIST
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "cifar-10":
        # Transforms for CIFAR-10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}. Use 'mnist-10', 'cifar-10', or 'cifar-100'.")

    return train_dataset, test_dataset

def remap_labels(dataset, task_classes):
    """
    Remap the labels in a deep copy of the dataset to fit within the range [0, classes_per_task-1].

    Parameters:
        dataset (Dataset): The dataset to remap labels for.
        task_classes (list): List of class labels for the current task.

    Returns:
        Subset: A subset of the copied dataset with labels remapped to the range [0, classes_per_task-1].
    """

    from copy import deepcopy
    from torch.utils.data import Subset
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
    
def print_class_counts(targets, task_groups):
    from collections import Counter

    class_counts = Counter(targets)

    # Aggregate counts based on task groups
    task_group_counts = {}
    for task_id, task_classes in enumerate(task_groups):
        task_count = sum(class_counts[class_label] for class_label in task_classes)
        task_group_counts[f"Task {task_id + 1}"] = task_count

    # Print results
    print("\nNumber of samples per task group:", flush=True)
    for task, count in task_group_counts.items():
        print(f"{task}: {count}")