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

from common import *

from models.model_loader import ModuleLoader
    
def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = get_dataset('cifar-10')
    classes_per_task = args['classes_per_task']  # Number of classes per task

    ci_results = {
        "average_accuracy": [],
        "average_incremental_accuracy": [],
        "forgetting_measure": [],
        "backward_transfer": [],
        "intermediate_representations": []
    }

    for ci_iteration in range(args['ci_iterations']): 
        # load the task_groups for the run
        if args['path_to_task_groups'] is not None:
            with open(args['path_to_task_groups'], 'r') as f: 
                task_groups = json.load(f)

            if args['shuffle_task_groups']: 
                random.shuffle(task_groups)
        else: 
            # If no path_to_task_groups is provided, then random classes_per_task-subsets are created
            num_classes = args.get('num_classes', 10)
            all_classes = list(range(num_classes))

            random.shuffle(all_classes)

            task_groups = [all_classes[i : min(i + classes_per_task, len(all_classes))] for i in range(0, len(all_classes), classes_per_task)]
            
        num_tasks = len(task_groups)
        print(f"num_tasks: {num_tasks}, task_groups: {task_groups}", flush=True)

        print_class_counts(test_dataset.targets, task_groups)

        loader = ModuleLoader()
        model = loader.load_model(args['model_name'].format(classes_per_task, num_tasks))
        model = model.to(device)

        # Training and evaluation
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)

        print(f"ci_iteration: {ci_iteration + 1}/{args['ci_iterations']}", flush=True)
        results = []
        curr_iteration_intermediate_representations = []

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
                    # print(f"task_classes: {task_classes}, labels: {labels}")
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images, task_label=task_id)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch {epoch + 1}/{args['train_epochs']}, Loss: {running_loss / len(train_loader)}", flush=True)

            # intermediate_representations.shape: [5, 256]
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
