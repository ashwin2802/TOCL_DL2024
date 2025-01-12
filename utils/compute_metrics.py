import torch

def compute_accuracy_matrix(results: dict, num_tasks: int): 
    # Step 1: Create a tensor for the accuracy matrix
    accuracy_matrix = torch.zeros((num_tasks, num_tasks))

    # Step 2: Process results and fill the matrix
    for i in range(num_tasks):
        task_results = results[i]  # Access results for task `i`
        task_results_polished = {}
        for key, value in task_results.items():
            if key.startswith('Top1_Acc_Exp'): 
                task_results_polished[key] = value

        task_keys_polished = sorted(task_results_polished.keys(), key=lambda k: int(k.split("/")[-1][-3:]))  # Sort keys by task index
        
        for j, key in enumerate(task_keys_polished):
            accuracy_matrix[j, i] = task_results_polished[key]  # Fill values in column `i`

    return accuracy_matrix

def compute_average_accuracy_matrix(accuracy_matrix: torch.Tensor): 
    # Compute the AA_k metric
    average_accuracy = torch.zeros(accuracy_matrix.shape[0])

    for i in range(accuracy_matrix.shape[0]): # either dimension is fine, as the accuracy matrix is square
        tmp = 0.0
        for j in range(i + 1): 
            tmp += accuracy_matrix[j, i]

        average_accuracy[i] = tmp / (i + 1)

    return average_accuracy

def compute_average_incremental_accuracy_matrix(average_accuracy_matrix: torch.Tensor):
    average_incremental_accuracy = torch.zeros(average_accuracy_matrix.shape[0])

    for i in range(average_accuracy_matrix.shape[0]): 
        average_incremental_accuracy[i] = torch.sum(average_accuracy_matrix[: i + 1]) / (i + 1)

    return average_incremental_accuracy

def compute_forgetting_matrix(accuracy_matrix: torch.Tensor): 
    forgetting_matrix = torch.zeros(accuracy_matrix.shape[0])

    for k in range(1, accuracy_matrix.shape[0]): 
        cum_max_diff = 0.0
        for j in range(k): 
            max_diff = 0.0
            for i in range(k): 
                max_diff = max(max_diff, accuracy_matrix[j, i].item() - accuracy_matrix[j, k].item())

            cum_max_diff += max_diff
        
        forgetting_matrix[k] = cum_max_diff / k
    
    return forgetting_matrix

def compute_backward_transfer_matrix(accuracy_matrix: torch.Tensor):
    backward_transfer = torch.zeros(accuracy_matrix.shape[0])

    for k in range(1, accuracy_matrix.shape[0]): 
        tmp = 0.0
        for j in range(k): 
            tmp += accuracy_matrix[j, k] - accuracy_matrix[j, j]

        backward_transfer[k] = tmp / k

    return backward_transfer

import json
import torch

# Assuming args.res_file is the file path where results will be saved
def save_results_to_file(file_path, accuracy_matrix, average_accuracy, average_incremental_accuracy, forgetting_measure, backward_transfer):
    """
    Save results to a JSON file instead of printing.

    Parameters:
    - args: Argument object containing `res_file` (output file path).
    - accuracy_matrix: PyTorch 2D tensor of accuracies.
    - average_accuracy: PyTorch 1D tensor or scalar tensor.
    - average_incremental_accuracy: PyTorch 1D tensor or scalar tensor.
    - forgetting_measure: PyTorch 1D tensor or scalar tensor.
    - backward_transfer: PyTorch 1D tensor or scalar tensor.
    """
    # Convert tensors to Python-native types
    results = {
        "accuracy_matrix": accuracy_matrix.tolist(),  # Convert 2D tensor to nested list
        "average_accuracy": average_accuracy.tolist() if average_accuracy.dim() > 0 else average_accuracy.item(),
        "average_incremental_accuracy": average_incremental_accuracy.tolist() if average_incremental_accuracy.dim() > 0 else average_incremental_accuracy.item(),
        "forgetting_measure": forgetting_measure.tolist() if forgetting_measure.dim() > 0 else forgetting_measure.item(),
        "backward_transfer": backward_transfer.tolist() if backward_transfer.dim() > 0 else backward_transfer.item()
    }

    # Save results to the specified file in JSON format
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results successfully saved to {file_path}")

# Assuming args.res_file is the file path where results will be saved
def save_ci_results_to_file(file_path, 
    average_accuracy_mean, average_accuracy_std,
    average_incremental_accuracy_mean, average_incremental_accuracy_std,
    forgetting_measure_mean, forgetting_measure_std, 
    backward_transfer_mean, backward_transfer_std, average_final_accuracy,
    intermediate_representations = None):
    """
    Save results to a JSON file instead of printing.

    Parameters:
    - file_path: Path to save the results.
    - average_accuracy_{mean/std}: PyTorch 1D tensor or scalar tensor.
    - average_incremental_accuracy_{mean/std}: PyTorch 1D tensor or scalar tensor.
    - forgetting_measure_{mean/std}: PyTorch 1D tensor or scalar tensor.
    - backward_transfer_{mean/std}: PyTorch 1D tensor or scalar tensor.
    """
    # Convert tensors to Python-native types
    results = {
        "average_accuracy_mean": average_accuracy_mean.tolist() if average_accuracy_mean.dim() > 0 else average_accuracy_mean.item(),
        "average_accuracy_std": average_accuracy_std.tolist() if average_accuracy_std.dim() > 0 else average_accuracy_std.item(),
        "average_incremental_accuracy_mean": average_incremental_accuracy_mean.tolist() if average_incremental_accuracy_mean.dim() > 0 else average_incremental_accuracy_mean.item(),
        "average_incremental_accuracy_std": average_incremental_accuracy_std.tolist() if average_incremental_accuracy_std.dim() > 0 else average_incremental_accuracy_std.item(),
        "forgetting_measure_mean": forgetting_measure_mean.tolist() if forgetting_measure_mean.dim() > 0 else forgetting_measure_mean.item(),
        "forgetting_measure_std": forgetting_measure_std.tolist() if forgetting_measure_std.dim() > 0 else forgetting_measure_std.item(),
        "backward_transfer_mean": backward_transfer_mean.tolist() if backward_transfer_mean.dim() > 0 else backward_transfer_mean.item(),
        "backward_transfer_std": backward_transfer_std.tolist() if backward_transfer_std.dim() > 0 else backward_transfer_std.item(),
        "average_final_accuracy": average_final_accuracy.tolist() if average_final_accuracy.dim() > 0 else average_final_accuracy.item(),         
    }

    if intermediate_representations is not None: 
        results.update({
            "intermediate_representations": [representation.tolist() for iteration in intermediate_representations for representation in iteration]
        })
        
    # Save results to the specified file in JSON format
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results successfully saved to {file_path}")
