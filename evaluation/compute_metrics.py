import torch

def compute_accuracy_matrix(results: dict, num_tasks: int): 
    # Step 1: Create a tensor for the accuracy matrix
    accuracy_matrix = torch.zeros((num_tasks, num_tasks))

    # Step 2: Process results and fill the matrix
    for i in range(num_tasks):
        task_results = results[i]  # Access results for task `i`
        task_keys = sorted(task_results.keys(), key=lambda k: int(k.split("/")[-1][-3:]))  # Sort keys by task index
        
        for j, key in enumerate(task_keys):
            accuracy_matrix[j, i] = task_results[key]  # Fill values in column `i`

    return accuracy_matrix

def compute_average_accuracy_matrix(accuracy_matrix: torch.Tensor): 
    # Compute the AA_k metric
    average_accuracy = torch.zeros(accuracy_matrix.shape[0])

    for i in range(accuracy_matrix.shape[0]): # either dimension is fine, as the accuracy matrix is square
        tmp = 0.0
        for j in range(i + 1): 
            tmp += accuracy_matrix[j, i]

        average_accuracy[i] = tmp / (i + 1)

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
                max_diff = max(max_diff, accuracy_matrix[j, i].item())

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