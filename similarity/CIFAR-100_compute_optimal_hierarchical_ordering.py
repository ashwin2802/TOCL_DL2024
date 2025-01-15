import json
import itertools
import numpy as np
from tqdm import tqdm

def compute_cumulative_cost(permutation, similarity_matrix):
    total_sum = 0
    for i in range(len(permutation)):
        current_group = permutation[i]
        for j in range(i + 1, len(permutation)):
            next_group = permutation[j]
            for k in current_group:
                for l in next_group:
                    total_sum += similarity_matrix[k, l]
    return total_sum

def compute_cost(permutation, similarity_matrix):
    total_sum = 0
    for i in range(len(permutation) - 1):
        current_group = permutation[i]
        next_group = permutation[i + 1]
        for k in current_group:
            for l in next_group:
                total_sum += similarity_matrix[k, l]
    return total_sum

def find_optimal_ordering(task_grouping, similarity_matrix, mode, cost_function):
    optimal_ordering = {}
    for key, value in task_grouping.items():
        print(f"Processing key: {key}, value: {value}")

        if key in ['2', '4']:  # Skip large groups for CIFAR-100
            continue

        if mode == 'min':
            best_cost = float('inf')
        else:
            best_cost = -float('inf')

        best_permutation = None
        for perm in tqdm(itertools.permutations(value)):
            cost = cost_function(perm, similarity_matrix)
            if (mode == 'min' and cost < best_cost) or (mode == 'max' and cost > best_cost):
                best_cost = cost
                best_permutation = perm

        optimal_ordering[key] = best_permutation

    return optimal_ordering

if __name__ == "__main__":

    """
    Example: 
    path_to_similarity_matrix = "./similarity_matrices/CIFAR-100_cambdridge_similarity_with_grad_prod_task_aware_resnet-18-2-50_epochs_10.json"
    path_to_opt_grouping = "./similarity_matrices/hierarchical_min_partition_CIFAR-100_cambdridge_similarity_with_grad_prod_task_aware_resnet-18-2-50_epochs_10.json"
    path_to_opt_ordering = "./similarity_matrices/{}_cut_hierarchical_min_partition_CIFAR-100_cambdridge_similarity_with_grad_prod_task_aware_resnet-18-2-50_epochs_10.json"
    """

    # Define paths at the beginning
    path_to_similarity_matrix = "path/to/similarity/matrix.json"
    path_to_opt_grouping = "path/to/task/grouping.json"
    path_to_opt_ordering = "path/to/{}_optimal_ordering.json"
    
    # Load data
    with open(path_to_opt_grouping, 'r') as f:
        task_grouping = json.load(f)

    with open(path_to_similarity_matrix, 'r') as f:
        similarity_matrix = np.array(json.load(f))

    # Process for both modes and cost functions
    for mode, cost_func in [('min', compute_cost), ('max', compute_cost), 
                            ('min_cum', compute_cumulative_cost), ('max_cum', compute_cumulative_cost)]:
        print(f"Calculating optimal ordering for mode: {mode}")
        optimal_ordering = find_optimal_ordering(
            task_grouping,
            similarity_matrix,
            mode.split('_')[0],  # Extract mode (min or max)
            cost_func
        )

        # Save the results
        output_path = path_to_opt_ordering.format(mode)
        with open(output_path, 'w') as f:
            json.dump(optimal_ordering, f)
        print(f"Saved results to {output_path}")
