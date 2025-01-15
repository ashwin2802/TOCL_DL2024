import json
import itertools
import numpy as np
from tqdm import tqdm

# Function to compute the cost for a given permutation of task groups
def compute_cost(permutation, task_grouping, similarity_matrix):
    total_sum = 0
    for i in range(len(permutation) - 1):
        current_group = permutation[i]
        next_group = permutation[i + 1]
        for k in current_group:
            for l in next_group:
                total_sum += similarity_matrix[k, l]
    return total_sum

# Function to compute the cumulative cost for a given permutation of task groups
def compute_cumulative_cost(permutation, task_grouping, similarity_matrix):
    total_sum = 0
    for i in range(len(permutation)):
        current_group = permutation[i]
        for j in range(i + 1, len(permutation)):
            next_group = permutation[j]
            for k in current_group:
                for l in next_group:
                    total_sum += similarity_matrix[k, l]
    return total_sum

if __name__ == "__main__":
    # Define paths

    """
    Example
    path_to_similarity_matrix = "./similarity_matrices/MNIST-10_cambdridge_similarity_task_aware_simpleMLP-784-256-3-2-10_epochs_10.json"
    path_to_task_grouping_file = "./similarity_matrices/optimal_max_partition_MNIST-10_cambdridge_similarity_task_aware_simpleMLP-784-256-3-2-10_epochs_10.json"
    template_path_output_ordering = "./similarity_matrices/{}_optimal_max_partition_MNIST-10_cambdridge_similarity_with_grad_product_task_aware_simpleMLP-784-256-3-2-10_epochs_10.json"
    """
    path_to_similarity_matrix = "path/to/similarity/matrix.json"
    path_to_task_grouping_file = "path/to/task/grouping.json"
    template_path_output_ordering = "./similarity_matrices/template_{}_file_path.json"

    # Load data
    with open(path_to_similarity_matrix, 'r') as f:
        similarity_matrix = np.array(json.load(f))

    with open(path_to_task_grouping_file, 'r') as f:
        task_grouping = json.load(f)

    # Iterate over modes for both sum and cumulative sum
    for mode, cost_function in [
        ("min", compute_cost),
        ("max", compute_cost),
        ("min_cum", compute_cumulative_cost),
        ("max_cum", compute_cumulative_cost),
    ]:
        print(f"Processing mode: {mode}")

        if "min" in mode:
            best_cost = float("inf")
        else:
            best_cost = -float("inf")

        best_permutation = None

        # Iterate over all permutations of task groups
        for perm in tqdm(itertools.permutations(task_grouping)):
            cost = cost_function(perm, task_grouping, similarity_matrix)
            if ("min" in mode and cost < best_cost) or ("max" in mode and cost > best_cost):
                best_cost = cost
                best_permutation = perm

        # Save the result
        output_path = template_path_output_ordering.format(mode)
        with open(output_path, 'w') as f:
            json.dump(best_permutation, f)
        print(f"Saved results to {output_path}")
