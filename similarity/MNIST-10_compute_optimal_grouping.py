import itertools
import numpy as np
import json
from tqdm import tqdm

def compute_min_partition_sum_exact(M, num_classes, classes_per_task):
    """
    Compute the maximum sum of M[i, j] across all possible partitions of num_classes into groups of size classes_per_task.

    Args:
        M (np.ndarray): A square matrix of size [num_classes, num_classes].
        num_classes (int): Total number of classes.
        classes_per_task (int): Number of classes per task in the partition.

    Returns:
        float: The maximum sum across all possible partitions.
        list of lists: The optimal partition.
    """
    # Generate all possible partitions
    class_indices = list(range(num_classes))
    min_sum = float('inf')
    min_partition = None

    # Generate all possible combinations of `classes_per_task` elements from `class_indices`
    all_combinations = list(itertools.combinations(class_indices, classes_per_task))

    # Iterate through all possible partitions
    for combination in tqdm(itertools.combinations(all_combinations, num_classes // classes_per_task), desc="Iterating combinations"):
        # Check if the combination is disjoint
        flat_partition = [item for subset in combination for item in subset]
        if len(set(flat_partition)) == num_classes:  # Ensure disjoint sets
            partition_sum = 0
            for subset in combination:
                # Compute sum of M[i, j] for all pairs (i, j) in the subset
                for i in subset:
                    for j in subset:
                        if i != j:  # Avoid self-pairs
                            partition_sum += M[i, j]
            # Update the maximum sum if the current partition sum is larger
            if partition_sum < min_sum:
                min_sum = partition_sum
                min_partition = combination

    return min_sum, [list(group) for group in min_partition]

def compute_max_partition_sum_exact(M, num_classes, classes_per_task):
    """
    Compute the maximum sum of M[i, j] across all possible partitions of num_classes into groups of size classes_per_task.

    Args:
        M (np.ndarray): A square matrix of size [num_classes, num_classes].
        num_classes (int): Total number of classes.
        classes_per_task (int): Number of classes per task in the partition.

    Returns:
        float: The maximum sum across all possible partitions.
        list of lists: The optimal partition.
    """
    # Generate all possible partitions
    class_indices = list(range(num_classes))
    max_sum = float('-inf')
    max_partition = None

    # Generate all possible combinations of `classes_per_task` elements from `class_indices`
    all_combinations = list(itertools.combinations(class_indices, classes_per_task))

    # Iterate through all possible partitions
    for combination in tqdm(itertools.combinations(all_combinations, num_classes // classes_per_task), desc="Iterating combinations"):
        # Check if the combination is disjoint
        flat_partition = [item for subset in combination for item in subset]
        if len(set(flat_partition)) == num_classes:  # Ensure disjoint sets
            partition_sum = 0
            for subset in combination:
                # Compute sum of M[i, j] for all pairs (i, j) in the subset
                for i in subset:
                    for j in subset:
                        if i != j:  # Avoid self-pairs
                            partition_sum += M[i, j]
            # Update the maximum sum if the current partition sum is larger
            if partition_sum > max_sum:
                max_sum = partition_sum
                max_partition = combination

    return max_sum, [list(group) for group in max_partition]

"""
Example: 
path_to_similarity_matrix = "./similarity_matrices/MNIST-10_cambdridge_similarity_task_aware_simpleMLP-784-256-3-2-10_epochs_10.json"
path_to_opt_grouping = "./similarity_matrices/optimal_{}_partition_MNIST-10_cambdridge_similarity_task_aware_simpleMLP-784-256-3-2-10_epochs_10.json"
"""
path_to_similarity_matrix = "path/to/similarity/matrix.json"

# Adapt the path and make sure you have the {} placeholder in the filename, so that it gets formatted for min- and max-grouping 
path_to_opt_grouping = "path/to/optimal_{}_grouping.json"

with open(path_to_similarity_matrix, 'r') as f: 
    similarity_matrix = json.load(f)
    
num_classes = len(similarity_matrix)
classes_per_task = 2

# Convert JSON list of lists to a NumPy array
M = np.array(similarity_matrix)

# Compute true optimal partition
result, partitions = compute_max_partition_sum_exact(M, num_classes, classes_per_task)

# Save the result
with open(path_to_opt_grouping.format('max'), 'w') as f: 
    json.dump(partitions, f)

print(f"optimal max-grouping saved at {path_to_opt_grouping.format('max')}")
# Compute true optimal partition
result, partitions = compute_min_partition_sum_exact(M, num_classes, classes_per_task)

# Save the result
with open(path_to_opt_grouping.format('min'), 'w') as f: 
    json.dump(partitions, f)

print(f"optimal min-grouping saved at {path_to_opt_grouping.format('min')}")