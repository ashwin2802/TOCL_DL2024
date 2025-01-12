import json
import itertools
import numpy as np
import random

# Load task grouping and similarity matrix
path_to_similarity_matrix = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/CIFAR-100_cambdridge_similarity_task_aware_resnet-18-2-100_epochs_10.json"
path_to_task_grouping_file = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/hierarchical_partition_CIFAR-100_cambdridge_similarity_task_aware_resnet-18-2-20_epochs_10_with_normalization.json"

with open(path_to_task_grouping_file, 'r') as f: 
    task_grouping = json.load(f)  # List of lists, where each inner list contains class indices for a task group
    
    # task_grouping = task_grouping["4"]

with open(path_to_similarity_matrix, 'r') as f: 
    similarity_matrix = np.array(json.load(f))  # Convert the list of lists to a NumPy array

# Function to compute the cost for a given permutation of task groups
def compute_cost(permutation, task_grouping, similarity_matrix):
    total_sum = 0
    # Iterate over adjacent groups in the permutation
    for i in range(len(permutation) - 1):
        current_group = permutation[i]
        next_group = permutation[i + 1]
        # Sum M[k, l] for all k in current_group and l in next_group
        for k in current_group:
            for l in next_group:
                total_sum += similarity_matrix[k, l]
    return total_sum

# Find the permutation that minimizes the cost
max_cost = -1.0 * float('inf')
best_permutation = None

# Iterate over all permutations of task groups
for perm in itertools.permutations(task_grouping):
    cost = compute_cost(perm, task_grouping, similarity_matrix)
    if cost > max_cost:
        max_cost = cost
        best_permutation = perm

# Print the result
path_to_optimal_ordering = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/max_cut_optimal_partition_CIFAR-100_cambdridge_similarity_task_aware_resnet_mnist-18-2-10_epochs_10_with_normalization.json"
with open(path_to_optimal_ordering, 'w') as f: 
    json.dump(best_permutation, f)
