import json
import itertools
import numpy as np
import random

from tqdm import tqdm

# Load task grouping and similarity matrix
path_to_similarity_matrix = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/CIFAR-100_cambdridge_similarity_task_aware_resnet-18-2-100_epochs_10.json"
path_to_task_grouping_file = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/hierarchical_min_partition_CIFAR-100_cambdridge_similarity_task_aware_resnet-18-2-20_epochs_10_with_normalization.json"

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

optimal_ordering = {}

for key, value in task_grouping.items(): 
    print(f"key: {key}, valeu: {value}")
    if key == '2' or key == '4': 
        continue
    # Find the permutation that minimizes the cost
    min_cost = float('inf')
    best_permutation = None

    # Iterate over all permutations of task groups
    for perm in tqdm(itertools.permutations(value)):
        cost = compute_cost(perm, value, similarity_matrix)
        if cost < min_cost:
            min_cost = cost
            best_permutation = perm

    optimal_ordering[key] = best_permutation

# Print the result
path_to_optimal_ordering = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/min_cut_optimal_partition_CIFAR-100_cambdridge_min_similarity_task_aware_resnet-18-2-50_epochs_10_with_normalization.json"
with open(path_to_optimal_ordering, 'w') as f: 
    json.dump(optimal_ordering, f)
