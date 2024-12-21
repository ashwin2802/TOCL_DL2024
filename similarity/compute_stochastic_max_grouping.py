import itertools
import numpy as np
from tqdm import tqdm
import random
import json

def sample_partitions(classes, classes_per_task, num_samples):
    """
    Sample a fixed number of partitions of `classes` into sets of size `classes_per_task`.
    The last set in a partition may be smaller if len(classes) % classes_per_task != 0.
    
    Args:
        classes (list): List of class indices.
        classes_per_task (int): Size of each subset in the partition.
        num_samples (int): Number of partitions to sample.

    Returns:
        list of lists: A list of sampled partitions.
    """
    partitions = []
    for _ in tqdm(range(num_samples)):
        random.shuffle(classes)
        partition = [classes[i:i + classes_per_task] for i in range(0, len(classes), classes_per_task)]
        partitions.append(partition)
    return partitions

def compute_max_partition_sum(M, num_classes, classes_per_task, num_samples):
    """
    Compute the maximum sum of M[i, j] across sampled partitions of num_classes.

    Args:
        M (np.ndarray): A square matrix of size [num_classes, num_classes].
        num_classes (int): Total number of classes.
        classes_per_task (int): Number of classes per task in the partition.
        num_samples (int): Number of partitions to sample.

    Returns:
        float: The maximum sum across sampled partitions.
    """
    class_indices = list(range(num_classes))
    sampled_partitions = sample_partitions(class_indices, classes_per_task, num_samples)

    # Initialize the maximum sum
    max_sum = float('-inf')
    max_partition = None

    # Iterate through sampled partitions
    for partition in tqdm(sampled_partitions):
        partition_sum = 0
        for subset in partition:
            # Compute sum of M[i, j] for all pairs (i, j) in the subset
            for i in subset:
                for j in subset:
                    if i != j:  # Avoid self-pairs
                        partition_sum += M[i, j]
        # Update the maximum sum if the current partition sum is larger
        if partition_sum > max_sum: 
            max_sum = partition_sum
            max_partition = partition
        else: 
            pass

    return max_sum, max_partition

# Paths and file handling
path_to_similarity_matrix = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/CIFAR-100_cambdridge_similarity_task_aware_resnet-18-2-20_epochs_10_without_normalization.json"
similarity_matrix_stem_name = path_to_similarity_matrix.split('/')[-1].split('.')[0]
path_to_final_partition = f"/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/stochastic_partition_{similarity_matrix_stem_name}.json"

print(f"saving to: {path_to_final_partition}")
with open(path_to_similarity_matrix, 'r') as f: 
    similarity_matrix = json.load(f)
    
num_classes = len(similarity_matrix)
classes_per_task = 5
num_samples = int(1e7)

# Convert JSON list of lists to a NumPy array
M = np.array(similarity_matrix)

# Compute greedy partition with a limit on the number of partitions
result, partitions = compute_max_partition_sum(M, num_classes, classes_per_task, num_samples)

# Save the result
with open(path_to_final_partition, 'w') as f: 
    json.dump(partitions, f)

print(f"Total sum for the greedy partition: {result}")
print(f"Greedy Partitions: {partitions}")

