import numpy as np
import json
import math

def greedy_partition_with_limit(M, num_classes, classes_per_task):
    """
    Create partitions using a greedy approach with a limit on the number of partitions.

    Args:
        M (np.ndarray): A square matrix of size [num_classes, num_classes].
        num_classes (int): Total number of classes.
        classes_per_task (int): Number of classes per task in each partition.

    Returns:
        list of lists: A greedy partition of the classes with a limited number of partitions.
    """
    # Compute the maximum number of partitions allowed
    max_partitions = math.ceil(num_classes / classes_per_task)

    # Create a list of all (i, j) pairs with their scores M[i, j] + M[j, i]
    scores = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                scores.append((i, j, M[i, j] + M[j, i]))

    # Sort scores in descending order
    scores.sort(key=lambda x: x[2], reverse=True)

    # Initialize partitions and keep track of which class is assigned
    partitions = [[] for _ in range(max_partitions)]
    assigned = set()

    # Process pairs in descending order of their scores
    for i, j, _ in scores:
        if i in assigned and j in assigned:
            continue  # Both classes are already assigned
        elif i in assigned:
            # If i is assigned, try to fit j into the same partition
            for partition in partitions:
                if len(partition) < classes_per_task and i in partition:
                    partition.append(j)
                    assigned.add(j)
                    break
        elif j in assigned:
            # If j is assigned, try to fit i into the same partition
            for partition in partitions:
                if len(partition) < classes_per_task and j in partition:
                    partition.append(i)
                    assigned.add(i)
                    break
        else:
            # If neither i nor j is assigned, assign them to the first partition with space
            for partition in partitions:
                if len(partition) + 2 <= classes_per_task:
                    partition.append(i)
                    partition.append(j)
                    assigned.add(i)
                    assigned.add(j)
                    break

    # Handle remaining unassigned classes
    unassigned = [c for c in range(num_classes) if c not in assigned]
    for cls in unassigned:
        # Try to fit into an existing partition with space
        for partition in partitions:
            if len(partition) < classes_per_task:
                partition.append(cls)
                break

    # Remove any empty partitions
    partitions = [p for p in partitions if p]

    return partitions

def compute_partition_sum_greedy_with_limit(M, num_classes, classes_per_task):
    """
    Compute the sum of M[i, j] for all (i, j) where i and j belong to the same set 
    in a greedy partition of num_classes into sets of size classes_per_task,
    with a limit on the number of partitions.

    Args:
        M (np.ndarray): A square matrix of size [num_classes, num_classes].
        num_classes (int): Total number of classes.
        classes_per_task (int): Number of classes per task in each partition.

    Returns:
        float: The computed sum for the greedy partition.
    """
    partitions = greedy_partition_with_limit(M, num_classes, classes_per_task)

    # Compute the sum for the greedy partition
    total_sum = 0
    for subset in partitions:
        for i in subset:
            for j in subset:
                if i != j:  # Avoid self-pairs
                    total_sum += M[i, j]

    return total_sum, partitions

# Paths and file handling
path_to_similarity_matrix = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/CIFAR-100_cambdridge_similarity_task_aware_resnet-18-2-20_epochs_10_without_normalization.json"
similarity_matrix_stem_name = path_to_similarity_matrix.split('/')[-1].split('.')[0]
path_to_final_partition = f"/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/greedy_partition_{similarity_matrix_stem_name}.json"

print(f"saving to: {path_to_final_partition}")
with open(path_to_similarity_matrix, 'r') as f: 
    similarity_matrix = json.load(f)

num_classes = len(similarity_matrix)
classes_per_task = 5

# Convert JSON list of lists to a NumPy array
M = np.array(similarity_matrix)

# Compute greedy partition with a limit on the number of partitions
result, partitions = compute_partition_sum_greedy_with_limit(M, num_classes, classes_per_task)

# Save the result
with open(path_to_final_partition, 'w') as f: 
    json.dump(partitions, f)

print(f"Total sum for the greedy partition: {result}")
print(f"Greedy Partitions: {partitions}")
