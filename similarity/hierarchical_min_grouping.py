import cvxpy as cp
import numpy as np
import math
from tqdm import tqdm
import time
import json

import cvxpy as cp

def minimize_pairwise_similarity(similarity_matrix):
    n = len(similarity_matrix)  # Number of elements

    # Decision variables: x[i][j] is binary, indicating whether task i is in subset j
    x = cp.Variable((n, n), integer=True)

    # Objective function: Minimize the sum of pairwise similarities within subsets
    objective = cp.Minimize(cp.sum(cp.multiply(similarity_matrix, x)))

    # Constraints
    constraints = []

    for i in range(n):
        constraints.append(cp.sum(x[i, :]) == 1)  # Each element belongs to exactly one subset
        constraints.append(x[i, i] == 0)

    # Constraint: If x[i,j] = 1 and x[j,k] = 1, then x[i,k] = 1 (transitivity constraint for triangular matrix)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                constraints.append(x[i, j] + x[j, k] - x[i, k] <= 1)

    constraints.append(x == x.T)
    constraints.append(x >= 0)
    constraints.append(x <= 1)

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    try:
        problem.solve(solver=cp.GUROBI)  # Use GUROBI solver for efficiency
    except cp.error.SolverError:
        problem.solve(solver=cp.ECOS_BB)  # Incremental solve

    # Check if the solution is optimal
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return x.value
    else:
        raise ValueError("No optimal solution found.")

def hierarchical_clustering(similarity_matrix):
    hierarchy = {}
    leaf_clusters = {i: [i] for i in range(similarity_matrix.shape[0])}  # Initial leaf tasks

    level = 1
    while similarity_matrix.shape[0] > 1:
        print("Current similarity matrix shape:", similarity_matrix.shape)
        print("Current similarity matrix:")
        print(np.round(similarity_matrix, 2))

        n = similarity_matrix.shape[0]

        # Add dummy tasks if the number of tasks is odd
        num_dummy_tasks = n % 2
        if num_dummy_tasks > 0:
            print("Adding dummy tasks to make the number of tasks even.")

            # Create dummy similarities with large values to avoid clustering with real tasks
            large_value = 1e6  # Adjust this value as needed
            dummy_similarity = np.full((1, n), large_value)
            # Set similarity of dummy task with itself to zero
            dummy_similarity[0, -1] = 0

            similarity_matrix = np.vstack((similarity_matrix, dummy_similarity))
            dummy_column = np.full((n + 1, 1), large_value)
            # Set similarity of dummy task with itself to zero
            dummy_column[-1, 0] = 0

            similarity_matrix = np.hstack((similarity_matrix, dummy_column))

        # Solve for the current similarity matrix
        solution = minimize_pairwise_similarity(similarity_matrix)

        # Extract pairs of tasks from the solution
        n = len(solution)
        current_clusters = []
        used = set()
        for i in range(n):
            if i not in used:
                for j in range(n):
                    if solution[i, j] > 0.5 and j not in used:
                        current_clusters.append([i, j])
                        used.add(i)
                        used.add(j)
                        break

        print("Current clusters (including dummies):", current_clusters)

        # Filter out dummy tasks from clusters
        filtered_clusters = []
        for cluster in current_clusters:
            filtered_cluster = [elem for elem in cluster if elem < n - num_dummy_tasks]
            if filtered_cluster:
                filtered_clusters.append(filtered_cluster)

        print("Filtered clusters (excluding dummies):", filtered_clusters)

        # Convert current level clusters to leaf clusters
        level_leaf_clusters = []
        for cluster in filtered_clusters:
            merged_cluster = []
            for elem in cluster:
                merged_cluster.extend(leaf_clusters[elem])
            level_leaf_clusters.append(merged_cluster)

        print("Level clusters with leaf tasks:", level_leaf_clusters)

        # Save current level clusters to the hierarchy with the cluster size as key
        cluster_size = 2 ** level
        hierarchy[cluster_size] = level_leaf_clusters

        # Update leaf clusters
        leaf_clusters = {i: level_leaf_clusters[i] for i in range(len(level_leaf_clusters))}

        print("Updated leaf clusters:", leaf_clusters)

        # Build a new similarity matrix for the next iteration
        new_size = len(filtered_clusters)
        new_similarity_matrix = np.zeros((new_size, new_size))
        for i in range(new_size):
            for j in range(new_size):
                total_similarity = 0
                count = 0
                for elem_i in filtered_clusters[i]:
                    for elem_j in filtered_clusters[j]:
                        total_similarity += similarity_matrix[elem_i, elem_j]
                        count += 1
                # Average the similarities to normalize
                if count > 0:
                    new_similarity_matrix[i, j] = total_similarity / count
                else:
                    new_similarity_matrix[i, j] = 0

        similarity_matrix = new_similarity_matrix
        print("New similarity matrix shape:", similarity_matrix.shape)

        level += 1

    return leaf_clusters, hierarchy

if __name__ == "__main__":
    path_to_similarity_matrix = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/CIFAR-100_cambdridge_similarity_with_grad_prod_task_aware_resnet-18-2-50_epochs_10.json"
    with open(path_to_similarity_matrix, 'r') as f: 
        similarity_matrix = json.load(f)
    
    # Convert JSON list of lists to a NumPy array
    similarity_matrix = np.array(similarity_matrix)[:50, :50]

    # Perform hierarchical clustering
    leaf_clusters, hierarchy = hierarchical_clustering(similarity_matrix)

    print("Leaf clusters:")
    print(leaf_clusters)

    print("Hierarchy:")
    for level, clusters in hierarchy.items():
        print(f"Cluster size {level}: {clusters}")

    # Save hierarchy to a JSON file
    output_path = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/hierarchical_min_partition_CIFAR-100_cambdridge_similarity_with_grad_prod_task_aware_resnet-18-2-50_epochs_10_with_normalization.json"
    with open(output_path, "w") as f:
        json.dump(hierarchy, f, indent=4)