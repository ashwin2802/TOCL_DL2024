import cvxpy as cp
import numpy as np
import math
import json
from tqdm import tqdm
import time

def maximize_pairwise_similarity(similarity_matrix, k):
    n = len(similarity_matrix)  # Number of elements

    # Decision variables: x[i][j] is binary, indicating whether task i is in subset j
    # x = cp.Variable((n, n), boolean=True)
    x = cp.Variable((n, n), integer=True)

    # Objective function: Maximize the sum of pairwise similarities within subsets
    objective = cp.Maximize(cp.sum(cp.multiply(similarity_matrix, x)))

    # Constraints
    constraints = []

    for i in range(n):
        constraints.append(cp.sum(x[i, :]) >= k - 1)  # Each element belongs to exactly one subset
        constraints.append(cp.sum(x[i, :]) <= k - 1)  # Each element belongs to exactly one subset
        constraints.append(x[i, i] == 0)

    # # Constraint 4: If x[i,j] = 1 and x[j,k] = 1, then x[i,k] = 1 (transitivity constraint for triangular matrix)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                constraints.append(x[i, j] + x[j, k] - x[i, k] <= 1)

    # # Constraint 4: If x[i,j] = 1 and x[j,k] = 1, then x[i,k] = 1 (transitivity constraint for longer chains)
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         for k in range(j + 1, n):
    #             for l in range(k + 1, n):
    #                 constraints.append(x[i, j] + x[j, k] + x[k, l] - x[i, l] <= 2)

    # # Constraint 4: If x[i,j] = 1 and x[j,k] = 1, then x[i,k] = 1 (transitivity constraint for longer chains)
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         for k in range(j + 1, n):
    #             for l in range(k + 1, n):
    #                 for p in range(l + 1, n):
    #                     constraints.append(x[i, j] + x[j, k] + x[k, l] + x[l, p] - x[i, p] <= 3)

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

# def extract_clusters(solution_matrix, num_dummy_tasks, threshold=0.5):
#     n = solution_matrix.shape[0]  # n: number of tasks, l: number of subsets
#     clusters = []
#     for i in range(n): 
#         cluster = sorted([j for j in range(n) if solution_matrix[i,j] > threshold] + [i])

#         clusters.append(cluster)
#     return clusters
#     # visited = set()
#     # for j in range(l):
#     #     cluster = [i for i in range(n) if solution_matrix[i, j] > threshold and i not in visited]
#     #     if cluster:  # Only add non-empty clusters
#     #         clusters.append(cluster)
#     #         visited.update(cluster)

#     # return clusters
#     # # Remove dummy tasks from clusters
#     # filtered_clusters = [[i for i in cluster if i < n - num_dummy_tasks] for cluster in clusters]
#     # return [cluster for cluster in filtered_clusters if cluster]  # Remove empty clusters

def extract_clusters(solution_matrix, num_dummy_tasks, threshold=0.5):
    n = solution_matrix.shape[0]  # n: number of tasks
    clusters = []

    # Iterate through rows to extract clusters
    for i in range(n): 
        cluster = sorted([j for j in range(n) if solution_matrix[i, j] > threshold] + [i])
        if cluster not in clusters:  # Avoid duplicates
            clusters.append(cluster)

    return clusters

    # Remove dummy tasks from clusters
    filtered_clusters = [[i for i in cluster if i < n - num_dummy_tasks] for cluster in clusters]
    return [cluster for cluster in filtered_clusters if cluster]  # Remove empty clusters

# Example Usage
if __name__ == "__main__":
    path_to_similarity_matrix = "/cluster/home/rrigoni/TOCL_DL2024/similarity_matrices/MNIST-10_cambdridge_similarity_task_aware_resnet_mnist-18-2-10_epochs_10_old.json"
    with open(path_to_similarity_matrix, 'r') as f: 
        similarity_matrix = json.load(f)
    
    # Convert JSON list of lists to a NumPy array
    similarity_matrix = np.array(similarity_matrix)

    # similarity_matrix = np.array([
    #     [10, 10, 10, 0, 0, 0, 0, 0, 0, 0],
    #     [10, 10, 10, 0, 0, 0, 0, 0, 0, 0],
    #     [10, 10, 10, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 10, 10, 10, 0, 0, 0, 0],
    #     [0, 0, 0, 10, 10, 10, 0, 0, 0, 0],
    #     [0, 0, 0, 10, 10, 10, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 10, 10, 10, 0],
    #     [0, 0, 0, 0, 0, 0, 10, 10, 10, 0],
    #     [0, 0, 0, 0, 0, 0, 10, 10, 10, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]  # Cluster of size 2
    # ])

    k = 2  # Subset size
    
    num_tasks = similarity_matrix.shape[0]
    num_dummy_tasks = (k - (num_tasks % k)) % k  # Number of dummy tasks to add
    

    if num_dummy_tasks > 0:
        # Create a dummy similarity matrix
        dummy_similarity = np.zeros((num_dummy_tasks, similarity_matrix.shape[1]))
        similarity_matrix = np.vstack((similarity_matrix, dummy_similarity))
        
        # Add zero columns for the dummy tasks
        dummy_columns = np.zeros((similarity_matrix.shape[0], num_dummy_tasks))
        similarity_matrix = np.hstack((similarity_matrix, dummy_columns))

    # print(f"new similarity_matrix.shape: {similarity_matrix.shape}")
    # print(f"new similarity_matrix:\n{np.round(similarity_matrix, 1)}")
    # Print the new similarity matrix
    # print(f"new similarity_matrix.shape: {similarity_matrix.shape}")
    
    # Solve the partitioning problem
    solution = maximize_pairwise_similarity(similarity_matrix, k)
    print("new similarity_matrix:")
    print(np.array2string(solution, formatter={'float_kind': lambda x: f"{x:.1f}"}))

    clusters = extract_clusters(solution, num_dummy_tasks)
    print("Optimal clusters:")
    print(clusters)
