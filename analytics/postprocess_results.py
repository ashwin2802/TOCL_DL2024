import os
import json
from scipy.stats import zscore
import numpy as np
from collections import OrderedDict

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def compute_correlation(x, y):
    """
    Compute Pearson and Spearman correlation coefficients.

    Parameters:
        x (array-like): Predictor variable.
        y (array-like): Response variable.

    Returns:
        dict: Pearson and Spearman correlation coefficients and their p-values.
    """
    from scipy.stats import pearsonr, spearmanr

    # Ensure input arrays are non-empty
    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must not be empty.")

    # Compute Pearson correlation
    pearson_corr, pearson_p_value = pearsonr(x, y)

    # Compute Spearman correlation
    spearman_corr, spearman_p_value = spearmanr(x, y)

    return {
        "pearson_corr": pearson_corr,
        "pearson_p_value": pearson_p_value,
        "spearman_corr": spearman_corr,
        "spearman_p_value": spearman_p_value,
    }
    
def reshape_data(data):
    """
    Reshape the data from [ci_iterations * num_tasks, num_tasks, hidden_dim]
    to [ci_iterations, num_tasks, num_tasks, hidden_dim].

    Parameters:
        data (numpy.ndarray): Input array of shape [ci_iterations * num_tasks, num_tasks, hidden_dim].

    Returns:
        numpy.ndarray: Reshaped array of shape [ci_iterations, num_tasks, num_tasks, hidden_dim].
    """
    if len(data.shape) != 3:
        raise ValueError("Input data must have 3 dimensions: [ci_iterations * num_tasks, num_tasks, hidden_dim]")

    total_iterations, num_tasks, hidden_dim = data.shape

    # Compute the number of ci_iterations
    ci_iterations = total_iterations // num_tasks
    if total_iterations % num_tasks != 0:
        raise ValueError("Total iterations must be divisible by num_tasks.")

    # Reshape the data
    reshaped_data = data.reshape(ci_iterations, num_tasks, num_tasks, hidden_dim)
    return reshaped_data

def compute_variance(data):
    """
    Compute the variance of each task embedding from the average task embedding across `tasks_prime`,
    and average it over `ci_iterations`.

    Parameters:
        data (numpy.ndarray): Input array of shape [ci_iterations * num_tasks_prime, num_tasks, hidden_dim].

    Returns:
        tuple: ("embedding_norms", task_embedding_norms)
            - task_embedding_norms: A list of averaged norms for each task.
    """
    # Convert the input data to a NumPy array
    data = np.array(data)

    # Validate the shape of the data
    if len(data.shape) != 3:
        raise ValueError("Input data must have 3 dimensions: [ci_iterations * num_tasks_prime, num_tasks, hidden_dim]")

    # Reshape data to [ci_iterations, num_tasks_prime, num_tasks, hidden_dim]
    num_iterations = data.shape[0] // data.shape[1]  # ci_iterations = total_rows / num_tasks_prime
    num_tasks = data.shape[1]
    hidden_dim = data.shape[2]
    reshaped_data = data.reshape(num_iterations, num_tasks, num_tasks, hidden_dim)

    # Compute the average task embedding across `tasks_prime`
    task_embedding_norms = []
    for task_idx in range(num_tasks):
        ci_norms = []
        for ci_idx in range(num_iterations):
            task_embeddings = reshaped_data[ci_idx, :, task_idx, :]  # Shape: [num_tasks_prime, hidden_dim]
            avg_embedding = np.mean(task_embeddings, axis=0)  # Average embedding: [hidden_dim]
            norm_sum = np.sum([np.linalg.norm(embedding - avg_embedding) for embedding in task_embeddings])
            ci_norms.append(norm_sum / task_embeddings.shape[0])
        task_embedding_norms.append(np.mean(ci_norms))  # Average over ci_iterations

    return "variance", task_embedding_norms

def compute_trajectory_length(data):
    """
    Compute the trajectory length for each task in the given data.

    Parameters:
        data: A Python list of shape [ci_iterations, num_tasks, num_tasks, hidden_dim].

    Returns:
        tuple: ("trajectory_lengths", task_trajectory_lengths)
            - task_trajectory_lengths: A list of average trajectory lengths for each task.
    """
    # Convert the input list to a NumPy array
    data = np.array(data)
    data = reshape_data(data)
    print(f"data.shape: {data.shape}")

    # Validate the shape of the data
    if len(data.shape) != 4:
        raise ValueError("Input data must have 4 dimensions: [ci_iterations, num_tasks, num_tasks, hidden_dim]")

    # Extract dimensions
    ci_iterations, _, num_tasks, hidden_dim = data.shape

    # Initialize a list to store trajectory lengths for each task
    task_trajectory_lengths = []

    # Iterate over tasks
    for task_idx in range(num_tasks):
        # Collect the hidden representations for this task: [ci_iterations, num_tasks, hidden_dim]
        task_data = data[:, :, task_idx, :]  # Shape: [ci_iterations, num_tasks, hidden_dim]

        # Compute trajectory length for each `ci_iteration`
        ci_trajectory_lengths = []
        for ci_iteration in range(ci_iterations):
            # Hidden representations for this `ci_iteration`: [num_tasks, hidden_dim]
            ci_task_data = task_data[ci_iteration, :, :]

            # Compute trajectory length (sum of Euclidean distances across `num_tasks`)
            ci_length = 0
            for i in range(num_tasks - 1):
                ci_length += np.linalg.norm(ci_task_data[i + 1] - ci_task_data[i])

            ci_trajectory_lengths.append(ci_length)

        # Average trajectory length across `ci_iterations`
        average_trajectory_length = np.mean(ci_trajectory_lengths)
        task_trajectory_lengths.append(average_trajectory_length)

    return "trajectory_lengths", task_trajectory_lengths

from scipy.special import softmax

def compute_kl_divergence(data):
    """
    Compute the KL divergence between task i at task_prime j and j+1,
    averaged across ci_iterations.

    Parameters:
        data (numpy.ndarray): Input array of shape [ci_iterations * num_tasks_prime, num_tasks, hidden_dim].

    Returns:
        tuple: ("kl_divergence", task_kl_divergence)
            - task_kl_divergence: A list of average KL divergence for each task.
    """
    # Convert the input data to a NumPy array
    data = np.array(data)

    # Validate the shape of the data
    if len(data.shape) != 3:
        raise ValueError("Input data must have 3 dimensions: [ci_iterations * num_tasks_prime, num_tasks, hidden_dim]")

    # Reshape data to [ci_iterations, num_tasks_prime, num_tasks, hidden_dim]
    num_iterations = data.shape[0] // data.shape[1]  # ci_iterations = total_rows / num_tasks_prime
    num_tasks = data.shape[1]
    hidden_dim = data.shape[2]
    reshaped_data = data.reshape(num_iterations, num_tasks, num_tasks, hidden_dim)

    # Compute KL divergence
    task_kl_divergence = []
    for task_idx in range(num_tasks):
        ci_kl_divergences = []
        for ci_idx in range(num_iterations):
            task_data = reshaped_data[ci_idx, :, task_idx, :]  # Shape: [num_tasks_prime, hidden_dim]
            for j in range(len(task_data) - 1):
                # Normalize to probability distributions using softmax
                p = softmax(task_data[j])       # Shape: [hidden_dim]
                q = softmax(task_data[j + 1])   # Shape: [hidden_dim]

                # Compute KL divergence
                kl_div = np.sum(p * np.log(p / q + 1e-10))  # Add small value to avoid division by zero
                ci_kl_divergences.append(kl_div)

        # Average KL divergence over all ci_iterations
        avg_kl_div = np.mean(ci_kl_divergences)
        task_kl_divergence.append(avg_kl_div)

    return "kl_divergence", task_kl_divergence

def compute_analytics_on_hidden_representations(data):
    """
    Computes analytics on the hidden representations using a list of analytic functions.

    Parameters:
        data: The input data on which analytics will be computed.

    Returns:
        dict: A dictionary where keys are analytic names and values are the corresponding analytic values.
    """

    # Define the list of analytic functions
    analytic_functions = [
        compute_trajectory_length,  # Add more functions as needed
        compute_variance, 
        compute_kl_divergence
    ]

    # Initialize the analytics dictionary
    analytics = {}

    # Loop through each analytic function, invoke it, and collect results
    for func in analytic_functions:
        analytic_name, analytic_value = func(data)
        analytics[analytic_name] = analytic_value

    return analytics

def process_json_files(folder_path, key_order):
    """
    Iterates over all .json files in a folder and its subdirectories,
    reorders the dictionary keys based on the specified key_order,
    and overwrites the files with the updated data.

    Parameters:
        folder_path (str): Path to the folder containing .json files.
        key_order (list): List specifying the desired key order.
    """
    print(f"Starting to process files in folder: {folder_path}")
    
    total_files = 0
    processed_files = 0
    skipped_files = 0
    errors = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                total_files += 1
                file_path = os.path.join(root, file)
                print(f"\nProcessing file: {file_path}")

                try:
                    # Read the JSON file
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        print(f"Successfully read file: {file_path}")
                    
                    # Compute analytics and update the 'analytics' key
                    if 'intermediate_representations' in data:
                        print(f"Computing analytics for: {file_path}")
                        data['analytics'] = compute_analytics_on_hidden_representations(data['intermediate_representations'])
                    else:
                        print(f"Skipping analytics computation: 'intermediate_representations' not found in {file_path}")

                    # Reorder the dictionary if it's a dictionary
                    if isinstance(data, dict):
                        ordered_data = OrderedDict((key, data[key]) for key in key_order if key in data)

                        # Overwrite the file with the ordered dictionary
                        with open(file_path, "w") as f:
                            json.dump(ordered_data, f, indent=4)

                        processed_files += 1
                        print(f"Successfully updated file: {file_path}")
                    else:
                        skipped_files += 1
                        print(f"Skipping file (not a dictionary): {file_path}")
                except Exception as e:
                    errors += 1
                    print(f"Error processing file {file_path}: {e}")

    # Summary of the operation
    print("\nProcessing complete.")
    print(f"Total files found: {total_files}")
    print(f"Files processed successfully: {processed_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Errors encountered: {errors}")

def analyze_forgetting(parameter, folder_path, result_path, z_threshold=2.5):
    """
    Iterates over .json files, retrieves 'forgetting_measure_mean' and specified parameter values,
    removes outliers, fits a linear regression, calculates correlation coefficients, and saves the plot.

    Parameters:
        parameter (str): The parameter to analyze (e.g., "trajectory_lengths").
        folder_path (str): Path to the folder containing .json files.
        result_path (str): Path to save the linear regression plot.
        z_threshold (float): Threshold for z-scores to identify outliers.
    """
    forgetting_last_values = []
    parameter_values = []

    # Iterate through files to extract forgetting measures and parameter values
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Retrieve 'forgetting_measure_mean' and the last element
                    if "forgetting_measure_mean" in data and "analytics" in data:
                        forgetting_values = data["forgetting_measure_mean"]
                        if forgetting_values:
                            forgetting_last_values.append(forgetting_values[-1])

                        # Retrieve the specified parameter from analytics
                        parameter_value = data["analytics"].get(parameter, [])
                        if parameter_value:
                            parameter_values.append(np.mean(parameter_value))

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    # if forgetting_last_values and parameter_values:
    #     # Convert to numpy arrays
    #     X = np.array(forgetting_last_values).reshape(-1, 1)
    #     y = np.array(parameter_values)

    #     # Combine data for outlier detection
    #     data = np.hstack([X, y.reshape(-1, 1)])
    #     z_scores = zscore(data, axis=0)
    #     non_outliers = (np.abs(z_scores) < z_threshold).all(axis=1)

    #     # Filter outliers
    #     X_clean = X[non_outliers]
    #     y_clean = y[non_outliers]

    #     # Linear regression on cleaned data
    #     model = LinearRegression()
    #     model.fit(X_clean, y_clean)

    #     # Compute correlation coefficients for cleaned data
    #     correlation_coef = compute_correlation(X_clean.flatten(), y_clean)

    #     # Plot results
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(X_clean, y_clean, label="Cleaned Data", alpha=0.7)
    #     plt.scatter(X[~non_outliers], y[~non_outliers], label="Outliers", color="red", alpha=0.7)
    #     plt.plot(X_clean, model.predict(X_clean), color="blue", label="Linear Fit")
    #     plt.xlabel("Forgetting Measure (Last Value)")
    #     plt.ylabel(f"{parameter} (Averaged)")
    #     plt.title(f"Linear Regression: Forgetting vs. {parameter}")
    #     plt.legend()
    #     plt.grid(True)

    #     # Save the plot
    #     os.makedirs(result_path, exist_ok=True)
    #     plot_path = os.path.join(result_path, f"linear_regression_forgetting_vs_{parameter}_plot.png")
    #     plt.savefig(plot_path)
    #     plt.close()

    #     print(f"Linear regression plot saved to: {plot_path}")
    #     print(f"Linear regression coefficients: {model.coef_}")
    #     print(f"Intercept: {model.intercept_}")
    #     print(f"Correlation Coefficients: {correlation_coef}")

    # else:
    #     print("Insufficient data to perform regression analysis.")

# Example usage:
if __name__ == "__main__":
    folder_path = "./results/"
    key_order = [
        'average_accuracy_mean',
        'average_accuracy_std',
        'average_incremental_accuracy_mean',
        'average_incremental_accuracy_std',
        'forgetting_measure_mean',
        'forgetting_measure_std',
        'backward_transfer_mean',
        'backward_transfer_std',
        'average_final_accuracy',
        'analytics',
        'intermediate_representations'
    ]
    process_json_files(folder_path, key_order)


# import torch
# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from itertools import cycle
# import os

# def plot_averaged_representations(averaged_representations, output_dir):
#     """
#     Plots the PCA of concatenated intermediate representations in 2D with detailed debug information.

#     Parameters:
#         averaged_representations (list): List of `num_tasks` many `[num_tasks, hidden_dim]` tensors.
#         output_dir (str): Directory to save the PCA plot.
#     """
#     print(f"Received {len(averaged_representations)} tensors in averaged_representations.", flush=True)

#     # Step 1: Concatenate tensors along the first dimension
#     concatenated_representations = torch.cat(averaged_representations, dim=0)  # Shape: [num_tasks * len(averaged_representations), hidden_dim]
#     print(f"Concatenated tensor shape: {concatenated_representations.shape}", flush=True)

#     # Step 2: Convert to numpy for PCA
#     all_representations = concatenated_representations.numpy()  # Convert PyTorch tensor to NumPy array
#     print(f"Converted to numpy. Shape: {all_representations.shape}", flush=True)

#     # Step 3: Apply PCA
#     print("Applying PCA...", flush=True)
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(all_representations)  # Shape: [num_tasks * len(averaged_representations), 2]
#     print(f"PCA completed. Result shape: {pca_result.shape}", flush=True)

#     # Prepare symbols and colors for plotting
#     num_tasks = averaged_representations[0].shape[0]  # Extract number of tasks from the first tensor
#     print(f"Number of tasks: {num_tasks}", flush=True)

#     symbols = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']  # Define a list of symbols
#     symbol_cycle = cycle(symbols[:num_tasks])  # Cycle through symbols for tasks
#     colors = plt.cm.viridis(np.linspace(0, 1, len(averaged_representations)))  # Progressive colormap
#     print(f"Prepared {len(symbols[:num_tasks])} symbols and {len(colors)} colors.", flush=True)

#     # Plot PCA results
#     print("Starting PCA plotting...", flush=True)
#     plt.figure(figsize=(10, 8))

#     # Total number of tensors (runs) used in averaged_representations
#     num_runs = len(averaged_representations)

#     # Flattened number of points per run (num_tasks * num_tasks)
#     num_points_per_run = num_tasks

#     for i, task_symbol in enumerate(symbols[:num_tasks]):  # Assign each task a unique symbol
#         print(f"Processing task {i + 1} with symbol '{task_symbol}'.", flush=True)
#         for run_idx, color in enumerate(colors):  # Iterate over runs for progressive coloring
#             # Compute indices for the current task in the concatenated PCA result
#             task_indices = [j * num_points_per_run + i for j in range(num_runs)]
#             print(f"Task {i + 1}, Run {run_idx + 1}: Indices -> {task_indices}", flush=True)

#             # Extract points for the current task and run
#             task_points = pca_result[task_indices]
#             print(f"Task {i + 1}, Run {run_idx + 1}: Points shape -> {task_points.shape}", flush=True)

#             # Scatter plot for the current task and run
#             plt.scatter(
#                 task_points[:, 0],
#                 task_points[:, 1],
#                 marker=task_symbol,
#                 color=color,
#                 alpha=0.7,
#                 label=f"Task {i + 1}, Run {run_idx + 1}" if run_idx == 0 else None  # Label only once per task
#             )

#     # Customize the plot
#     print("Finalizing the plot...", flush=True)
#     plt.title("PCA of Concatenated Intermediate Representations")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.grid(True)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot

#     # Save the plot
#     plot_path = os.path.join(output_dir, "pca_averaged_intermediate_representations.png")
#     plt.savefig(plot_path, bbox_inches="tight")
#     print(f"PCA plot saved to {plot_path}", flush=True)
#     plt.close()
