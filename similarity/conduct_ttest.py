import json
from scipy.stats import ttest_ind
import os

def load_data_from_json(file_path):
    """
    Load the 'average_final_accuracy' values from a JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extract the "average_final_accuracy" attribute
            accuracies = data['average_final_accuracy']
        return accuracies
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

def perform_ttest(data1, data2):
    """
    Perform a one-tailed t-test to test if the mean of data1 > mean of data2.
    """
    # Perform a two-tailed t-test first
    t_stat, p_value = ttest_ind(data1, data2, alternative='greater')
    return t_stat, p_value

if __name__ == "__main__":
    # File paths for the two JSON files
    prefix = "/cluster/home/rrigoni"
    file_path1 = os.path.join(prefix, "TOCL_DL2024/results/MNIST-10_task_aware_simpleMLP-784-256-3-2-5_classes_per_task_2_similarity_random_grouping_random_ordering_random_epochs_20.json")  # Replace with your file path
    file_path2 = os.path.join(prefix, "TOCL_DL2024/results/MNIST-10_task_aware_simpleMLP-784-256-3-2-5_classes_per_task_2_similarity_cambridge_grouping_optimal_ordering_min_cut_epochs_20.json")  # Replace with your file path

    # Load data
    data1 = load_data_from_json(file_path1)
    data2 = load_data_from_json(file_path2)

    # Check if data was successfully loaded
    if not data1 or not data2:
        print("Failed to load data. Check your JSON files.")
    else:
        # Perform the one-tailed t-test
        t_stat, p_value = perform_ttest(data2, data1)

        # Output the results
        print("One-Tailed T-Test Results:")
        print(f"T-Statistic: {t_stat}")
        print(f"P-Value: {p_value}")

        # Interpret the results
        alpha = 0.05  # Significance level
        if p_value < alpha:
            print("Reject the null hypothesis: Data1 has a greater mean than Data2.")
        else:
            print("Fail to reject the null hypothesis: Insufficient evidence to say Data1 > Data2.")
