import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

class ExperimentVisualizer:
    def __init__(self, task_id, results_folder="results/", plots_folder="plots/"):
        """
        Initialize the ExperimentVisualizer with a task ID and results folder.

        Parameters:
        task_id (str): Task ID to filter JSON files in the folder.
        results_folder (str): Path to the folder containing JSON result files.
        plots_folder (str): Path to save the generated plots.
        """
        self.task_id = task_id
        self.results_folder = Path(results_folder)
        self.plots_folder = Path(plots_folder)
        self.experiments = {}

        # Create the plots folder if it doesn't exist
        self.plots_folder.mkdir(parents=True, exist_ok=True)

        # Load the experiment data
        self.load_experiments()

    def load_experiments(self):
        """
        Load all JSON files starting with the task ID from the results folder.
        """
        for file in self.results_folder.glob(f"{self.task_id}*.json"):
            with open(file, "r") as f:
                experiment_data = json.load(f)
                self.experiments[file.stem] = experiment_data

    def save_combined_plots(self):
        """
        Save all metrics as a single combined .png file with subplots.
        """
        if not self.experiments:
            print(f"No experiments found for task ID '{self.task_id}'.")
            return

        # Find all metric keys from the first experiment
        sample_experiment = next(iter(self.experiments.values()))
        metric_keys = [key for key in sample_experiment.keys() if key != "accuracy_matrix"]

        # Prepare the plot
        num_metrics = len(metric_keys)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))
        fig.tight_layout(pad=5.0)

        # Ensure axes is iterable, even if there is only one plot
        if num_metrics == 1:
            axes = [axes]

        # Plot each metric
        for ax, metric_key in zip(axes, metric_keys):
            for experiment_name, data in self.experiments.items():
                if metric_key in data:
                    ax.plot(data[metric_key], label=experiment_name)
            ax.set_title(f"{metric_key.replace('_', ' ').title()} Comparison")
            ax.set_xlabel("Time")
            ax.set_ylabel(metric_key.replace('_', ' ').title())
            ax.legend()
            ax.grid(True)

        # Save the combined plot
        output_file = self.plots_folder / f"{self.task_id}_combined_plots.png"
        plt.savefig(output_file)
        plt.close(fig)

        print(f"Combined plots saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    # Initialize visualizer with a task ID
    task_id = "CIFAR-100"  # Replace with your actual task ID
    visualizer = ExperimentVisualizer(task_id=task_id, results_folder="results/", plots_folder="plots/")

    # Save combined plots
    visualizer.save_combined_plots()
