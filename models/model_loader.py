from typing import Any, List
from models.resnet import resnet
from models.resnet_mnist import resnet_mnist
from models.task_aware_resnet import task_aware_resnet
from models.task_aware_resnet_mnist import task_aware_resnet_mnist
from models.vgg import vgg
import torch.nn as nn

class ModuleLoader:
    def __init__(self):
        """
        Initializes the ModuleLoader with logic to load specific models based on keywords.
        """
        self.supported_models = {
            "resnet": self._load_resnet,
            "task_aware_resnet": self._load_task_aware_resnet,
            "task_aware_resnet_mnist": self._load_task_aware_resnet_mnist,
            "vgg": self._load_vgg,
            "simpleMLP": self._load_simpleMLP,
            "task_aware_simpleMLP": self._load_task_aware_simpleMLP
            # Add additional models here if needed
        }

    def load_model(self, keyword: str) -> Any:
        """
        Loads a model based on the given keyword.

        Parameters:
        keyword (str): The keyword specifying the model (e.g., "resnet-18-100").

        Returns:
        Any: The instantiated model.

        Raises:
        ValueError: If the keyword format is invalid or the model is not supported.
        """
        try:
            # Extract model type from the keyword
            parts = keyword.split("-")
            model_type = parts[0]

            # Check if the model type is supported
            if model_type not in self.supported_models:
                raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {list(self.supported_models.keys())}")

            # Delegate to the corresponding loader function
            return self.supported_models[model_type](parts)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid keyword format '{keyword}'.") from e

    def _load_task_aware_resnet(self, parts: list) -> Any:
        """
        Loads a Task-Aware ResNet model based on its ID.

        Parameters:
            parts (list): The parts of the model keyword (e.g., ["resnet", "18", "100", "5"], where:
                - "18" is the depth of the ResNet
                - "100" is the number of classes per task
                - "5" is the number of tasks

        Returns:
            nn.Module: The instantiated Task-Aware ResNet model.

        Raises:
            ValueError: If the ID format is invalid or depth is not supported.
        """
        # Ensure the parts contain sufficient information
        if len(parts) != 4:
            raise ValueError(f"Invalid ResNet ID format. Expected 'resnet-<depth>-<classes_per_task>-<num_tasks>', got: {'-'.join(parts)}")

        # Parse parameters from the parts
        try:
            depth = int(parts[1])
            num_classes_per_task = int(parts[2])
            num_tasks = int(parts[3])
        except ValueError:
            raise ValueError(f"Invalid ResNet ID format. Parameters must be integers: 'resnet-<depth>-<classes_per_task>-<num_tasks>'.")

        # Validate the depth
        supported_depths = [18, 34, 50, 101, 152]
        if depth not in supported_depths:
            raise ValueError(f"Unsupported ResNet depth '{depth}'. Supported depths: {supported_depths}")

        # Load the task-aware ResNet
        return task_aware_resnet(depth=depth, num_tasks=num_tasks, num_classes_per_task=num_classes_per_task)

    def _load_task_aware_resnet_mnist(self, parts: list) -> Any:
        """
        Loads a Task-Aware ResNet model adapted for MNIST based on its ID.

        Parameters:
            parts (list): The parts of the model keyword (e.g., ["resnet", "18", "10", "1"], where:
                - "18" is the depth of the ResNet
                - "10" is the number of classes per task (default for MNIST is 10)
                - "1" is the number of tasks (default for MNIST is a single task)

        Returns:
            nn.Module: The instantiated Task-Aware ResNet model.

        Raises:
            ValueError: If the ID format is invalid or depth is not supported.
        """
        # Ensure the parts contain sufficient information
        if len(parts) != 4:
            raise ValueError(f"Invalid ResNet ID format. Expected 'resnet-<depth>-<classes_per_task>-<num_tasks>', got: {'-'.join(parts)}")

        # Parse parameters from the parts
        try:
            depth = int(parts[1])  # Depth of the ResNet
            num_classes_per_task = int(parts[2])  # Number of classes per task (default is 10 for MNIST)
            num_tasks = int(parts[3])  # Number of tasks (default is 1 for MNIST)
        except ValueError:
            raise ValueError(f"Invalid ResNet ID format. Parameters must be integers: 'resnet-<depth>-<classes_per_task>-<num_tasks>'.")

        # Validate the depth
        supported_depths = [18, 34, 50, 101, 152]  # Supported ResNet depths
        if depth not in supported_depths:
            raise ValueError(f"Unsupported ResNet depth '{depth}'. Supported depths: {supported_depths}")

        # Load the task-aware ResNet for MNIST
        return task_aware_resnet_mnist(depth=depth, num_tasks=num_tasks, num_classes_per_task=num_classes_per_task)

    def _load_resnet(self, parts: list) -> Any:
        """
        Loads a ResNet model based on its ID.

        Parameters:
        parts (list): The parts of the model keyword (e.g., ["resnet", "18", "100"]).

        Returns:
        nn.Module: The instantiated ResNet model.

        Raises:
        ValueError: If the depth is not supported or the ID is invalid.
        """
        # from resnet import resnet  # Assuming the earlier ResNet implementation is in resnet.py

        # Parse ResNet depth and number of classes
        if len(parts) != 3:
            raise ValueError(f"Invalid ResNet ID format. Expected 'resnet-<depth>-<classes>', got: {'-'.join(parts)}")

        depth = int(parts[1])
        num_classes = int(parts[2])

        # Check if the depth is supported
        supported_depths = [18, 34, 50, 101, 152]
        if depth not in supported_depths:
            raise ValueError(f"Unsupported ResNet depth '{depth}'. Supported depths: {supported_depths}")

        return resnet(depth=depth, num_classes=num_classes)

    def _load_vgg(self, parts: list) -> Any:
        """
        Loads a VGG model based on its ID.

        Parameters:
        parts (list): The parts of the model keyword (e.g., ["vgg", "16", "100"]).

        Returns:
        nn.Module: The instantiated VGG model.

        Raises:
        ValueError: If the depth is not supported or the ID is invalid.
        """
        # Parse VGG depth, number of classes, and batch normalization flag
        if len(parts) < 3 or len(parts) > 4:
            raise ValueError(f"Invalid VGG ID format. Expected 'vgg-<depth>-<classes>[-bn]', got: {'-'.join(parts)}")

        depth = int(parts[1])
        num_classes = int(parts[2])
        batch_norm = len(parts) == 4 and parts[3] == "bn"

        # Check if the depth is supported
        supported_depths = [11, 13, 16, 19]
        if depth not in supported_depths:
            raise ValueError(f"Unsupported VGG depth '{depth}'. Supported depths: {supported_depths}")

        # Map depth to VGG configurations
        vgg_name = f"VGG{depth}"
        return vgg(name=vgg_name, num_classes=num_classes, batch_norm=batch_norm)


    def _load_simpleMLP(self, parts: list) -> Any:
        """
        Loads a simple MLP model for classification with a configurable number of hidden layers.

        Parameters:
            parts (list): The parts of the model keyword (e.g., ["mlp", "3072", "256", "3", "100"], where:
                - 3072 is the input dimension
                - 256 is the hidden dimension
                - 3 is the number of hidden layers
                - 100 is the number of classes

        Returns:
            nn.Module: The instantiated MLP model.

        Raises:
            ValueError: If the ID format is invalid or missing required parameters.
        """
        if len(parts) != 5 or not all(p.isdigit() for p in parts[1:]):
            raise ValueError(f"Invalid MLP ID format. Expected 'mlp-<input_dim>-<hidden_dim>-<num_hidden_layers>-<num_classes>', got: {'-'.join(parts)}")

        input_dim = int(parts[1])
        hidden_dim = int(parts[2])
        num_hidden_layers = int(parts[3])
        num_classes = int(parts[4])

        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_classes):
                super(SimpleMLP, self).__init__()
                self.flatten = nn.Flatten()

                # Dynamically create hidden layers
                layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                for _ in range(num_hidden_layers - 1):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

                self.hidden_layers = nn.Sequential(*layers)
                self.output_layer = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = self.flatten(x)
                x = self.hidden_layers(x)
                x = self.output_layer(x)
                return x

        return SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes)

    def _load_task_aware_simpleMLP(self, parts: List[str]) -> Any:
        """
        Loads a task-aware MLP model with a multi-headed classification head, 
        where the `forward` method uses a task_label to select the appropriate head.

        Parameters:
            parts (list): The parts of the model keyword (e.g., ["task_mlp", "3072", "256", "3", "100,3"], where:
                - "3072" is the input dimension
                - "256" is the hidden dimension
                - "3" is the number of hidden layers
                - "100" is the number of classes per task (output dimension)
                - "3" is the number of tasks (number of heads)

        Returns:
            nn.Module: The instantiated task-aware MLP model.

        Raises:
            ValueError: If the ID format is invalid or missing required parameters.
        """
        if len(parts) != 6 or not all(p.isdigit() for p in parts[1:]):
            raise ValueError(
                f"Invalid task-aware MLP ID format. Expected 'task_mlp-<input_dim>-<hidden_dim>-<num_hidden_layers>-<classes_per_task>-<num_tasks>', got: {'-'.join(parts)}"
            )

        input_dim = int(parts[1])
        hidden_dim = int(parts[2])
        num_hidden_layers = int(parts[3])
        classes_per_task = int(parts[4])
        num_tasks = int(parts[5])

        class TaskAwareMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_hidden_layers, classes_per_task, num_tasks):
                super(TaskAwareMLP, self).__init__()
                self.flatten = nn.Flatten()

                # Dynamically create hidden layers
                layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                for _ in range(num_hidden_layers - 1):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

                self.hidden_layers = nn.Sequential(*layers)

                # Create classification heads, one for each task
                self.task_heads = nn.ModuleList([nn.Linear(hidden_dim, classes_per_task) for _ in range(num_tasks)])

            def forward(self, x, task_label: int):
                """
                Forward pass for the model. Selects the appropriate head based on the task_label.

                Parameters:
                    x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
                    task_label (int): The label of the task (0-indexed) to select the corresponding head.

                Returns:
                    torch.Tensor: The output of the selected classification head.
                """
                if task_label < 0 or task_label >= len(self.task_heads):
                    raise ValueError(f"Invalid task_label {task_label}. Must be between 0 and {len(self.task_heads) - 1}.")

                x = self.flatten(x)
                x = self.hidden_layers(x)
                x = self.task_heads[task_label](x)  # Use the head corresponding to the task_label
                return x

        return TaskAwareMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            classes_per_task=classes_per_task,
            num_tasks=num_tasks
        )

