from typing import Any
from models.resnet import resnet

class ModuleLoader:
    def __init__(self):
        """
        Initializes the ModuleLoader with logic to load specific models based on keywords.
        """
        self.supported_models = {
            "resnet": self._load_resnet,
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

    # Add more model-specific loaders here if needed
