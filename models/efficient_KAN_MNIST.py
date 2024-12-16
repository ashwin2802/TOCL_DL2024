from avalanche.models.efficient_kan import KAN
import torch

class KAN_MNIST(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 10):

        super(KAN_MNIST, self).__init__()
        input_dim = 28 * 28  # MNIST images are 28x28 pixels
        self.model = KAN([input_dim, hidden_dim, num_classes])

    def forward(self, x: torch.Tensor):
        return self.model(x)
    

# Example Usage: model = KAN_MNIST(hidden_dim = 64, num_classes = 10)
