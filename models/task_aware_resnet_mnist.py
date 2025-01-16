import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out)

        return out


class TaskAwareResNetMNIST(nn.Module):
    def __init__(self, block, layers, num_tasks=1, num_classes_per_task=10):
        super(TaskAwareResNetMNIST, self).__init__()
        self.in_channels = 64
        self.num_tasks = num_tasks

        # Adjust the input to single channel for MNIST
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Separate classification heads for each task
        self.task_heads = nn.ModuleList([
            nn.Linear(512 * block.expansion, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, task_label):
        """
        Forward pass through the ResNet with task awareness.

        Parameters:
            x (Tensor): Input tensor.
            task_label (int): The task label indicating which classification head to use.

        Returns:
            Tensor: The output logits for the specified task.
        """
        if task_label >= self.num_tasks:
            raise ValueError(f"Invalid task_label {task_label}. Must be in range [0, {self.num_tasks - 1}].")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Use the appropriate classification head based on the task label
        x = self.task_heads[task_label](x)

        return x

    def forward_feature_extraction(self, x):
        """
        Forward pass through the ResNet with task awareness.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The output logits for the specified task.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def task_aware_resnet_mnist(depth, num_tasks=1, num_classes_per_task=10):
    """
    Creates a task-aware ResNet model for MNIST with separate classification heads for each task.

    Parameters:
    depth (int): The depth of the ResNet (18, 34, 50, 101, 152).
    num_tasks (int): Number of tasks (i.e., separate classification heads).
    num_classes_per_task (int): Number of classes for each task.

    Returns:
    nn.Module: The task-aware ResNet model for MNIST.
    """
    depths_to_layers = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (BottleneckBlock, [3, 4, 6, 3]),
        101: (BottleneckBlock, [3, 4, 23, 3]),
        152: (BottleneckBlock, [3, 8, 36, 3])
    }

    if depth not in depths_to_layers:
        raise ValueError(f"Unsupported depth {depth}. Choose from {list(depths_to_layers.keys())}.")

    block, layers = depths_to_layers[depth]
    return TaskAwareResNetMNIST(block, layers, num_tasks=num_tasks, num_classes_per_task=num_classes_per_task)


if __name__ == "__main__":
    # Example usage of the Task-Aware ResNet for MNIST
    model = task_aware_resnet_mnist(depth=18, num_tasks=2, num_classes_per_task=10)
    print(model)

    # Example input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    x = torch.randn(1, 1, 28, 28)
    output = model(x, task_label=0)
    print(output.shape)  # Should output torch.Size([1, 10])
