import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP # change with better architecture
from avalanche.training.supervised import Naive

from compute_metrics import *

from avalanche.logging import (
    TextLogger,
)

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    class_accuracy_metrics # unused but could be helpful for more detailed information within each task
)

from avalanche.training.plugins import EvaluationPlugin

def main(args):
    # Device config
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    # model
    model = SimpleMLP(num_classes=10)

    

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark
from avalanche.models import pytorchcv_wrapper
from avalanche.training.supervised import Naive, Cumulative, ICaRL
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, GEMPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

from models.model_loader import ModuleLoader

from utils.compute_metrics import *


def main(args):
    loader = ModuleLoader()

    print(f"args: {args}")
    model = loader.load_model(args['model_name'])
    if "resnet" in args['model_name']:
        # Modify the first convolution layer to accept single-channel input
        model.conv1 = nn.Conv2d(
            in_channels=1,  # For MNIST
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Modify the fully connected layer to output 10 classes (MNIST has 10 classes)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
    
    # --- CONFIG
    device = torch.device(
        f"cuda" if torch.cuda.is_available() else "cpu"
    )

    # original_classes_in_exp = [{2 * i, 2 * i + 1} for i in range(5)]
    # original_classes_in_exp = [{0, 1, 2, 3}, {4, 5, 6, 7, 8, 9}]
    # benchmark = SplitMNIST(n_experiences=len(original_classes_in_exp), shuffle=True, seed=1, return_task_id=False, original_classes_in_exp=original_classes_in_exp)
    benchmark = SplitMNIST(n_experiences=args['num_tasks'], shuffle=True, seed=1, return_task_id=False)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=False),
        loggers=[interactive_logger],
    )

    from avalanche.training.plugins import LRSchedulerPlugin
    from torch.optim.lr_scheduler import LambdaLR

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)

    # Define a linear learning rate scheduler
    # Lambda function: Linear decay from initial LR to 0 over training epochs
    lambda_lr = lambda epoch: max(0.0, 1.0 - epoch / args['train_epochs'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    # Create the LR scheduler plugin for Avalanche
    lr_scheduler_plugin = LRSchedulerPlugin(scheduler, step_granularity="epoch")

    # CREATE THE STRATEGY INSTANCE (Naive, with Replay)
    if args['strategy'] == "naive": 
        cl_strategy = Naive(
            model,
            # torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            # try increasing the learning rate
            # torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            optimizer,
            CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            evaluator=eval_plugin,
            # plugins=[lr_scheduler_plugin]
        )
        
    # elif args['strategy'] == "naive-w-replay":
    #     cl_strategy = Naive(
    #         model,
    #         torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
    #         CrossEntropyLoss(),
    #         train_mb_size=args['train_mb_size'],
    #         train_epochs=args['train_epochs'],
    #         eval_mb_size=args['test_mb_size'],
    #         device=device,
    #         plugins=[ReplayPlugin(mem_size=5120)],
    #         evaluator=eval_plugin,
    #     )
    # elif args['strategy'] == "naive-w-ewc": 
    #     print(f"strategy: naive-w-ewc")
    #     cl_strategy = Naive(
    #         model,
    #         torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
    #         CrossEntropyLoss(),
    #         train_mb_size=args['train_mb_size'],
    #         train_epochs=args['train_epochs'],
    #         eval_mb_size=args['test_mb_size'],
    #         device=device,
    #         plugins=[EWCPlugin(ewc_lambda=10.0)],
    #         evaluator=eval_plugin,
    #     )
    # elif args['strategy'] == "naive-w-gem": 
    #     print(f"strategy: naive-w-gem")
    #     cl_strategy = Naive(
    #         model,
    #         torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
    #         CrossEntropyLoss(),
    #         train_mb_size=args['train_mb_size'],
    #         train_epochs=args['train_epochs'],
    #         eval_mb_size=args['test_mb_size'],
    #         device=device,
    #         plugins=[GEMPlugin(5120, 0.5)],
    #         evaluator=eval_plugin,
    #     )
    # elif args['strategy'] == "cumulative": 
    #     cl_strategy = Cumulative(
    #         model,
    #         torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
    #         CrossEntropyLoss(),
    #         train_mb_size=args['train_mb_size'],
    #         train_epochs=args['train_epochs'],
    #         eval_mb_size=args['test_mb_size'],
    #         device=device,
    #         # plugins=[ReplayPlugin(mem_size=1000)],
    #         evaluator=eval_plugin,
    #     )
    # elif args['strategy'] == "icarl": 
    #     # Separate the feature extractor and the classification head
    #     # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
    #     # classification_head = model.fc if hasattr(model, 'fc') else model.output  # Get the classification head
    #     import copy
    #     import itertools

    #     # Assuming model is a ResNet or similar model with a classification head
    #     classification_head = copy.deepcopy(model.fc)  # Make a deep copy of the classification head
    #     model.fc = torch.nn.Identity()
    #     # classification_head = torch.nn.Linear(in_features=64, out_features=100)

    #     # Combine parameters from feature_extractor and classification_head
    #     params = itertools.chain(model.parameters(), classification_head.parameters())

    #     print("stragety: icarl")
    #     cl_strategy = ICaRL(
    #         model,
    #         classification_head,
    #         torch.optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
    #         5120,
    #         None,
    #         True,
    #         train_mb_size=args['train_mb_size'],
    #         train_epochs=args['train_epochs'],
    #         eval_mb_size=args['test_mb_size'],
    #         device=device,
    #         # plugins=[ReplayPlugin(mem_size=1000)],
    #         evaluator=eval_plugin,
    #     )
    # elif args['strategy'] == "naive-w-gem": 
    #     print("strategy: naive-w-gem")
    #     cl_strategy = Naive(
    #         model,
    #         torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
    #         CrossEntropyLoss(),
    #         train_mb_size=args['train_mb_size'],
    #         train_epochs=args['train_epochs'],
    #         eval_mb_size=args['test_mb_size'],
    #         device=device,
    #         plugins=[GEMPlugin(5120, 0.5)],
    #         evaluator=eval_plugin,
    #     )
    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))

    accuracy_matrix = compute_accuracy_matrix(results, args['num_tasks'])

    average_accuracy = compute_average_accuracy_matrix(accuracy_matrix)
    average_incremental_accuracy = compute_average_incremental_accuracy_matrix(average_accuracy)

    forgetting_measure = compute_forgetting_matrix(accuracy_matrix)
    backward_transfer = compute_backward_transfer_matrix(accuracy_matrix)

    file_path = args['res_file_template'].format(args['model_name'], args['train_epochs'], args['num_tasks'], args['strategy'])
    save_results_to_file(file_path, accuracy_matrix, average_accuracy, average_incremental_accuracy, forgetting_measure, backward_transfer)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help=""
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f: 
        config = json.load(f)

    main(config)
