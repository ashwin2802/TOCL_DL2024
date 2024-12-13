import sys
import os
# Add project_root to the Python path
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

from utils.compute_metrics import *


def main(args):
    if args['model_name'] == "resnet":
        # Model getter: specify dataset and depth of the network.
        model = pytorchcv_wrapper.resnet("cifar100", depth=args['depth'], pretrained=False)
    else: 
        raise Exception(f"Unsupported model {args['model_name']}")

    # Or get a more specific model. E.g. wide resnet, with depth 40 and growth
    # factor 8 for Cifar 10.
    # model = pytorchcv_wrapper.get_model("wrn40_8_cifar10", pretrained=False)

    # --- CONFIG
    device = torch.device(
        f"cuda" if torch.cuda.is_available() else "cpu"
    )

    # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer

    # x = torch.rand((3, 32, 32), device=device)
    # output = feature_extractor(x)

    # print(f"output.shape: {output.shape}")

    # --- TRANSFORMATIONS
    transform = transforms.Compose(
        [
            ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ]
    )

    # --- BENCHMARK CREATION
    cifar_train = CIFAR100(
        # we specify the root where to store the data
        # root=expanduser("~") + "/.avalanche/data/cifar100/",
        root=args['data_dir'],
        train=True,
        download=True,
        transform=transform,
    )
    cifar_test = CIFAR100(
        # root=expanduser("~") + "/.avalanche/data/cifar100/",
        root=args['data_dir'],
        train=False,
        download=True,
        transform=transform,
    )
    benchmark = nc_benchmark(
        cifar_train,
        cifar_test,
        args['num_tasks'],
        task_labels=False,
        seed=1234,
        # fixed_class_order=[i for i in range(100)],
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    # text_logger = TextLogger(open("results/CIFAR-100.txt", "w"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=False),
        # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # CREATE THE STRATEGY INSTANCE (Naive, with Replay)
    if args['strategy'] == "naive": 
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            evaluator=eval_plugin,
        )
    elif args['strategy'] == "naive-w-replay":
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            plugins=[ReplayPlugin(mem_size=5120)],
            evaluator=eval_plugin,
        )
    elif args['strategy'] == "naive-w-ewc": 
        print(f"strategy: naive-w-ewc")
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            plugins=[EWCPlugin(ewc_lambda=10.0)],
            evaluator=eval_plugin,
        )
    elif args['strategy'] == "naive-w-gem": 
        print(f"strategy: naive-w-gem")
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            plugins=[GEMPlugin(5120, 0.5)],
            evaluator=eval_plugin,
        )
    elif args['strategy'] == "cumulative": 
        cl_strategy = Cumulative(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            # plugins=[ReplayPlugin(mem_size=1000)],
            evaluator=eval_plugin,
        )
    elif args['strategy'] == "icarl": 
        # Separate the feature extractor and the classification head
        # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
        # classification_head = model.fc if hasattr(model, 'fc') else model.output  # Get the classification head
        
        model.output = torch.nn.Identity()
        classification_head = torch.nn.Linear(in_features=64, out_features=100)
        import itertools

        # Combine parameters from feature_extractor and classification_head
        params = itertools.chain(model.parameters(), classification_head.parameters())

        print(f"model: {model}")

        print("stragety: icarl")
        cl_strategy = ICaRL(
            model,
            classification_head,
            torch.optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            5120,
            None,
            True,
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            # plugins=[ReplayPlugin(mem_size=1000)],
            evaluator=eval_plugin,
        )
    elif args['strategy'] == "naive-w-gem": 
        print("strategy: naive-w-gem")
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            plugins=[GEMPlugin(5120, 0.5)],
            evaluator=eval_plugin,
        )
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

    file_path = args['res_file_template'].format(args['model_name'], args['depth'], args['num_tasks'], args['strategy'])
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
