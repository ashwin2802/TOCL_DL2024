################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Eli Verwimp                                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example shows how to train models provided by pytorchcv with the rehearsal
strategy.
"""

from os.path import expanduser

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark
from avalanche.models import pytorchcv_wrapper
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

from compute_metrics import *


def main(args):
    # Model getter: specify dataset and depth of the network.
    model = pytorchcv_wrapper.resnet("cifar100", depth=20, pretrained=False)

    # Or get a more specific model. E.g. wide resnet, with depth 40 and growth
    # factor 8 for Cifar 10.
    # model = pytorchcv_wrapper.get_model("wrn40_8_cifar10", pretrained=False)

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

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
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    cifar_test = CIFAR100(
        # root=expanduser("~") + "/.avalanche/data/cifar100/",
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    benchmark = nc_benchmark(
        cifar_train,
        cifar_test,
        5,
        task_labels=False,
        seed=1234,
        fixed_class_order=[i for i in range(100)],
    )

    # choose some metrics and evaluation method
    # interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open("results/CIFAR-100.txt", "w"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=False),
        # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # forgetting_metrics(experience=True),
        loggers=[text_logger],
    )

    # CREATE THE STRATEGY INSTANCE (Naive, with Replay)
    cl_strategy = Naive(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=10,
        eval_mb_size=100,
        device=device,
        # plugins=[ReplayPlugin(mem_size=1000)],
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

    accuracy_matrix = compute_accuracy_matrix(results)

    average_accuracy = compute_accuracy_matrix(accuracy_matrix)
    average_incremental_accuracy = compute_average_incremental_accuracy_matrix(average_accuracy)

    forgetting_measure = compute_forgetting_matrix(accuracy_matrix)
    backward_transfer = compute_backward_transfer_matrix(accuracy_matrix)

    save_results_to_file(args.res_file, accuracy_matrix, average_accuracy, average_incremental_accuracy, forgetting_measure, backward_transfer)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--data_dir", 
        type=str,
        default="/cluster/scratch/rrigoni/dl_project/CIFAR-100", 
        help="directory where to store the datasets for the experiments"
    )
    parser.add_argument(
        "--res_path", 
        type=str,
        default="/home/rrigoni/TOCL_DL2024/results/CIFAR-100.json", 
        help="directory where to store the datasets for the experiments"
    )

    args = parser.parse_args()
    main(args)
