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

    original_classes_in_exp = [{2 * i, 2 * i + 1} for i in range(5)]
    # original_classes_in_exp = [{0, 1, 2, 3}, {4, 5, 6, 7, 8, 9}]
    benchmark = SplitMNIST(n_experiences=len(original_classes_in_exp), shuffle=False, seed=1, return_task_id=False, original_classes_in_exp=original_classes_in_exp)

    # Prepare for training & testing
    # Change optimizer too
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    text_logger = TextLogger(open("results/MNIST-base.txt", "a"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=False,
            epoch_running=False,
            experience=True,
            stream=False,
        ),
        # class_accuracy_metrics(
        #     experience=False, stream=True, classes=list(range(benchmark.n_classes))
        # ),
        loggers=[text_logger],
        collect_all=True,
    )

    # Continual learning strategy with default logger
    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=32,
        evaluator=eval_plugin,
        device=device,
        eval_every=1,
    )

    print("Starting experiment...")
    results = []
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary containing last recorded value
        # for each metric.
        res = cl_strategy.train(experience, eval_streams=[benchmark.test_stream])
        results.append(res)
        print("Training completed")
    
    # Number of tasks
    num_tasks = len(original_classes_in_exp)

    accuracy_matrix = compute_accuracy_matrix(results)

    average_accuracy = compute_accuracy_matrix(accuracy_matrix)
    average_incremental_accuracy = compute_average_incremental_accuracy_matrix(average_accuracy)

    forgetting_measure = compute_forgetting_matrix(accuracy_matrix)
    backward_transfer = compute_backward_transfer_matrix(accuracy_matrix)

    # do something with these matrices, maybe serialize them or store in wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist_type",
        type=str,
        default="split",
        choices=["rotated", "permuted", "split"],
        help="Choose between MNIST variations: " "rotated, permuted or split.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()

    main(args)