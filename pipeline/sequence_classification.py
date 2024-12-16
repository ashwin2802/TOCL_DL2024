import sys
import os

# Add project_root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import json

from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import (
    accuracy_metrics, 
    loss_metrics
)

from utils.compute_metrics import *
from utils.sequence_classification import HGNaive, CustomDataCollatorSeq2SeqBeta
from avalanche.training.supervised import Naive, Cumulative, ICaRL
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, GEMPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)

from utils.amazon_review import AmazonReviewDataset

def main(args): 
    device = torch.device(
        f"cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])

    if args['pretrained']: 
        model = BertForSequenceClassification.from_pretrained(args['model_name'], num_labels=2).to(device)
    else: 
        config = BertConfig.from_pretrained(args['model_name'], num_labels=2)
        model = BertForSequenceClassification(config).to(device)
        
    data_collator = CustomDataCollatorSeq2SeqBeta(tokenizer=tokenizer, model=model)

    amazon_reviews = AmazonReviewDataset(num_samples_per_domain=args['num_samples_per_domain'], domain_groups=args['domain_groups'], tokenizer=tokenizer, model=model, cache_dir=args['cache_dir'])
    benchmark, num_tasks = amazon_reviews.create_benchmark()

    # Evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=False),
        loss_metrics(minibatch=False),
        loggers=[InteractiveLogger()],
    )

    # CREATE THE STRATEGY INSTANCE (Naive, with Replay)
    if args['strategy'] == "naive": 
        # Strategy for incremental learning
        strategy = HGNaive(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4),
            torch.nn.CrossEntropyLoss(),
            train_mb_size=args['train_mb_size'],
            train_epochs=args['train_epochs'],
            eval_mb_size=args['test_mb_size'],
            device=device,
            # plugins=[GEMPlugin(5120, 0.5)],
            evaluator=eval_plugin,
        )
    elif args['strategy'] == "naive-w-replay":
        cl_strategy = HGNaive(
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
        cl_strategy = HGNaive(
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
        cl_strategy = HGNaive(
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
    elif args['strategy'] == "icarl": 
        # Separate the feature extractor and the classification head
        # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
        # classification_head = model.fc if hasattr(model, 'fc') else model.output  # Get the classification head
        import copy
        import itertools

        # Assuming model is a ResNet or similar model with a classification head
        classification_head = copy.deepcopy(model.fc)  # Make a deep copy of the classification head
        model.fc = torch.nn.Identity()
        # classification_head = torch.nn.Linear(in_features=64, out_features=100)

        # Combine parameters from feature_extractor and classification_head
        params = itertools.chain(model.parameters(), classification_head.parameters())

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
    else: 
        raise Exception(f"Unsupported strategy: {args['strategy']}")
    

    # Training and validation
    results = []
    for experience in benchmark.train_stream:
        print(f"Training on experience: {experience.current_experience}")
        strategy.train(experience, collate_fn=data_collator)
        
        print("Testing...")
        results.append(strategy.eval(benchmark.test_stream))  # Validation after each training experience

    accuracy_matrix = compute_accuracy_matrix(results, num_tasks)

    average_accuracy = compute_average_accuracy_matrix(accuracy_matrix)
    average_incremental_accuracy = compute_average_incremental_accuracy_matrix(average_accuracy)

    forgetting_measure = compute_forgetting_matrix(accuracy_matrix)
    backward_transfer = compute_backward_transfer_matrix(accuracy_matrix)

    pretrained_key = 'pretrained' if args['pretrained'] else 'scratch'
    # for the model_name, consider only the name stem
    file_path = args['res_file_template'].format(args['model_name'].split('/')[-1], pretrained_key, num_tasks, args['strategy'], args['train_epochs'])
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
