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

from utils.amazon_review import AmazonReviewDataset

def main(args): 
    device = torch.device(
        f"cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    
    # config = BertConfig.from_pretrained(args['model_name'], num_labels=2)
    # model = BertForSequenceClassification(config).to(device)
    
    model = BertForSequenceClassification.from_pretrained(args['model_name'], num_labels=2).to(device)

    data_collator = CustomDataCollatorSeq2SeqBeta(tokenizer=tokenizer, model=model)

    amazon_reviews = AmazonReviewDataset(num_samples_per_domain=args['num_samples_per_domain'], domain_groups=args['domain_groups'], tokenizer=tokenizer, model=model, cache_dir=args['cache_dir'])
    benchmark, num_tasks = amazon_reviews.create_benchmark()

    # Evaluation plugin
    eval_plugin = EvaluationPlugin(
        # loss_metrics(epoch=True, experience=True, stream=True),
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=False),
        loss_metrics(minibatch=True),
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

    file_path = args['res_file_template'].format(args['model_name'], num_tasks, args['strategy'])
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
