import sys
import os

# Add project_root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from avalanche.benchmarks import CLScenario, CLStream, CLExperience
from avalanche.benchmarks.utils import AvalancheDataset, ConstantSequence
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import (
    loss_metrics, 
    accuracy_metrics
)

from avalanche.benchmarks.utils import DataAttribute
from datasets import Dataset
from datasets import load_dataset

from utils.compute_metrics import *
from utils.sequence_classification import HGNaive, CustomDataCollatorSeq2SeqBeta

import avalanche
from datasets import Dataset, DatasetDict
from typing import Optional, Union, Any
from transformers.utils import PaddingStrategy
from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase

device = torch.device(
    f"cuda" if torch.cuda.is_available() else "cpu"
)
# Check dataset structure
# Load the dataset
# def filter_none_entries(dataset):
#     # Function to filter entries with `None` in "text"
#     filtered_dataset = {}
#     for split in dataset:  # Iterate through splits (e.g., "train", "test")
#         filtered_dataset[split] = {
#             "text": [],
#             "label": [],
#             "domain": []
#         }
#         for text, label, domain in zip(dataset[split]["text"], dataset[split]["label"], dataset[split]["domain"]):
#             if text is not None:  # Keep entries where `text` is not None
#                 filtered_dataset[split]["text"].append(text)
#                 filtered_dataset[split]["label"].append(label)
#                 filtered_dataset[split]["domain"].append(domain)

#     return filtered_dataset

# dataset = load_dataset("json", data_files={"train": "data/sequence_classification_train.json", "test": "data/sequence_classification_test.json"})

# # Tokenizer and model for sequence classification
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# # Preprocessing function for sequence classification
# def preprocess_function(examples):
#     inputs = tokenizer(examples["text"], max_length=256, truncation=True, padding=True)
#     inputs["labels"] = examples["label"]
#     inputs["domain"] = examples["domain"]
#     return inputs

# # print(f"datasets before processing: {dataset}")

# dataset = filter_none_entries(dataset)

# train_dataset = Dataset.from_dict(dataset["train"])
# test_dataset = Dataset.from_dict(dataset["test"])


# # Combine into a DatasetDict
# dataset = DatasetDict({
#     "train": train_dataset,
#     "test": test_dataset
# })

# dataset = dataset.map(preprocess_function, batched=True)

# # Incremental Domain Setup: Split dataset by domains
# domains = set(dataset["train"]["domain"])
# num_tasks = len(domains)
# train_exps = []
# test_exps = []
# data_collator = CustomDataCollatorSeq2SeqBeta(tokenizer=tokenizer, model=model)

# for task_id, domain in enumerate(domains):
#     # Use the dataset for this domain as an experience
#     domain_train = dataset.filter(lambda x: x["domain"] == domain)["train"]
#     domain_test = dataset.filter(lambda x: x["domain"] == domain)["test"]

#     columns_to_keep = ["input_ids", "attention_mask", "labels"]
    
#     domain_train = domain_train.select_columns(columns_to_keep)
#     domain_train = domain_train.select(range(1))
#     domain_test = domain_test.select_columns(columns_to_keep)
#     domain_test = domain_test.select(range(1))

#     # Create training experience
#     tl_train = DataAttribute(ConstantSequence(task_id, len(domain_train)), "targets_task_labels")
#     exp_train = CLExperience(task_id, None)
#     exp_train.dataset = AvalancheDataset(
#         [domain_train],
#         data_attributes=[tl_train],
#         collate_fn=data_collator
#     )
#     train_exps.append(exp_train)

#     tl_test = DataAttribute(ConstantSequence(task_id, len(domain_test)), "targets_task_labels")
#     exp_test = CLExperience(task_id, None)
#     exp_test.dataset = AvalancheDataset(
#         [domain_test],
#         data_attributes=[tl_test],
#         collate_fn=data_collator
#     )

#     test_exps.append(exp_test)

# # Create incremental benchmark scenario
# benchmark = CLScenario(
#     [
#         CLStream("train", train_exps),  # Training stream
#         CLStream("test", test_exps),     # Validation stream
#     ]
# )

from utils.amazon_review import AmazonReviewDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
data_collator = CustomDataCollatorSeq2SeqBeta(tokenizer=tokenizer, model=model)
amazon_reviews = AmazonReviewDataset(num_samples_per_domain=1000, domain_groups=None, tokenizer=tokenizer, model=model)
benchmark, num_tasks = amazon_reviews.create_benchmark()

# Evaluation plugin
eval_plugin = EvaluationPlugin(
    # loss_metrics(epoch=True, experience=True, stream=True),
    accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=False),
    loggers=[InteractiveLogger()],
)
# Replay plugin
# plugins = [eval_plugin]

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Strategy for incremental learning
strategy = HGNaive(
    model,
    optimizer,
    torch.nn.CrossEntropyLoss(),
    # evaluator=eval_plugin,
    train_epochs=1,
    train_mb_size=1,
    eval_mb_size=1,
    # plugins=plugins,
    evaluator=eval_plugin,
    device=device
)

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

file_path = 'results/test_sequence_to_sequence.json'
save_results_to_file(file_path, accuracy_matrix, average_accuracy, average_incremental_accuracy, forgetting_measure, backward_transfer)
    
