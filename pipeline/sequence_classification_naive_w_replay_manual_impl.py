import sys
import os

# Add project_root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import random

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch

from utils.amazon_review_manual_impl import AmazonReviewDataset 
import os

import torch
from sklearn.metrics import accuracy_score
import argparse

from utils.compute_metrics import *

def train_domain(model, train_dataset, memory_buffer, num_epochs=1, batch_size=100, learning_rate=2e-5):
   """
    Train the model for a single domain, integrating memory buffer examples for experience replay.

    Parameters:
    - model: Pre-trained transformer model (e.g., BertForSequenceClassification).
    - train_dataset: Dataset for training the current domain.
    - memory_buffer: Dictionary with exemplars for each key (domain or class).
    - num_epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for optimizer.
    """
    from torch.utils.data import DataLoader, ConcatDataset
    from transformers import AdamW
    import torch

    # Convert memory buffer entries to TensorDataset
    buffer_datasets = []
    for examples in memory_buffer.values():
        input_ids = torch.tensor([entry['input_ids'] for entry in examples], dtype=torch.long)
        attention_mask = torch.tensor([entry['attention_mask'] for entry in examples], dtype=torch.long)
        labels = torch.tensor([entry['labels'] for entry in examples], dtype=torch.long)

        buffer_datasets.append(TensorDataset(input_ids, attention_mask, labels))

    # Combine current training dataset with memory buffer
    if buffer_datasets:
        combined_dataset = ConcatDataset([train_dataset, *buffer_datasets])
    else:
        combined_dataset = train_dataset  # Use only the current training dataset if no buffer


    # Create dataloaders for train dataset and exemplars
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        for batch_idx, batch in enumerate(train_loader):
            # Prepare inputs for the current training batch
            input_ids = torch.stack(batch['input_ids'], dim=1)
            attention_mask = torch.stack(batch['attention_mask'], dim=1)
            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
                'labels': batch['labels'].to(device)
            }

            # Forward pass for training data
            outputs = model(**inputs)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def validation(model, datasets, batch_size):
    """
    Validates the model on multiple groups and calculates average accuracy for each group.

    Parameters:
    - model (BertForSequenceClassification): The model to validate.
    - datasets (dict): A dictionary where keys are group names and values are datasets.
    - batch_size (int): Batch size for the DataLoader.

    Returns:
    - results (list): A list of average accuracies for each group.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    results = []
    sorted_keys = sorted(
        datasets.keys(),  # Get the keys
        key=lambda x: int(x.split('_')[1])  # Extract the integer part (dd) and sort by it
    )

    groups = sorted_keys

    for group in groups:
        group_dataset = datasets[group]

        # Instantiate DataLoader
        group_loader = DataLoader(group_dataset, batch_size=batch_size, shuffle=False)

        # Track predictions and true labels
        all_preds = []
        all_labels = []

        for batch in group_loader:
            # Prepare inputs for the current batch
            input_ids = torch.stack(batch['input_ids'], dim=1)
            attention_mask = torch.stack(batch['attention_mask'], dim=1)
            labels = batch['labels']

            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
                'labels': labels.to(device)
            }

            # Forward pass
            with torch.no_grad():  # Disable gradient computation for validation
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

            # Store predictions and true labels
            # print(f"preds: {preds}\nlabels: {labels}")
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy for the current group
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Group '{group}' Accuracy: {accuracy:.4f}")
        results.append(accuracy)

    return results

def print_overview_memory_buffer(memory_buffer): 
    print(f"------ OVERVIEW MEMORY BUFFER ------")
    for key, value in memory_buffer.items(): 
        print(f"{key}: {len(value)}")

def update_memory_buffer(memory_buffer, new_dataset, buffer_size):
    """
    Updates the memory buffer with samples from the new dataset, grouped by domain.

    Parameters:
    - memory_buffer (dict): Current memory buffer with domain keys and their samples.
    - new_dataset (list): New dataset with samples to add to the buffer.
    - buffer_size (int): Total size of the memory buffer.

    Returns:
    - memory_buffer (dict): Updated memory buffer.
    """
    import random
    from collections import defaultdict

    # Identify new domains in the new dataset
    new_domains = list(set(sample['labels'] for sample in new_dataset))
    print(f"new_domains: {new_domains}")

    # Calculate number of samples to retain per domain
    samples_per_domain = buffer_size // (len(memory_buffer.keys()) + len(new_domains))
    print(f"samples_per_domain: {samples_per_domain}")

    # Update existing domains in memory buffer
    for domain, samples in memory_buffer.items():
        memory_buffer[domain] = samples[:min(len(samples), samples_per_domain)]

    # Group new samples by domain
    grouped_samples = defaultdict(list)
    for sample in new_dataset:
        grouped_samples[sample['labels']].append(sample)

    for key, value in grouped_samples.items(): 
        print(f"grouped_samples[{key}]: {len(value)}")

    # Add new domains to memory buffer
    for new_domain, domain_samples in grouped_samples.items():
        memory_buffer[new_domain] = random.sample(domain_samples, min(len(domain_samples), samples_per_domain))

    print_overview_memory_buffer(memory_buffer)
    return memory_buffer


def main(args): 
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args['model_name'])
    model = BertForSequenceClassification.from_pretrained(args['model_name'], num_labels=21).to(device) 

    amazon_reviews = AmazonReviewDataset(tokenizer=tokenizer, cache_dir=args['cache_dir'])

    if args['domain_groups'] and args['domain_groups'].startswith('random'):
        size_per_group = int(args['domain_groups'].split('-')[-1])  # Extract 'd' from 'random-d'
        groups = amazon_reviews.categories  # List of domain categories
        
        domain_groups = []
        for i in range(0, len(groups), size_per_group):  # Step through the list in chunks of size 'd'
            domain_groups.append(groups[i:i + size_per_group])  # Add a chunk of size 'd' (or smaller for the last one)

        print("Domain Groups:", domain_groups)
    
    grouped_dataset, domain2id_mapping = amazon_reviews.create_benchmark(domain_groups)

    sorted_keys = sorted(
        grouped_dataset['train'].keys(),  # Get the keys
        key=lambda x: int(x.split('_')[1])  # Extract the integer part (dd) and sort by it
    )

    memory_buffer = {}
    results = []
    for group in sorted_keys:
        new_dataset = grouped_dataset['train'][group]

        train_domain(model, new_dataset, memory_buffer, args['train_epochs'], args['train_mb_size'])

        update_memory_buffer(memory_buffer, new_dataset, buffer_size=5212)

        validation_scores = validation(model, grouped_dataset['test'], args['test_mb_size'])

        validation_scores_dict = {}
        template = "Top1_Acc_Exp/Exp{:03d}"
        for idx, score in enumerate(validation_scores): 
            validation_scores_dict[template.format(idx)] = score

        print(f"validation_scores_dict: {validation_scores_dict}")
        results.append(validation_scores_dict)

    num_tasks = len(results)
    accuracy_matrix = compute_accuracy_matrix(results, num_tasks)

    average_accuracy = compute_average_accuracy_matrix(accuracy_matrix)
    average_incremental_accuracy = compute_average_incremental_accuracy_matrix(average_accuracy)

    forgetting_measure = compute_forgetting_matrix(accuracy_matrix)
    backward_transfer = compute_backward_transfer_matrix(accuracy_matrix)

    file_path = args['res_file_template'].format(args['model_name'], num_tasks, args['strategy'], args['train_epochs'])
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