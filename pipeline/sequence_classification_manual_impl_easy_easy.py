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

from utils.amazon_review_manual_impl import AmazonReviewDataset 
import os

from utils.compute_metrics import *

# Load BERT tokenizer and model
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=21).to(device) 

# Memory buffer for exemplars
memory_buffer = {}

# Loss function for knowledge distillation
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_probs = torch.nn.functional.softmax(student_logits / temperature, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    return torch.nn.functional.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')

def get_infinite_dataloader(dataloader):
    """
    Creates an infinite dataloader that restarts once the original dataloader is exhausted.

    Parameters:
    - dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object.

    Yields:
    - Batch data from the dataloader, restarting from the beginning when exhausted.
    """
    while True:  # Infinite loop
        for batch in dataloader:
            yield batch

def train_domain(model, train_dataset, exemplars, memory_buffer, num_epochs=1, batch_size=32, learning_rate=2e-5):
    """
    Train the model for a single domain, integrating distillation loss using exemplars.

    Parameters:
    - model: Pre-trained transformer model (e.g., BertForSequenceClassification).
    - tokenizer: Tokenizer for preprocessing.
    - train_dataset: Dataset for training the current domain.
    - exemplars: Exemplar dataset for distillation.
    - memory_buffer: Memory buffer for exemplars (not directly used here but assumed available).
    - num_epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for optimizer.
    """
    from torch.utils.data import DataLoader
    from transformers import AdamW

    # Create dataloaders for train dataset and exemplars
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if exemplars:
        exemplar_loader = DataLoader(exemplars, batch_size=batch_size, shuffle=True)
        exemplar_infinite_loader = get_infinite_dataloader(exemplar_loader)

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

            # Add distillation loss if exemplars exist
            if exemplars:
                # Draw a batch from the exemplars
                exemplar_batch = next(exemplar_infinite_loader)
                exemplar_input_ids = torch.stack(exemplar_batch['input_ids'], dim=1)
                exemplar_attention_mask = torch.stack(exemplar_batch['attention_mask'], dim=1)
                exemplar_inputs = {
                    'input_ids': exemplar_input_ids.to(device),
                    'attention_mask': exemplar_attention_mask.to(device),
                    'labels': exemplar_batch['labels'].to(device)
                }

                # Compute teacher logits from exemplars
                with torch.no_grad():
                    teacher_logits = model(**exemplar_inputs).logits

                # Compute student logits for the current training batch
                student_logits = outputs.logits

                # take the minimum of the two
                min_batch_size = min(teacher_logits.shape[0], student_logits.shape[0])

                teacher_logits = teacher_logits[:min_batch_size]
                student_logits = student_logits[:min_batch_size]

                # Compute distillation loss
                distill_loss = distillation_loss(student_logits, teacher_logits)
                loss += 200 * distill_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"Batch {batch_idx + 1} | Loss: {loss.item():.4f}")

import torch
from sklearn.metrics import accuracy_score

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

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch

def validation_with_class_means(model, datasets, batch_size, memory_buffer):
    """
    Validates the model using the nearest class mean approach and calculates average accuracy for each group.

    Parameters:
    - model (BertForSequenceClassification): The model to validate.
    - datasets (dict): A dictionary where keys are group names and values are datasets.
    - batch_size (int): Batch size for the DataLoader.
    - memory_buffer (dict): Dictionary with class keys and their exemplar samples.

    Returns:
    - results (list): A list of average accuracies for each group.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    results = []
    print(f"--- INSIDE VALIDATION WITH CLASS MEANS ---")
    # Sort group keys based on numeric part of the name
    sorted_keys = sorted(
        datasets.keys(), 
        key=lambda x: int(x.split('_')[1])  # Extract numeric part for sorting
    )

    # Precompute class means
    class_means = {}
    for cls, samples in memory_buffer.items():
        print(f"computing class means for class: {cls}")
        class_means[cls] = compute_class_mean(model, samples, device)  # Precompute mean embeddings

    print(f"sorted_keys: {sorted_keys}")
    for group in sorted_keys:
        group_dataset = datasets[group]

        # Instantiate DataLoader for the current group
        group_loader = DataLoader(group_dataset, batch_size=batch_size, shuffle=False)

        # Track predictions and true labels
        all_preds = []
        all_labels = []

        for batch in group_loader:
            # Prepare inputs for the current batch
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
            labels = batch['labels'].to(device)

            # Extract features for the batch
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                features = outputs.hidden_states[-1][:, 0, :]  # CLS embeddings

            # Compute distances to class means for each feature in the batch
            preds = []
            for feature in features:
                distances = {cls: torch.norm(feature - mean) for cls, mean in class_means.items()}
                closest_class = min(distances, key=distances.get)  # Class with the smallest distance
                preds.append(closest_class)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy for the current group
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Group '{group}' Accuracy: {accuracy:.4f}")
        results.append(accuracy)

    return results

def compute_class_mean(model, examples, device):
    """
    Computes the mean of CLS embeddings for a set of examples using a model.

    Parameters:
    - model (BertForSequenceClassification): The model used to extract CLS embeddings.
    - examples (list): List of tokenized examples (output of the tokenizer).
    - device: The device (CPU/GPU) to use.

    Returns:
    - class_mean (torch.Tensor): Mean CLS embedding for the given examples.
    """
    model.eval()
    model.to(device)

    # Prepare batches of inputs
    # input_ids = [example]
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in examples]).to(device)
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in examples]).to(device)

    with torch.no_grad():
        # Forward pass through the model to get hidden states
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # CLS embeddings are the [CLS] token's representation in the last hidden layer
        cls_embeddings = hidden_states[-1][:, 0, :]  # Shape: (batch_size, hidden_size)

    # Compute mean of CLS embeddings
    class_mean = cls_embeddings.mean(dim=0)  # Shape: (hidden_size)
    return class_mean
    
# Nearest-Mean-of-Exemplars classification
def classify_with_exemplars(features, memory_buffer):
    class_means = {}
    for cls, examples in memory_buffer.items():
        class_means[cls] = torch.mean(examples, dim=0)

    distances = {cls: torch.norm(features - mean) for cls, mean in class_means.items()}
    return min(distances, key=distances.get)

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

"""
Retrieves a dataset of the form {"train": {"domain_1", "domain_2", ...}, "test": {"domain_1", "domain_2", ...}}.
Assumes the existance of a domain2id_mapping, and a domain_groups
"""
amazon_reviews = AmazonReviewDataset(tokenizer=tokenizer, cache_dir='/cluster/scratch/rrigoni/.cache/huggingface')

groups = amazon_reviews.categories[:5]
random.shuffle(groups)

domain_groups = [[group] for group in groups]
grouped_dataset, domain2id_mapping = amazon_reviews.create_benchmark(domain_groups)

sorted_keys = sorted(
    grouped_dataset['train'].keys(),  # Get the keys
    key=lambda x: int(x.split('_')[1])  # Extract the integer part (dd) and sort by it
)

results = []
for group in sorted_keys:
    new_dataset = grouped_dataset['train'][group]

    exemplars = []
    for _, examples in memory_buffer.items():
        exemplars.extend(examples)

    train_domain(model, new_dataset, exemplars, memory_buffer)

    update_memory_buffer(memory_buffer, new_dataset, buffer_size=5212)

    # validation(model, grouped_dataset['test'], 16)
    validation_scores = validation_with_class_means(model, grouped_dataset['test'], 16, memory_buffer)

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

# file_path = args['res_file_template'].format(args['model_name'], args['depth'], args['num_tasks'], args['strategy'])
file_path = "results/example_icarl_sequence_classification.json"
save_results_to_file(file_path, accuracy_matrix, average_accuracy, average_incremental_accuracy, forgetting_measure, backward_transfer)
    
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config",
#         type=str,
#         help=""
#     )

#     args = parser.parse_args()

#     with open(args.config, 'r') as f: 
#         config = json.load(f)

#     main(config)