from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import pandas as pd
import random
import json

# Load the dataset from Hugging Face Hub
dataset = load_dataset("JSSICE/Multi-Domain-Sentiment-Dataset")

def extract_between(text):
    stringA, stringB = "<review_text>", "</review_text>"
    # Find the start and end indices of the substrings
    start_idx = text.find(stringA)
    end_idx = text.find(stringB, start_idx + len(stringA))
    
    # Check if both substrings are found
    if start_idx != -1 and end_idx != -1:
        # Extract and return the substring
        return text[start_idx + len(stringA):end_idx]
    return None  # Return None if one of the substrings is not found

# print(f"dataset: {dataset}")
# print(f"dataset['train]: {dataset['train']}")

# __key__ is of the form sorted_data_acl/<domain>/{positive, negative, unlabeled}
data = {"domain": [], "text": [], "label": []}

for i in range(len(dataset['train'])): 
    # print(f"{i}-th entry: {dataset['train'][i]['__key__']}")
    
    label = dataset['train'][i]['__key__'].split('/')[2]
    # print(f"label: {label}")
    if label == "negative": 
        label = 0
    elif label == "positive": 
        label = 1
    else: 
        print(f"skipping unlabelled data...")
        continue

    domain = dataset['train'][i]['__key__'].split('/')[1]
    
    raw_reviews = dataset['train'][i]['review'].decode('utf-8').split('\n</review>')
    texts = list(map(extract_between, raw_reviews))

    for text in texts: 
        data["domain"].append(domain)
        data["text"].append(text)
        data["label"].append(label)

    print(f"domain: {domain}, label: {label}, len(texts): {len(texts)}")

number_train_entries = int(len(data["text"]) * 0.8)
indices_train = random.sample(range(len(data["text"])), number_train_entries)

train_test_split = {"train": {"domain": [], "text": [], "label": []}, "test": {"domain": [], "text": [], "label": []}}

for i in range(len(data["text"])): 
    if i in indices_train: 
        train_test_split["train"]["domain"].append(data["domain"][i])
        train_test_split["train"]["text"].append(data["text"][i])
        train_test_split["train"]["label"].append(data["label"][i])
    else:
        train_test_split["test"]["domain"].append(data["domain"][i])
        train_test_split["test"]["text"].append(data["text"][i])
        train_test_split["test"]["label"].append(data["label"][i])

# save the dataset
# Save train and test splits to separate JSON files
for split in ["train", "test"]:
    output_file = f"data/sequence_classification_{split}.json"
    with open(output_file, "w") as f:
        json.dump(train_test_split[split], f, indent=4)
    print(f"{split.capitalize()} split saved to {output_file}")

# building on here, we can construct the dataset pretty easily. 

# # Filter the dataset to include only positive (label=1) or negative (label=0) samples
# def filter_labeled_samples(example):
#     return example['label'] in [0, 1]

# filtered_datasets = {}
# for split in dataset.keys():  # Split keys like "train", "test", etc.
#     filtered_datasets[split] = dataset[split].filter(filter_labeled_samples)

# print(f"filtered_datasets.keys(): {filtered_datasets.keys()}")

# # Combine domains into a single DatasetDict
# combined_datasets = DatasetDict()
# for split, data in filtered_datasets.items():
#     # Convert to Pandas DataFrame for easier processing by domain
#     df = pd.DataFrame(data)
    
#     # Group by domain and split into train and test datasets
#     train_datasets = []
#     test_datasets = []
#     for domain in df["domain"].unique():
#         domain_df = df[df["domain"] == domain]
#         train_size = int(len(domain_df) * 0.8)
#         train_df = domain_df[:train_size]
#         test_df = domain_df[train_size:]
#         train_datasets.append(Dataset.from_pandas(train_df))
#         test_datasets.append(Dataset.from_pandas(test_df))
    
#     # Combine train and test datasets across all domains
#     train_dataset = concatenate_datasets(train_datasets)
#     test_dataset = concatenate_datasets(test_datasets)
#     combined_datasets["train"] = train_dataset
#     combined_datasets["test"] = test_dataset

# # Check the structure of the final dataset
# print(combined_datasets)
