from datasets import (
    load_dataset, 
    concatenate_datasets,
    Dataset, 
    DatasetDict
)
from sklearn.model_selection import train_test_split
from avalanche.benchmarks import CLScenario, CLStream, CLExperience
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils import DataAttribute
from avalanche.benchmarks.utils.data_attribute import ConstantSequence
from utils.sequence_classification import CustomDataCollatorSeq2SeqBeta

from tqdm import tqdm
import os

class AmazonReviewDataset:
    """
    A class to manage the download and processing of Amazon Review datasets for various categories.
    """
    def __init__(self, tokenizer, cache_dir=None):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir

        # Initialize the list of categories
        self.categories = [
            "All_Beauty",
            "Amazon_Fashion",
            "Appliances",
            "Arts_Crafts_and_Sewing",
            "Automotive",
            "Baby_Products",
            "Books",
            "CDs_and_Vinyl",
            "Gift_Cards",
            "Handmade_Products",
            "Health_and_Personal_Care",
            "Magazine_Subscriptions",
            "Subscription_Boxes",
            "Toys_and_Games",
            "Video_Games",
            "Grocery_and_Gourmet_Food",
            "Industrial_and_Scientific",
            "Movies_and_TV",
            "Musical_Instruments",
            "Pet_Supplies",
            "Sports_and_Outdoors",
        ]


    def filter_and_transform_dataset(self, dataset):
        """
        Filters and transforms the dataset to:
        - Include only entries where "rating" is a number and "text" is a non-empty string.
        - Shuffle the dataset after filtering.
        - Take a balanced number of positive and negative entries based on the threshold (rating <= 2.5).
        - Apply the transformation only to the selected subset.

        Parameters:
        dataset (Dataset): The dataset to filter and transform.
        num_samples (int): The total number of samples to keep (half positive, half negative).

        Returns:
        Dataset: The filtered, balanced, and transformed dataset.
        """

        def is_valid(entry):
            # Check if rating is a number and text is a non-empty string
            return isinstance(entry.get("rating"), (int, float)) and isinstance(entry.get("text"), str) and len(entry["text"].strip()) > 0

        # Shuffle the filtered dataset
        shuffled_dataset = dataset.shuffle()
        # Split into positive and negative entries based on the threshold (rating <= 2.5)
        positive_samples = []
        negative_samples = []
        for entry in tqdm(shuffled_dataset):
            if entry["rating"] > 2.5 and len(positive_samples) < self.num_samples_per_domain // 2:
                positive_samples.append(entry)
            elif entry["rating"] <= 2.5 and len(negative_samples) < self.num_samples_per_domain // 2:
                negative_samples.append(entry)
            if len(positive_samples) == self.num_samples_per_domain // 2 and len(negative_samples) == self.num_samples_per_domain // 2:
                break

        print(f"len(positive_samples): {len(positive_samples)}, len(negative_samples): {len(negative_samples)}")

        # Combine positive and negative samples into a single dataset
        balanced_samples = positive_samples + negative_samples
         # Convert the list of samples into a dictionary of lists
        balanced_dict = {key: [entry[key] for entry in balanced_samples] for key in balanced_samples[0].keys()}

        # Transform the balanced samples
        def transform(entry):
            return {
                "text": entry["text"],
                "label": 0 if entry["rating"] <= 2.5 else 1,
            }

        # Apply transformation to the balanced dataset
        transformed_dataset = Dataset.from_dict(balanced_dict).filter(is_valid).map(transform)
        transformed_dataset = transformed_dataset.select_columns(["text", "label"])

        return transformed_dataset


    def download(self, category):
        """
        Downloads the Amazon Reviews dataset for a given category, keeps only the "full" split,
        filters and transforms the dataset, and balances the dataset.

        Parameters:
        category (str): The category name.
        num_samples_per_category (int): The total number of samples to keep (half positive, half negative).

        Returns:
        Dataset: The filtered and balanced "full" split of the dataset, or None if an error occurs.
        """
        if self.cache_dir: 
            # Load the dataset
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_review_{category}",
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        else: 
            # Load the dataset
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_review_{category}",
                trust_remote_code=True
            )
        print(f"Successfully downloaded dataset for category: {category}")
        
        # # Get the "full" split
        # full_split = dataset["full"]
        full_split = dataset["full"]
        
        # Filter, transform, and balance the dataset
        processed_split = self.filter_and_transform_dataset(full_split)
        
        print(f"Processed and balanced dataset size for {category}: {len(processed_split)}")
        # print(f"processed_split: {processed_split}")
        return processed_split

    def download_all(self):
        """
        Downloads and processes the Amazon Reviews datasets for all valid categories,
        balancing each dataset and keeping the specified number of samples.

        Parameters:
        num_samples_per_category (int): The total number of samples to keep for each category
                           (half positive, half negative).

        Returns:
        dict: A dictionary where the keys are category names and the values are the processed datasets.
        """
        datasets = {}
        for category in self.categories:
            print(f"Processing category: {category}")
            processed_dataset = self.download(category)
            if processed_dataset:
                datasets[category] = processed_dataset
        return datasets
    
    def stratify_split(self, dataset, test_size=0.2):
        """
        Creates a stratified train-test split for a given dataset.

        Parameters:
        dataset (Dataset): The dataset to split.
        test_size (float): Fraction of the dataset to use for testing.

        Returns:
        DatasetDict: Train-test split as a DatasetDict.
        """
        # Convert Dataset to pandas DataFrame for stratified splitting
        df = dataset.to_pandas()
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df["labels"], random_state=42
        )
        return DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df)
        })
    
    def create_benchmark(self, domain_groups):
        """
        Creates a dataset with train and test splits, organized by domains.
        Each domain in the splits contains input IDs, attention masks, and labels.

        Returns:
        dict: Dictionary with "train" and "test" splits, where each split is 
            further organized by domains.
        """
        from datasets.config import HF_DATASETS_CACHE
        import os

        # Use default cache directory for Hugging Face datasets
        processed_dataset_path = os.path.join(self.cache_dir, "amazon_review_dataset")

        # Check if the preprocessed dataset already exists in the cache
        if os.path.exists(processed_dataset_path):
            print("Loading preprocessed dataset from cache directory...")
            dataset_dict = DatasetDict.load_from_disk(processed_dataset_path)
        else:
            print("Creating file because missing")
            # Download and preprocess the dataset
            dataset_dict = self.download_all()
            DatasetDict(dataset_dict).save_to_disk(processed_dataset_path)
            print(f"Dataset saved at path: {processed_dataset_path}")
        
        # Preprocess dataset for sequence classification
        def preprocess_function(examples):
            inputs = self.tokenizer(examples["text"], max_length=256, truncation=True, padding="max_length", return_tensors="pt")
            inputs["labels"] = examples["label"]
            return inputs

        # Apply preprocessing to the dataset
        dataset_dict = {
            key: value.map(preprocess_function, batched=True)
            for key, value in dataset_dict.items()
        }

        if domain_groups: 
            domain2id_mapping = {}
            counter = 0
            for group in domain_groups: 
                for domain in group: 
                    domain2id_mapping[domain] = counter
                    counter += 1
        else: 
            # return result_dataset, domain2id_mapping
            raise Exception(f"Please specify domain_groups (generate random ones outside the function invokation)")

        # Create domain groups if specified, otherwise use individual domains
        initial_domain_groups = [{domain} for domain in domain2id_mapping.keys()]

        # Split the dataset into train and test for each domain - using stratified positive and negative labels
        result_dataset = {"train": {}, "test": {}}

        for group in initial_domain_groups:
            for domain in group:
                domain_dataset = dataset_dict[domain]
                # Stratify the split (assumes stratify_split returns a dict with "train" and "test")
                domain_split = self.stratify_split(domain_dataset)
                
                # Add preprocessed train and test splits to result
                result_dataset["train"][domain] = domain_split["train"].select_columns(
                    ["input_ids", "attention_mask", "labels"]
                )
                result_dataset["test"][domain] = domain_split["test"].select_columns(
                    ["input_ids", "attention_mask", "labels"]
                )

        
        for split in result_dataset.keys():  # Iterate over splits (e.g., "train" and "test")
            for domain, dataset in result_dataset[split].items():  # Iterate over domains within the split
                domain_id = domain2id_mapping[domain]
                # Map each entry's label to the domain ID
                result_dataset[split][domain] = dataset.map(
                    lambda batch: {**batch, "labels": [domain_id] * len(batch["labels"])},  # Correctly broadcast domain_id
                    batched=True
                )
        
        if domain_groups: 
            grouped_dataset = {"train": {}, "test": {}}

            for split in result_dataset.keys():  # Iterate over splits ("train", "test")
                for group_idx, group in enumerate(domain_groups):  # Iterate over partitions
                    # print(f"split: {split}, group: {group}")
                    grouped_domains = []
                    for domain in group:
                        if domain in result_dataset[split]:  # Check if domain exists in split
                            grouped_domains.append(result_dataset[split][domain])
                    # Concatenate datasets for the current group
                    if grouped_domains:
                        grouped_dataset[split][f"group_{group_idx}"] = concatenate_datasets(grouped_domains)
            
            return grouped_dataset, domain2id_mapping
        else: 
            # return result_dataset, domain2id_mapping
            raise Exception(f"Please specify domain_groups (generate random ones outside the function invokation)")

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, BertForSequenceClassification

    # Create an instance of the class
    amazon_reviews = AmazonReviewDataset()
    # Tokenizer and model for sequence classification
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    # Create a new configuration for the model
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)

    # Instantiate a model with random weights
    model = BertForSequenceClassification(config)

    # Download and process datasets for all categories with 1000 samples each
    all_datasets = amazon_reviews.create_benchmark(num_samples_per_category=1000, domain_groups=None, tokenizer=tokenizer, model=model)

    # # Example: Accessing the processed dataset for a specific category
    # if "All_Beauty" in all_datasets:
    #     print(all_datasets["All_Beauty"][0])  # Print the first entry of the "All_Beauty" dataset
