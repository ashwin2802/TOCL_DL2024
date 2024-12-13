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

class AmazonReviewDataset:
    """
    A class to manage the download and processing of Amazon Review datasets for various categories.
    """
    def __init__(self, num_samples_per_domain, domain_groups, tokenizer, model):
        self.domain_groups = domain_groups
        self.tokenizer = tokenizer
        self.model = model
        self.num_samples_per_domain = num_samples_per_domain

        # Initialize the list of categories
        self.categories = [
            "All_Beauty",
            "Amazon_Fashion",
            # "Appliances",
            # "Arts_Crafts_and_Sewing",
            # "Automotive",
            # "Baby_Products",
            # "Beauty_and_Personal_Care",
            # "Books",
            # "CDs_and_Vinyl",
            # "Cell_Phones_and_Accessories",
            # "Clothing_Shoes_and_Jewelry",
            # "Digital_Music",
            # "Electronics",
            # "Gift_Cards",
            # "Grocery_and_Gourmet_Food",
            # "Handmade_Products",
            # "Health_and_Household",
            # "Health_and_Personal_Care",
            # "Home_and_Kitchen",
            # "Industrial_and_Scientific",
            # "Kindle_Store",
            # "Magazine_Subscriptions",
            # "Movies_and_TV",
            # "Musical_Instruments",
            # "Office_Products",
            # "Patio_Lawn_and_Garden",
            # "Pet_Supplies",
            # "Software",
            # "Sports_and_Outdoors",
            # "Subscription_Boxes",
            # "Tools_and_Home_Improvement",
            # "Toys_and_Games",
            # "Video_Games"
        ]

    def filter_and_transform_dataset(self, dataset):
        """
        Filters and transforms the dataset to:
        - Include only entries where "rating" is a number and "text" is a non-empty string.
        - Keep only the "text" and "rating" fields.
        - Replace "rating" with "label", where "label" is 0 if rating <= 2.5, and 1 otherwise.
        - Balance the dataset between positive and negative samples.

        Parameters:
        dataset (Dataset): The dataset to filter and transform. Entries of the dataset are of the form
            {"text": ..., "label": ...}.
        num_samples (int): The total number of samples to keep (half positive, half negative).

        Returns:
        Dataset: The filtered and transformed dataset.
        """
        # TODO: remove this!
        dataset = dataset.select(range(2000))

        def is_valid(entry):
            # Check if rating is a number and text is a non-empty string
            return isinstance(entry.get("rating"), (int, float)) and isinstance(entry.get("text"), str) and len(entry["text"].strip()) > 0

        def transform(entry):
            # Keep only "text" and "rating", rename "rating" to "label"
            return {
                "text": entry["text"],
                "label": 0 if entry["rating"] <= 2.5 else 1,
                # "domain": domain
            }

        # Filter dataset
        filtered_dataset = dataset.filter(is_valid)
        transformed_dataset = filtered_dataset.map(transform)
        transformed_dataset = transformed_dataset.select_columns(["text", "label"])

        # Balance the dataset
        positive_samples = transformed_dataset.filter(lambda x: x["label"] == 1)
        negative_samples = transformed_dataset.filter(lambda x: x["label"] == 0)

        # Determine number of samples per class
        if self.num_samples_per_domain:
            num_samples_per_class = self.num_samples_per_domain // 2
            positive_samples = positive_samples.select(range(min(len(positive_samples), num_samples_per_class)))
            negative_samples = negative_samples.select(range(min(len(negative_samples), num_samples_per_class)))

       # Concatenate and shuffle
        balanced_dataset = concatenate_datasets([positive_samples, negative_samples]).shuffle()
        return balanced_dataset

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
        try:
            # Load the dataset
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_review_{category}",
                trust_remote_code=True
            )
            print(f"Successfully downloaded dataset for category: {category}")
            
            # Get the "full" split
            full_split = dataset["full"]
            
            # Filter, transform, and balance the dataset
            processed_split = self.filter_and_transform_dataset(full_split)
            
            print(f"Processed and balanced dataset size for {category}: {len(processed_split)}")
            # print(f"processed_split: {processed_split}")
            return processed_split
        except Exception as e:
            print(f"An error occurred while downloading or processing the dataset for category '{category}': {e}")
            return None

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
            df, test_size=test_size, stratify=df["label"], random_state=42
        )
        return DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df)
        })
    
    def create_benchmark(self):
        """
        Creates training and testing streams for incremental domain learning.

        Parameters:
        dataset_dict (dict): Dictionary of datasets with "train" and "test" splits.
        domain_groups (list of sets): Groups of domains to combine. If None, treat domains individually.
        tokenizer: Tokenizer object for preprocessing text.
        model: Model for custom data collator.

        Returns:
        CLScenario: Incremental learning benchmark scenario.
        """

        dataset_dict = self.download_all()

        # Preprocess dataset for sequence classification
        def preprocess_function(examples):
            inputs = self.tokenizer(examples["text"], max_length=256, truncation=True, padding=True)
            inputs["labels"] = examples["label"]
            # inputs["domain"] = examples["domain"]
            return inputs

        # Apply preprocessing and keep only specified columns
        dataset_dict = {
            key: value.map(preprocess_function, batched=True).select_columns(["input_ids", "attention_mask", "labels"])
            for key, value in dataset_dict.items()
        }
        if self.domain_groups is None:
            domain_groups = [{domain} for domain in dataset_dict.keys()]

        train_exps = []
        test_exps = []
        data_collator = CustomDataCollatorSeq2SeqBeta(tokenizer=self.tokenizer, model=self.model)

        for task_id, group in enumerate(domain_groups):
            train_datasets = []
            test_datasets = []

            for domain in group:
                domain_dataset = dataset_dict[domain]
                domain_dataset = self.stratify_split(domain_dataset)
                train_datasets.append(domain_dataset["train"])
                test_datasets.append(domain_dataset["test"])

            combined_train = concatenate_datasets(train_datasets)
            combined_test = concatenate_datasets(test_datasets)

            # Training experience
            tl_train = DataAttribute(ConstantSequence(task_id, len(combined_train)), "targets_task_labels")
            exp_train = CLExperience(task_id, None)
            exp_train.dataset = AvalancheDataset(
                [combined_train],
                data_attributes=[tl_train],
                collate_fn=data_collator
            )
            train_exps.append(exp_train)

            # Testing experience
            tl_test = DataAttribute(ConstantSequence(task_id, len(combined_test)), "targets_task_labels")
            exp_test = CLExperience(task_id, None)
            exp_test.dataset = AvalancheDataset(
                [combined_test],
                data_attributes=[tl_test],
                collate_fn=data_collator
            )
            test_exps.append(exp_test)

        # Create incremental benchmark scenario
        return CLScenario(
            [
                CLStream("train", train_exps),
                CLStream("test", test_exps),
            ]
        ), len(train_exps)


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, BertForSequenceClassification

    # Create an instance of the class
    amazon_reviews = AmazonReviewDataset()
    # Tokenizer and model for sequence classification
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Download and process datasets for all categories with 1000 samples each
    all_datasets = amazon_reviews.create_benchmark(num_samples_per_category=1000, domain_groups=None, tokenizer=tokenizer, model=model)

    # # Example: Accessing the processed dataset for a specific category
    # if "All_Beauty" in all_datasets:
    #     print(all_datasets["All_Beauty"][0])  # Print the first entry of the "All_Beauty" dataset
