import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from avalanche.benchmarks import CLScenario, CLStream, CLExperience
from avalanche.benchmarks.utils import AvalancheDataset, ConstantSequence
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.supervised import Naive
from avalanche.benchmarks.utils import DataAttribute
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import load_dataset

import avalanche
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Optional, Union, Any
from transformers.utils import PaddingStrategy
from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase

@dataclass
class CustomDataCollatorSeq2SeqBeta:
    """The collator is a standard huggingface collate.
    No need to change anything here.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        if (
            self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
    
class HGNaive(avalanche.training.Naive):
    """There are only a couple of modifications needed to
    use huggingface:
    - we add a bunch of attributes corresponding to the batch items,
        redefining mb_x and mb_y too
    - _unpack_minibatch sends the dictionary values to the GPU device
    - forward and criterion are adapted for machine translation tasks.
    """

    @property
    def mb_attention_mask(self):
        return self.mbatch["attention_mask"]

    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch["input_ids"]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch["labels"]

    @property
    def mb_decoder_in_ids(self):
        """Current mini-batch target."""
        return self.mbatch["decoder_input_ids"]

    @property
    def mb_token_type_ids(self):
        return self.mbatch[3]

    def _unpack_minibatch(self):
        """HuggingFace minibatches are dictionaries of tensors.
        Move tensors to the current device."""

        from torch.nn.utils.rnn import pad_sequence
        batched_input_ids = torch.stack(self.mbatch["input_ids"], dim=1) if not isinstance(self.mbatch["input_ids"], torch.Tensor) else self.mbatch["input_ids"]
        batched_attention_mask = torch.stack(self.mbatch["attention_mask"], dim=1) if not isinstance(self.mbatch['attention_mask'], torch.Tensor) else self.mbatch['attention_mask']

        self.mbatch["attention_mask"] = batched_attention_mask
        self.mbatch["input_ids"] = batched_input_ids

    def forward(self):
        out = self.model(
            input_ids=self.mb_x,
            attention_mask=self.mb_attention_mask,
            labels=self.mb_y,
        )
        return out.logits

    def criterion(self):
        mb_output = self.mb_output.view(-1, self.mb_output.size(-1))
        ll = self._criterion(mb_output, self.mb_y.view(-1))
        return ll

<<<<<<< Updated upstream
=======
device = torch.device(
    f"cuda" if torch.cuda.is_available() else "cpu"
)
>>>>>>> Stashed changes
# Check dataset structure
# Load the dataset
def filter_none_entries(dataset):
    # Function to filter entries with `None` in "text"
    filtered_dataset = {}
    for split in dataset:  # Iterate through splits (e.g., "train", "test")
        filtered_dataset[split] = {
            "text": [],
            "label": [],
            "domain": []
        }
        for text, label, domain in zip(dataset[split]["text"], dataset[split]["label"], dataset[split]["domain"]):
            if text is not None:  # Keep entries where `text` is not None
                filtered_dataset[split]["text"].append(text)
                filtered_dataset[split]["label"].append(label)
                filtered_dataset[split]["domain"].append(domain)

    return filtered_dataset

dataset = load_dataset("json", data_files={"train": "data/sequence_classification_train.json", "test": "data/sequence_classification_test.json"})
print(dataset)

# Tokenizer and model for sequence classification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Preprocessing function for sequence classification
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], max_length=256, truncation=True, padding=True)
    inputs["labels"] = examples["label"]
    inputs["domain"] = examples["domain"]
    return inputs

print(f"datasets before processing: {dataset}")

dataset = filter_none_entries(dataset)
<<<<<<< Updated upstream
# Slice to include only the first 20 entries

=======
dataset["train"] = 
>>>>>>> Stashed changes
train_dataset = Dataset.from_dict(dataset["train"])
test_dataset = Dataset.from_dict(dataset["test"])


# Combine into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

dataset = dataset.map(preprocess_function, batched=True)

# Incremental Domain Setup: Split dataset by domains
domains = set(dataset["train"]["domain"])
train_exps = []
test_exps = []
data_collator = CustomDataCollatorSeq2SeqBeta(tokenizer=tokenizer, model=model)

for task_id, domain in enumerate(domains):
    # Use the dataset for this domain as an experience
    domain_train = dataset.filter(lambda x: x["domain"] == domain)["train"]
    domain_test = dataset.filter(lambda x: x["domain"] == domain)["test"]

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    
    domain_train = domain_train.select_columns(columns_to_keep)
    domain_train = domain_train.select(range(20))
    domain_test = domain_test.select_columns(columns_to_keep)
    domain_test = domain_test.select(range(20))

    # Create training experience
    tl_train = DataAttribute(ConstantSequence(task_id, len(domain_train)), "targets_task_labels")
    exp_train = CLExperience(task_id, None)
    exp_train.dataset = AvalancheDataset(
        [domain_train],
        data_attributes=[tl_train],
        collate_fn=data_collator
    )
    train_exps.append(exp_train)

    tl_test = DataAttribute(ConstantSequence(task_id, len(domain_test)), "targets_task_labels")
    exp_test = CLExperience(task_id, None)
    exp_test.dataset = AvalancheDataset(
        [domain_test],
        data_attributes=[tl_test],
        collate_fn=data_collator
    )

    test_exps.append(exp_test)

# Create incremental benchmark scenario
benchmark = CLScenario(
    [
        CLStream("train", train_exps),  # Training stream
        CLStream("test", test_exps),     # Validation stream
    ]
)

# Evaluation plugin
eval_plugin = EvaluationPlugin(
    loss_metrics(epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()],
    strict_checks=False,
)

# Replay plugin
plugins = [ReplayPlugin(mem_size=200), eval_plugin]

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Strategy for incremental learning
strategy = HGNaive(
    model,
    optimizer,
    torch.nn.CrossEntropyLoss(),
    # evaluator=eval_plugin,
    train_epochs=2,
    train_mb_size=1,
    eval_mb_size=1,
    plugins=plugins,
    device=device
)

# Training and validation
for experience in benchmark.train_stream:
    print(f"Training on experience: {experience.current_experience}")
    strategy.train(experience, collate_fn=data_collator)
    
    print("Testing...")
    strategy.eval(benchmark.test_stream)  # Validation after each training experience
