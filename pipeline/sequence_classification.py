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
        # if return_tensors is None:
        #     return_tensors = self.return_tensors
        # labels = (
        #     [feature["labels"] for feature in features]
        #     if "labels" in features[0].keys()
        #     else None
        # )
        # print(f"\n\n\nfeatures: {features}")
        # # We have to pad the labels before calling `tokenizer.pad` as this
        # # method won't pad them and needs them of the
        # # same length to return tensors.
        # if labels is not None:
        #     print(f"inside collator: {labels}")
        #     # max_label_length = max(len(lab) for lab in labels)
        #     max_label_length = len(labels)
        #     if self.pad_to_multiple_of is not None:
        #         max_label_length = (
        #             (max_label_length + self.pad_to_multiple_of - 1)
        #             // self.pad_to_multiple_of
        #             * self.pad_to_multiple_of
        #         )

        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         remainder = [self.label_pad_token_id] * (
        #             max_label_length - len(feature["labels"])
        #         )
        #         if isinstance(feature["labels"], list):
        #             feature["labels"] = (
        #                 feature["labels"] + remainder
        #                 if padding_side == "right"
        #                 else remainder + feature["labels"]
        #             )
        #         elif padding_side == "right":
        #             feature["labels"] = np.concatenate(
        #                 [feature["labels"], remainder]
        #             ).astype(np.int64)
        #         else:
        #             feature["labels"] = np.concatenate(
        #                 [remainder, feature["labels"]]
        #             ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        # prepare decoder_input_ids
        if (
            # labels is not None
            # and self.model is not None
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
        # Pad and stack along dimension 0 (batch dimension)
        # batched_input_ids = pad_sequence(self.mbatch["input_ids"], batch_first=True, padding_value=0).to(self.device)
        # batched_attention_mask = pad_sequence(self.mbatch["attention_mask"], batch_first=True, padding_value=0).to(self.device)
        print(f"self.mbatch['input_ids']: {self.mbatch['input_ids']}")

        batched_input_ids = torch.stack(self.mbatch["input_ids"], dim=1) if not isinstance(self.mbatch["input_ids"], torch.Tensor) else self.mbatch["input_ids"]
        batched_attention_mask = torch.stack(self.mbatch["attention_mask"], dim=1) if not isinstance(self.mbatch['attention_mask'], torch.Tensor) else self.mbatch['attention_mask']

        self.mbatch["attention_mask"] = batched_attention_mask
        self.mbatch["input_ids"] = batched_input_ids
        print(f"attention_mask.shape: {self.mbatch['attention_mask'].shape}\ninput_ids.shape: {self.mbatch['input_ids'].shape}")
        # print(f"self.mbatch: {self.mbatch}")
        # for k in self.mbatch.keys():
        #     for j in range(len(self.mbatch[k])):
        #         self.mbatch[k][j] = self.mbatch[k][j].to(self.device)

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


# Parameters for synthetic data generation
NUM_DOMAINS = 3
SAMPLES_PER_DOMAIN = 320
MAX_SEQ_LENGTH = 20
LABELS = [0, 1]  # Binary classification (0 = negative, 1 = positive)

# Generate random text
def generate_random_text(max_length):
    words = ["good", "bad", "neutral", "happy", "sad", "amazing", "terrible", "excellent", "horrible"]
    length = random.randint(5, max_length)
    return " ".join(random.choices(words, k=length))

# Generate synthetic dataset
synthetic_data = {"domain": [], "text": [], "label": []}

for domain_id in range(NUM_DOMAINS):
    domain_name = f"domain_{domain_id}"
    for _ in range(SAMPLES_PER_DOMAIN):
        synthetic_data["domain"].append(domain_name)
        synthetic_data["text"].append(generate_random_text(MAX_SEQ_LENGTH))
        synthetic_data["label"].append(random.choice(LABELS))

# Convert to DataFrame
df = pd.DataFrame(synthetic_data)

# Split each domain into train and test
train_datasets = []
test_datasets = []
for domain_name in df["domain"].unique():
    domain_df = df[df["domain"] == domain_name]
    train_size = int(len(domain_df) * 0.8)
    train_df = domain_df[:train_size]
    test_df = domain_df[train_size:]
    train_datasets.append(Dataset.from_pandas(train_df))
    test_datasets.append(Dataset.from_pandas(test_df))

# Combine train datasets for all domains
train_dataset = concatenate_datasets(train_datasets)

# Combine test datasets for all domains
test_dataset = concatenate_datasets(test_datasets)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Check dataset structure
print(dataset)


# Tokenizer and model for sequence classification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Preprocessing function for sequence classification
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], max_length=128, truncation=True, padding=True)
    inputs["labels"] = examples["label"]
    return inputs

print(f"datasets before processing: {dataset}")
# Apply preprocessing
dataset = dataset.map(preprocess_function, batched=True)
print(f"datasets after processing: {dataset}")

# Incremental Domain Setup: Split dataset by domains
domains = set(dataset["train"]["domain"])
train_exps = []
val_exps = []
data_collator = CustomDataCollatorSeq2SeqBeta(tokenizer=tokenizer, model=model)


for task_id, domain in enumerate(domains):
    # Use the dataset for this domain as an experience
    domain_data = dataset.filter(lambda x: x["domain"] == domain)["train"]
    
    # Split into train and validation sets
    domain_train, domain_val = train_test_split(domain_data.to_pandas(), test_size=0.2, stratify=domain_data["labels"], random_state=42)
    # Split the DataFrame into train and validation
    domain_train = domain_train[:train_size].reset_index(drop=True)  # Reset index to avoid adding `__index_level_0__`
    domain_val = domain_val[train_size:].reset_index(drop=True)    # Reset index

    # Select the columns to keep
    columns_to_keep = ["input_ids", "attention_mask", "labels"]

    domain_train = Dataset.from_pandas(domain_train)
    domain_train = domain_train.select_columns(columns_to_keep)

    domain_val = Dataset.from_pandas(domain_val)
    domain_val = domain_val.select_columns(columns_to_keep)

    # Create training experience
    tl_train = DataAttribute(ConstantSequence(task_id, len(domain_train)), "targets_task_labels")
    exp_train = CLExperience(task_id, None)
    exp_train.dataset = AvalancheDataset(
        [domain_train],
        data_attributes=[tl_train],
        collate_fn=data_collator
    )
    train_exps.append(exp_train)

    # # Create validation experience
    # tl_val = DataAttribute(ConstantSequence(task_id, len(domain_val)), "targets_task_labels")
    # val_exp = AvalancheDataset(
    #     [domain_val],
    #     data_attributes=[tl_val],
    #     collate_fn=data_collator
    # )
    # val_exps.append(val_exp)

# Create incremental benchmark scenario
benchmark = CLScenario(
    [
        CLStream("train", train_exps),  # Training stream
        # CLStream("val", val_exps),     # Validation stream
    ]
)

# Evaluation plugin
eval_plugin = EvaluationPlugin(
    loss_metrics(epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()],
    strict_checks=False,
)

# Replay plugin
plugins = [ReplayPlugin(mem_size=200)]

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Strategy for incremental learning
strategy = HGNaive(
    model,
    optimizer,
    torch.nn.CrossEntropyLoss(),
    evaluator=eval_plugin,
    train_epochs=3,
    train_mb_size=32,
    plugins=plugins,
)

# Training and validation
for experience in benchmark.train_stream:
    print(f"Training on experience: {experience.current_experience}")
    strategy.train(experience, collate_fn=data_collator)
    
    # print("Validating...")
    # strategy.eval(benchmark.val_stream)  # Validation after each training experience
