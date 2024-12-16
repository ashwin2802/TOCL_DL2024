import torch
import avalanche
from typing import Optional, Union, Any
from transformers.utils import PaddingStrategy
from dataclasses import dataclass

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

        batched_input_ids = batched_input_ids.to(self.model.device)
        batched_attention_mask = batched_attention_mask.to(self.model.device)
        self.mbatch["labels"] = self.mbatch["labels"].to(self.model.device)

        # print(f"self.mbatch.keys(): {self.mbatch.keys()}")
        self.mbatch["attention_mask"] = batched_attention_mask
        self.mbatch["input_ids"] = batched_input_ids

    def forward(self):
        out = self.model(self.mb_x)
        return out
        
    def criterion(self):
        mb_output = self.mb_output.view(-1, self.mb_output.size(-1))
        ll = self._criterion(mb_output, self.mb_y.view(-1))
        return ll
    
class HGICaRL(avalanche.training.ICaRL):
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

        device = self.model.feature_extractor.device
        batched_input_ids = batched_input_ids.to(device)
        batched_attention_mask = batched_attention_mask.to(device)
        self.mbatch["labels"] = self.mbatch["labels"].to(device)

        # print(f"self.mbatch.keys(): {self.mbatch.keys()}")
        self.mbatch["attention_mask"] = batched_attention_mask
        self.mbatch["input_ids"] = batched_input_ids

    def forward(self):
        # print(f"self.mb_x: {self.mb_x}")
        input = {'input_ids': self.mb_x, 'attention_mask': self.mb_attention_mask}
        # out = self.model(
        #     input_ids=self.mb_x,
        #     attention_mask=self.mb_attention_mask,
        #     labels=self.mb_y,
        # )
        out = self.model(input)
        return out

    def criterion(self):
        mb_output = self.mb_output.view(-1, self.mb_output.size(-1))
        ll = self._criterion(mb_output, self.mb_y.view(-1))
        return ll