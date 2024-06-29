import json
from typing import Dict, Sequence
import torch.utils
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm


class MambaChatDataset(Dataset):
    def __init__(self, json_path, conversation_template, tokenizer, max_length=512):
        self.data = self._read_json(json_path)
        self.conversation_template = conversation_template
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-process and get the tokenized chats (input_ids)
        self.data = self._apply_chat_template(self.data)

    def __getitem__(self, idx) -> dict:
        return dict(input_ids=self.data[idx], labels=self.data[idx])

    def __len__(self):
        return len(self.data)

    def _read_json(self, json_path: str) -> list:
        with open(json_path, "r") as f:
            data = [json.loads(line) for line in f][:1000]
        return data

    def _convert_chat(self, conversation: list) -> torch.Tensor:
        """Converts a given conversation to chat template form"""
        tokenized_conv = self.tokenizer.apply_chat_template(
            conversation,
            chat_template=self.conversation_template,
            max_length=self.max_length,
            truncation=True,
        )
        return torch.tensor(tokenized_conv, dtype=torch.long)

    def _apply_chat_template(self, data: list) -> dict:
        """Applies the chat template to all the conversations in the dataset"""
        input_ids = [self._convert_chat(conv["messages"]) for conv in tqdm(data)]
        return input_ids


def mamba_chat_collate_fn(batch: Sequence[Dict], pad_token_id: int) -> dict:
    """Collate function that pads input_ids and labels

    Args:
        batch (Sequence[Dict]): A batch of dictionaries with each having input_ids and labels

    Returns:
        dict: A dictionary with padded input_ids, labels and attention_mask
    """
    # Get input_ids and labels from the batch
    input_ids, labels = tuple(
        [sample[key] for sample in batch] for key in ["input_ids", "input_ids"]
    )

    # Pad them
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    # Attention mask will be True everywhere except at padded positions
    attention_mask = input_ids.ne(pad_token_id)

    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
