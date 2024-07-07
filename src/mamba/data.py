import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from tqdm.auto import tqdm


class OpenMathDataset(Dataset):
    def __init__(
        self, data_path, system_prompt_path, tokenizer, max_samples=50_000, max_len=512
    ):
        self.data = self.read_json(data_path, max_samples)
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(system_prompt_path, "r") as fl:
            self.system_prompt = json.load(fl)["prompt"]

    def __getitem__(self, idx):
        return self.tokenize(self.data[idx])

    def __len__(self):
        return len(self.data)

    def read_json(self, data_path, max_samples):
        with open(data_path, "r") as fl:
            data = [json.loads(sample) for sample in fl][:max_samples]
        return data

    def tokenize(self, sample: dict):
        qna = f"{self.system_prompt}\n\nquestion:{sample['question']}\nanswer:{sample['answer']}"
        tokenized_qna = self.tokenizer(
            qna, max_length=self.max_len, truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": tokenized_qna["input_ids"].squeeze().long(),
            "labels": tokenized_qna["input_ids"].squeeze().long(),
        }


def openmath_collate_fn(batch, pad_token_id):
    """Collates the input_ids and labels with the padding token id

    Args:
        batch (dict): Batch of data consisting of input_ids, attention_mask and labels
        pad_token_id (int): ID for the padding token
    """
    # Get input_ids and labels from the batch
    input_ids, labels = tuple(
        [sample[key] for sample in batch] for key in ("input_ids", "input_ids")
    )

    # Pad them
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=-100
    )
    # Attention mask will be True everywhere except at padded positions
    attention_mask = input_ids.ne(pad_token_id)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
