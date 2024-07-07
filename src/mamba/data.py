import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from tqdm.auto import tqdm


class OpenMathDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        pass
