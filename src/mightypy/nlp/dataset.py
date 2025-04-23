# import random
import torch
import os
from pathlib import Path

import requests
from torch.utils.data import Dataset
from tokkit import data_loader


class CustomDataset(Dataset):
    def __init__(self, path, tokenizer, context_length=5, dataset_path = "datasets", device="cpu"):
        self.context_length = context_length
        self.dataset_path = dataset_path
        self.device = device
        self._load(path, tokenizer)
        

    def _save_file_local(self, http_path):
        assert http_path.endswith("txt")
        
        filename = http_path.split("/")[-1]
        directory = os.path.join(os.getcwd(), self.dataset_path)
        filepath = os.path.join(directory, filename)
        dir_p = Path(directory)
        file_p = Path(filepath)
        if not file_p.is_file():
            dir_p.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as file:
                response = requests.get(http_path)
                file.writelines(response.text)
        return filepath

    def _load(self, path, tokenizer):
        if path.startswith("http"):
            path = self._save_file_local(path)
        self.vocab_size = tokenizer.size
        self.raw_data = data_loader(path.encode("utf-8"))
        self.tokens = tokenizer.encode_corpus(self.raw_data)
        self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.tokens) - self.context_length - 1)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokens[idx : idx + self.context_length]).to(self.device),
            torch.tensor([self.tokens[idx + self.context_length]]).to(self.device)
        )


if __name__ == "__main__":
    from tokkit import PyBytePairTokenizer
    from torch.utils.data import DataLoader

    tokenizer = PyBytePairTokenizer()
    url = "https://raw.githubusercontent.com/NishantBaheti/tokkit/refs/heads/main/datasets/raw/combined.txt"
    dataset = CustomDataset(url, tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for X_batch, y_batch in dataloader:
        print(X_batch, y_batch)
        for x, y in zip(X_batch, y_batch):
            print(tokenizer.decode(x), tokenizer.decode(y))
        break

