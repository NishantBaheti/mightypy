"""
Dataset
--------
"""

import torch

from torch.utils.data import Dataset
from tokkit import data_loader

from mightypy.datautils.download import FileDownloader


class CustomDataset(Dataset):
    def __init__(self, path, tokenizer, context_length=5, dataset_path = "datasets", device="cpu"):
        self.context_length = context_length
        self.dataset_path = dataset_path
        self.device = device
        self.downloader = FileDownloader(self.dataset_path)
        self._load(path, tokenizer)
        
    def _load(self, path, tokenizer):
        if path.startswith("http"):
            path = self.downloader.save_file_local(path)
        self.vocab_size = tokenizer.size
        self.raw_data = data_loader(path)
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
    dataset = CustomDataset(url, tokenizer, context_length=100)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for X_batch, y_batch in dataloader:
        print(X_batch, y_batch)
        for x, y in zip(X_batch, y_batch):
            print(tokenizer.decode(x), tokenizer.decode(y))
        break

