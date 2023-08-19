import os
from pathlib import Path
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader


class Shakespear(Dataset):
    def __init__(self, path: Path, block_size=8):
        self.path = path
        self.block_size = block_size
        self.text = self._get_text()
        self.data = self.encode(self.text)
        self._make_data()

    def _get_text(self):
        with open(self.path) as f:
            text = f.read()
            self.chars = sorted(list(set(text)))
            self.vocab_size = len(self.chars)

            return text

    def _make_data(self) -> None:
        ix = range(len(self.data) - self.block_size)
        self.x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        self.y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])

    def encode(self, string):
        stoi = {ch: i for i, ch in enumerate(self.chars)}

        return torch.tensor([stoi[c] for c in string])

    def decode(self, tokens):
        itos = {i: ch for i, ch in enumerate(self.chars)}

        return ''.join([itos[t] for t in tokens])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ShakespearDataModule(pl.LightningDataModule):
    def __init__(self, data_path, block_size, batch_size):
        super().__init__()
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.cpu_count = os.cpu_count()

    def setup(self, stage: str):
        # Assign datasets for use in dataloaders
        if stage == "fit":
            self.dataset = Shakespear(
                self.data_path,
                self.block_size
            )
            self.vocab_size = self.dataset.vocab_size
            train_size = int(0.7 * len(self.dataset))
            valid_size = len(self.dataset) - train_size

            self.train_dataset, self.valid_dataset = random_split(
                self.dataset,
                [train_size, valid_size]
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.cpu_count)
