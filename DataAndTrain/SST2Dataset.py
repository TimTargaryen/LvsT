import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SST2dataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.list = pd.read_csv(file, sep="\t")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list.iloc[idx]['sentence'], self.list.iloc[idx]['label']



