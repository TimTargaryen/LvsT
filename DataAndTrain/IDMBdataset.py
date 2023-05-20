import torch
from torch.utils.data import Dataset, DataLoader
import os

class IDMBdataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.fileList = [os.path.join(path, "neg", name) for
                         name in os.listdir(os.path.join(path, "neg"))]

        self.fileList.extend([os.path.join(path, "pos", name) for
                         name in os.listdir(os.path.join(path, "pos"))])

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        name = self.fileList[idx]
        label = 1 if int(name.split("_")[-1].split(".")[0]) >= 7 else 0
        comment = ""

        with open(name) as f:
            comment = f.read()
            f.close()

        return comment, label

if __name__ == "__main__":
    myDataset = IDMBdataset("F:\\datasets\\NLP\\aclImdb_v1\\aclImdb\\train")
    myDataLoader = DataLoader(myDataset, batch_size=4, collate_fn=collate4Glove, shuffle=True)

    for idx, (comment, mask, label) in enumerate(myDataLoader):
        print(comment, mask, label)
        break





