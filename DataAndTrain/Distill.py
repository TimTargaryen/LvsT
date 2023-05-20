from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from .collate import collate4Glove
import numpy as np

def Bert2mid(bert, loader, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert = bert.to(device)
    logits = []

    with torch.no_grad():
        for i, (seq, pad, _) in enumerate(loader):
            seq.to(device)
            pad.to(device)

            pred = bert.BertBackbone(input_ids=seq, attention_mask=pad)[0][:, 0]
            pred = bert.probe(pred)
            pred = pred.squeeze(0).detach().numpy()
            logits.append(pred)
            print("{}th is over, logits is:".format(i), pred)
            if i > 300:
                break

    logits = np.array(logits)
    np.save(name, logits)


class DistillDataset(Dataset):
    def __init__(self, formal, name):
        super().__init__()
        self.formal = formal
        self.logits = np.load(name)

    def __len__(self):
        return self.formal.__len__()

    def __getitem__(self, idx):
        return self.formal[idx], self.logits[idx]


def collate4Distill(batchData):
    formalData = [data[0] for data in batchData]
    dgts = [torch.from_numpy(data[1]) for data in batchData]

    return collate4Glove(formalData) , torch.stack(dgts)


class DistillLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft = nn.Softmax(dim=-1)
        self.mse = nn.MSELoss()
        self.xe = nn.CrossEntropyLoss()

    def forward(self, pred, gt, dgt, alpha=0.5, T = 1):
        dgt = self.soft(dgt / T)
        distill = self.mse(pred, dgt)
        std = self.xe(pred, gt)

        return alpha * distill + (1 - alpha) * std

'''
if __name__ == "__main__":
    pred = torch.tensor([[0.1, 0.9], [0.2, 0.8]], requires_grad=True)
    dgt = torch.tensor([[0.05, 0.95], [0.04, 0.96]])
    gt = torch.LongTensor([1, 0])
    cirterion = DistillLoss()
    cirterion2 = nn.CrossEntropyLoss()

    loss = cirterion(pred, gt, dgt, alpha=1)
    loss2 = cirterion2(pred, gt)
    loss2.backward()
    loss.backward()
    print(loss, loss2)
'''

'''
if __name__ == "__main__":
    from DataAndTrain.SST2Dataset import SST2dataset
    from DataAndTrain.collate import collate4Bert
    from torch.utils.data import DataLoader
    from model.Bert4CLS import Bert4CLS

    sst2 = SST2dataset("../SST2/train.tsv")
    sst2 = DataLoader(sst2, batch_size=1, collate_fn=collate4Bert)
    bert = Bert4CLS(768, 2, torch.load("../bert.pth"))

    Bert2mid(bert, sst2, "sst2train.npz")
'''

if __name__ == "__main__":
    from DataAndTrain.SST2Dataset import SST2dataset

    sst2 = SST2dataset("../SST2/train.tsv")
    sst2d = DistillDataset(sst2, "sst2train.npy")
    sst2d = DataLoader(sst2d, batch_size=4, collate_fn=collate4Distill)

    for i, ((seq, pad, label), slabel) in enumerate(sst2d):
        print(i, seq, pad, label, slabel)
        break




