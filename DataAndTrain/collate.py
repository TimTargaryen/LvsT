import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from .gloveTokenizer import GloveTokenizer
import os

bertTokenizer = BertTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.txt"))
gloveTokenizer = GloveTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "glove6b300d.txt"))
glove = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "glove6b300d.pth"))


def collate4Bert(batchData):
    comments, labels = [pair[0] for pair in batchData], [pair[1] for pair in batchData]
    all = bertTokenizer(comments, return_tensors="pt", padding=True, max_length=512, truncation=True)
    labels = torch.LongTensor(labels)

    return all['input_ids'], all['attention_mask'], labels

def collate4Glove(batchData):
    comments, labels = [pair[0] for pair in batchData], [pair[1] for pair in batchData]
    input_ids, masks = gloveTokenizer(comments)
    labels = torch.LongTensor(labels)

    return glove(input_ids), masks, labels

if __name__ == "__main__":
    from SST2Dataset import SST2dataset

    sst2 = SST2dataset("../SST2/train.tsv")
    dsst2 = DataLoader(sst2, batch_size=4, collate_fn=collate4Glove)

    for idx, (seq, mask, label) in enumerate(dsst2):
        print(idx, seq, mask, label)
        break


