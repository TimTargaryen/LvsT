import torch
from torch import nn

class Bert4CLS(nn.Module):
    def __init__(self, dim, kinds, backbone):
        super().__init__()
        self.BertBackbone = backbone
        self.probe = nn.Linear(dim, kinds)
        self.softmax = nn.Softmax(dim=-1)
        self.isfreeze = True

    def freeze(self):
        for name, para in self.BertBackbone.named_parameters():
            para.requires_grad_(False)
        self.isfreeze = True

    def unfreeze(self):
        for name, para in self.BertBackbone.named_parameters():
            para.requires_grad_(True)
        self.isfreeze = False

    def forward(self, seq, mask=None):
        if mask is None:
            mask = torch.LongTensor([[0 for _ in range(len(seq[0]))]])
        x = self.BertBackbone(input_ids=seq, attention_mask=mask)[0][:, 0]
        x = self.probe(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    bert = torch.load("../bert.pth")
    net = Bert4CLS(768, 4, bert)
    print(net(torch.LongTensor([[101, 227, 201]]), torch.LongTensor([[1, 1, 1]])))

