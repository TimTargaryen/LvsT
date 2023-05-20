import torch
from torch import nn

class Trm4CLS(nn.Module):
    def __init__(self, dim, kind, layers=3, max_length=800):
        super().__init__()
        mask = torch.empty(max_length, max_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        self.mask = mask

        self.proj = nn.Parameter(torch.empty(300, dim))
        self.proj2 = nn.Parameter(torch.empty(dim, 300))
        self.posEmbed = nn.Parameter(torch.empty(max_length, dim))
        self.TrmBlocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, nhead=8, batch_first=True) for _ in range(layers)])
        self.linear = nn.Linear(300, kind)
        self.softmax = nn.Softmax(dim=-1)
        self.GELU = nn.GELU()

    def forward(self, x, padMask):
        x = x @ self.proj
        x = x + self.posEmbed[:x.size(1)]
        srcMask = self.mask[:x.size(1), :x.size(1)].to(x.device)

        pos = torch.LongTensor([len(padMask[0]) for _ in range(len(padMask))]) - torch.sum(padMask, dim=-1) - 1
        pos = pos.to(torch.int64)

        for i in range(len(self.TrmBlocks)):
            x = self.TrmBlocks[i](x, src_mask=srcMask, src_key_padding_mask=padMask)

        hidden = x @ self.proj2
        y = self.linear(hidden[torch.arange(hidden.shape[0]), pos])
        y = self.GELU(y)
        y = self.softmax(y)

        return y, hidden, padMask

if __name__ == "__main__":
    net = Trm4CLS(768, 4)
    print(net(torch.rand(2, 3, 300), torch.LongTensor([[0, 0, 0], [0, 0, 1]])))




