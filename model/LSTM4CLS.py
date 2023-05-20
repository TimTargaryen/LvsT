import torch
from torch import nn

class LSTM4CLS(nn.Module):
    def __init__(self, layers, kinds):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=300, hidden_size=300, batch_first=True, num_layers=layers)
        self.MLP = nn.Linear(300, kinds)
        self.GLEU = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq, mask):
        pos = torch.LongTensor([len(mask[0]) for _ in range(len(mask))]) - torch.sum(mask, dim=-1) - 1
        pos = pos.to(torch.int64)

        out = self.LSTM(seq)
        y = self.MLP(out[0][torch.arange(seq.shape[0]), pos])
        y = self.GLEU(y)
        y = self.softmax(y)
        hidden = out[0]

        return y, hidden, mask

if __name__ == "__main__":
    lstm = LSTM4CLS(2, 4)
    print(lstm(torch.rand(2, 5, 300), torch.LongTensor([[0, 0, 0, 0, 0], [0, 0, 0, 1, 1]])))
