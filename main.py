import torch
from torch import nn
from torchtext.vocab import GloVe

if __name__ == "__main__":
    token = torch.rand(1, 5, 10)
    net = nn.LayerNorm(10)
    print(token, net(token))

