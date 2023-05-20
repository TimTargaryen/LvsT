import torch
from torch import nn

import sys
sys.path.append("..")

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
import time
import datetime
import os

from .Distill import DistillLoss, DistillDataset, collate4Distill


config, beginTime = None, None


def line(c):
    for i in range(50):
        print(c, end="")
    print()


def distill(model, trainLoader, testLoader, epochs, LR):
    writer = SummaryWriter(os.path.join(config['saveDir'], beginTime))
    citerition = DistillLoss()

    interval = config['Interval']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    show = 1

    for epoch in range(epochs):

        cnt = 0
        Loss = 0.0
        correct = 0.0
        rate = (10 - epoch) / 10
        t = (10 - epoch) // 4

        model.train()
        for i, ((seq, pad, label), slabel) in enumerate(trainLoader):
            cnt += 1

            seq = seq.to(device)
            pad = pad.to(device)
            label = label.to(device)
            slabel = slabel.to(device)

            predict = model(seq=seq, mask=pad)[0]

            loss = citerition(predict, label, slabel, alpha=rate, T=t)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            Loss += loss.item()

            correct += (torch.sum(torch.argmax(predict, dim=-1) == label) / len(label)).item()

            if loss.item() > 20000 or Loss != Loss:
                print(loss)
                print("sth wrong")
                exit(1)

            if cnt % interval == 0 and cnt != 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print("epoch:{}/{}, step:{}/{}, avgloss:{}, correctRate:{}"
                      .format(epoch + 1, epochs, cnt,
                              trainLoader.__len__(), Loss / interval, correct / interval, lr))

                writer.add_scalar('avgloss', Loss / interval, epoch * trainLoader.__len__() + cnt)
                writer.add_scalar('correctRate', correct / interval, epoch * trainLoader.__len__() + cnt)
                Loss = 0
                correct = 0.0
                break

        if (epoch + 1) % show == 0 and epoch > 0:
            cnt = 0

            with torch.no_grad():
                for i, (seq, pad, label) in enumerate(testLoader):
                    cnt += 1

                    seq = seq.to(device)
                    pad = pad.to(device)
                    label = label.to(device)

                    predict = model(seq, pad)[0]
                    correct += (torch.sum(torch.argmax(predict, dim=-1) == label) / len(label)).item()

                    if cnt % interval == 0 and cnt != 0:
                        line("-")
                        print("epoch:{}/{}, step:{}/{}, AvgCorrectRate:{}"
                              .format(epoch + 1, epochs, cnt,
                                      trainLoader.__len__(), correct / cnt))

                        writer.add_scalar('AvgCorrectRate', correct / cnt, epoch * trainLoader.__len__() + cnt)
                        break

                line("*")
                line("*")
                print("epoch:{}/{}, testCorrectRate:{}".format(epoch + 1, epochs, correct / cnt))
                line("*")
                line("*")

            writer.add_scalar('testCorrectRate', correct / cnt, epoch + 1)
            torch.save(model, os.path.join(config['saveDir'], beginTime, str(epoch + 1) + ".pth"))

if __name__ == "__main__":
    from DataAndTrain.SST2Dataset import SST2dataset
    from collate import collate4Glove

    config = yaml.load(open("../config.yaml"), yaml.FullLoader)
    beginTime = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    sst2 = SST2dataset("../SST2/train.tsv")
    sst2d = DistillDataset(sst2, "sst2train.npy")
    sst2val = SST2dataset("../SST2/dev.tsv")
    sst2d = DataLoader(sst2d, batch_size=4, collate_fn=collate4Distill)
    sst2val = DataLoader(sst2val, batch_size=4, collate_fn=collate4Glove)

    from model import LSTM4CLS

    lstm = LSTM4CLS(3, 2)

    distill(lstm, sst2d, sst2val, 2, 1e-5)


