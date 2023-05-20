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

config, beginTime = None, None


def line(c):
    for i in range(50):
        print(c, end="")
    print()


def getRate(model, LR, decay):
    rate = [{'params': model.BertBackbone.encoder.layer[i].parameters(),
             'lr': LR * (decay ** (11 - i))} for i in range(12)]
    rate.append({'params': model.BertBackbone.embeddings.parameters(), 'lr': 0})
    return rate


def FineTune(model, trainLoader, testLoader, epochs, LR):
    writer = SummaryWriter(os.path.join(config['saveDir'], beginTime))
    citerition = nn.CrossEntropyLoss()

    interval = config['Interval']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    show = 1

    for epoch in range(epochs):

        rate = getRate(model, LR, 0.95)
        optimizer = torch.optim.AdamW(rate, lr=LR)

        cnt = 0
        Loss = 0.0
        correct = 0.0

        model.train()
        model.unfreeze()
        for i, (seq, pad, label) in enumerate(trainLoader):
            cnt += 1

            seq = seq.to(device)
            pad = pad.to(device)
            label = label.to(device)

            predict = model(seq=seq, mask=pad)

            loss = citerition(predict, label)
            loss = loss.mean()
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

                    predict = model(seq, pad)
                    correct += (torch.sum(torch.argmax(predict, dim=-1) == label) / len(label)).item()



                    if cnt % interval == 0 and cnt != 0:
                        line("-")
                        print("epoch:{}/{}, step:{}/{}, AvgCorrectRate:{}"
                              .format(epoch + 1, epochs, cnt,
                                      trainLoader.__len__(), correct / cnt))

                        writer.add_scalar('AvgCorrectRate', correct / cnt, epoch * trainLoader.__len__() + cnt)


                line("*")
                line("*")
                print("epoch:{}/{}, testCorrectRate:{}".format(epoch + 1, epochs, correct / cnt))
                line("*")
                line("*")

            writer.add_scalar('testCorrectRate', correct / cnt, epoch + 1)
            torch.save(model, os.path.join(config['saveDir'], beginTime, str(epoch + 1) + ".pth"))

if __name__ == "__main__":
    from model import Bert4CLS
    from collate import collate4Bert
    from IDMBdataset import IDMBdataset
    from SST2Dataset import SST2dataset
    from torch.utils.data import DataLoader

    config = yaml.load(open("../config.yaml"), yaml.FullLoader)
    beginTime = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    bert = Bert4CLS(768, 2, torch.load("../bert.pth"))
    IDMBtrain = DataLoader(IDMBdataset("F:\\datasets\\NLP\\aclImdb_v1\\aclImdb\\train"), shuffle=True, batch_size=2, collate_fn=collate4Bert)
    IDMBtest = DataLoader(IDMBdataset("F:\\datasets\\NLP\\aclImdb_v1\\aclImdb\\test"), shuffle=True, batch_size=2, collate_fn=collate4Bert)
    FineTune(bert, IDMBtrain, IDMBtest, 5, 1e-5)

















