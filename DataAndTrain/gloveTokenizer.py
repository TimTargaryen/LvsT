import torch
from torch import nn
from nltk import word_tokenize



class GloveTokenizer(nn.Module):
    def __init__(self, gloveList):
        super().__init__()
        self.list = {}

        with open(gloveList, "r+", encoding='utf-8') as f:
            for i in range(400000):
                self.list[f.readline().strip("\n")] = i
            f.close()

    def forward(self, sentences):
        input_ids, masks = [], []

        for sentence in sentences:
            input_id = []
            words = word_tokenize(sentence)

            for i in range(len(words)):
                try:
                    input_id.append(self.list[words[i]])
                except:
                    input_id.append(self.list['_'])

            input_ids.append(input_id)
            masks.append([0 for _ in range(len(input_id))])

        maxLen = max(map(lambda x: len(x), input_ids))
        [mask.extend([1 for _ in range(maxLen - len(input_id))]) for input_id, mask in zip(input_ids, masks)]
        [input_id.extend([self.list['_'] for _ in range(maxLen - len(input_id))]) for input_id in input_ids]

        return torch.LongTensor(input_ids), torch.LongTensor(masks)

if __name__ == "__main__":
    gloveTokenizer = GloveTokenizer("glove6b300d.txt")
    print(gloveTokenizer(["yes you mother fucker", "no shit!"]))





