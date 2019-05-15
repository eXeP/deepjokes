from io import open
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class Data(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.OUT_MAX = 128
        self.load_data()
        self.N_TOKENS = len(self.idx2word)

    def load_data(self):
        data = open("shortjokes_noquote.txt")
        lines = 0
        self.idx2word.append('<E>')
        self.word2idx['<E>'] = len(self.idx2word)-1
        for line in data:
            words = ['<SOS>'] + line.split() + ['<EOS>']
            lines = lines + 1
            self.OUT_MAX = max(self.OUT_MAX, len(words))
            for word in words:
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word)-1
        data.close()
        data = open("shortjokes_noquote.txt")
        self.inputs = []
        self.outputs = []
        max_len = 0
        for line in data:
            words = ['<SOS>'] + line.split() + ['<EOS>']
            max_len = max(max_len, len(words))
            ind = 0
            input = torch.zeros([len(words)], dtype=torch.int64)
            output = torch.zeros([len(words)], dtype=torch.int64)
            for word in words:
                input[ind] = self.word2idx[word]
                if ind > 0:
                    output[ind-1] = self.word2idx[word]
                ind = ind+1
            self.inputs.append(input)
            self.outputs.append(output)
        data.close()
        print(len(input), " tensors. Max length ", max_len)

class JokeData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        self.data = Data()
        self.SOS_token = self.data.word2idx['<SOS>']

    def __len__(self):
        return len(self.data.inputs)

    def __getitem__(self, idx):
        sample = (self.data.inputs[idx], self.data.outputs[idx])
        return sample