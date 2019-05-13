from io import open
import torch

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
            words = line.split() + ['<EOS>']
            lines = lines + 1
            self.OUT_MAX = max(self.OUT_MAX, len(words))
            for word in words:
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word)-1
        data.close()
        data = open("shortjokes_noquote.txt")
        tensors = []
        max_len = 0
        for line in data:
            words = line.split() + ['<EOS>']
            max_len = max(max_len, len(words))
            ind = 0
            tensor = torch.zeros([self.OUT_MAX], dtype=torch.int32)
            for word in words:
                tensor[ind] = self.word2idx[word]
                ind = ind+1
            tensors.append(tensor)
        data.close()
        self.tensors = torch.stack(tensors)
        print(len(tensors), " tensors. Max length ", max_len)
        print(self.tensors.shape)