from io import open
import torch

class Data(object):
	def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        load_data()

	def load_data():
		data = open("shortjokes_noquote.txt")
		for line in data:
			words = ['<SOS>'] + line.split() + ['<EOS>']
			for word in words:
				if word not in wordtoidx:
					self.idx2word.append(word)
					self.word2idx[word] = len(self.idx2word)-1
		data.close()
		data = open("shortjokes_noquote.txt")
		tensors = []
		for line in data:
			words = ['<SOS>'] + line.split() + ['<EOS>']
			tensor = torch.zeros([len(words)], dtype=torch.int32)
			ind = 0
			for word in words:
				tensor[ind] = self.word2idx[word]
				ind = ind+1
			tensors.append(tensor)
		data.close()