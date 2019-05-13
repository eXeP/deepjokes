import torch.nn as nn

import data


train_data = data.Data()


class WordRNN(nn.Module):
	def __init__(self, ntokens, nemb=256, nhid=512 hidden_size):
		super(WordRNN, self).__init__()
		self.drop = nn.Dropout(0.2)
		self.embedding = nn.Embedding(ntokens, nemb)
		self.rnn = nn.LSTM(nemb, nhid)

	def forward(self, input, hidden)