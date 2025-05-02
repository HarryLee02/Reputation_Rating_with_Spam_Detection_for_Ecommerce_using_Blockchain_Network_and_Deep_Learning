import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk.tokenize import RegexpTokenizer
# vocab lay tu https://github.com/undertheseanlp/dictionary
# https://github.com/duyet/vietnamese-wordlist

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return torch.sigmoid(output)



def encode_sentence(s, vocab):
    return [vocab.get(i, 0) for i in s.lower().split()]

dataset_path='vispamdetection_dataset'

data = pd.read_csv(dataset_path+'/reviews.csv/reviews.csv')
# vocab = pd.read_json('vocab/vocab.json', orient='index')
vocab = pd.read_json('test_vocab.json', orient='index')

test_dataset = data
vocab_dict = vocab.to_dict()[0]


tokenizer = RegexpTokenizer(r'\w+')

tokens = tokenizer.tokenize(test_dataset['comment'][0])
print("Tokens:", tokens)

