import torch
from nltk.tokenize import RegexpTokenizer
from wordfreq import word_frequency, zipf_frequency
import json
import re
import pandas
import numpy
from tqdm import tqdm

no_layers = 2
output_dim = 1
embedding_dim = 64
hidden_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTM(torch.nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(LSTM,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
   
        # embedding and LSTM layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
       
        #lstm
        self.lstm = torch.nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim, num_layers=no_layers, batch_first=True)
       
        # dropout layer
        self.dropout = torch.nn.Dropout(0.3)
   
        # linear and sigmoid layer
        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)
        self.sig = torch.nn.Sigmoid()
       
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)
       
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
       
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
       
        # sigmoid function
        sig_out = self.sig(out)
       
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)


        sig_out = sig_out[:, -1]
        return sig_out, hidden
       
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden 
    
model = torch.load('full_model.pth',weights_only=False)
model.eval()

def padding_n_truncate(seq, max_len):

    if len(seq) > max_len:
        return seq[:max_len]  # Truncate
    else:
        return seq + [0] * (max_len - len(seq))  # Pad with zeros

# vispamdetection_dataset/reviews.csv/reviews.csv
def load_dataset(vocab_path,type):
    if type == "csv":
        # return a pandas DataFrame
        dataset = pandas.read_csv(vocab_path)
        return dataset
    else:
        print("Not yet implemented")
        return None

def load_vocab(vocab_path):
    vocab = pandas.read_json(vocab_path,orient='index')
    vocab.columns=['freq','value']
    return vocab
    # vocab is a DataFrame with 3 columns: 'index', 'freq', 'value'
    # to call them, use vocab.index, .freq, .value
    # use vocab.loc(someword,'freq') to get the freq of the word, same for 'index'

def encode_sentence(tokens, vocab):
    # token is a list of words [] from ntlk.word_tokenize(sen)
    for i in range(len(tokens)):
        if tokens[i] in vocab.index:
            tokens[i] = vocab.loc[tokens[i],'value']
        else:
            tokens[i] = 0
    return tokens

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

text = "Gói hàng đẹp , mà giao lâu quá à tận 10 mấy ngày Cái bao bì xinh vl mà tui hem chụp ảnh , vẽ nét mảnh khá đẹp mà mới vẽ mấy lần có dấu hịu nhòe ngòi r chắc tại mạnh tay quá Đơn sau dc giảm 15% hihi"
vocab = load_vocab("vocab/vocab_with_freq.json")
tokenizer = RegexpTokenizer(r'\w+')
token = tokenizer.tokenize(text)
encode_token = encode_sentence(token, vocab)
encode_token = padding_n_truncate(encode_token, 100)
encode_token = torch.tensor(encode_token)

with torch.no_grad():
    output = model(encode_token)
    prediction = torch.argmax(output, dim=1)
    print(f"Predicted Class: {prediction.item()}")