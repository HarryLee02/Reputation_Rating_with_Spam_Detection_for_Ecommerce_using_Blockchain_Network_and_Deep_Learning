import torch
from nltk.tokenize import RegexpTokenizer
from wordfreq import word_frequency, zipf_frequency
import json
import re
import pandas
import numpy
from tqdm import tqdm


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

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
       
        self.dropout = torch.nn.Dropout(0.3)
   
        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)
        self.sig = torch.nn.Sigmoid()
       
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)
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

no_layers = 2
output_dim = 1
embedding_dim = 64
hidden_dim = 100

vocab = load_vocab("vocab/vocab_with_freq.json")
dataset = load_dataset("vispamdetection_dataset/reviews.csv/reviews.csv","csv")
# take 2000 samples
dataset = dataset.sample(n=2000, random_state=42).reset_index(drop=True)
vocab_size = vocab.size

model = LSTM(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)


#moving to gpu
model.to(device)
print(model)

tokenizer = RegexpTokenizer(r'\w+')

# Tokenize and filter out empty sentences
tokens_list = [tokenizer.tokenize(dataset['comment'][i]) for i in range(len(dataset))]
tokens_list = [tokens for tokens in tokens_list if len(tokens) > 0]  # Remove empty token lists

print("Tokens size after filtering:", len(tokens_list))

# Encode sentences and filter out empty encodings
encode_tokens_list = [encode_sentence(tokens, vocab) for tokens in tokens_list]
# tính max len lại
encode_tokens_list = [padding_n_truncate(tokens, 100) for tokens in encode_tokens_list]  # Pad or truncate to max length
encode_tokens_list = [torch.tensor(tokens, dtype=torch.long) for tokens in encode_tokens_list if len(tokens) > 0]  # Remove empty tensors
print("Encoded tokens size after filtering:", len(encode_tokens_list))

labels_tensor = torch.tensor(dataset['label'].values,dtype=torch.long) # tensor for each sentence


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

clip = 5
epochs = 15
valid_loss_min = numpy.inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

batch_size = 1

for epoch in tqdm(range(epochs)):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state
    h = model.init_hidden(batch_size)
    for inputs, labels in zip(encode_tokens_list, labels_tensor):
        inputs = inputs.unsqueeze(0) # add batch dimension and move to GPU
        inputs, labels = inputs.to(device), labels.to(device)  
        
        h = tuple([each.data for each in h])
       
        model.zero_grad()
        output,h = model(inputs,h)
       
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = acc(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
       
    
    model.eval()
    
           
    epoch_train_loss = numpy.mean(train_losses)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_train_acc = train_acc/len(encode_tokens_list)
    print(f'Epoch {epoch+1}')
    print(f'train_loss : {epoch_train_loss}')
    print(f'train_accuracy : {epoch_train_acc}')
    # if epoch_val_loss <= valid_loss_min:
    #     torch.save(model.state_dict(), 'state_dict.pt')
    #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
    #     valid_loss_min = epoch_val_loss
    print(25*'==')

torch.save(model.state_dict(), 'state_dict.pth')
    

           
    # epoch_train_loss = numpy.mean(train_losses)
    # epoch_val_loss = numpy.mean(val_losses)
    # epoch_train_acc = train_acc/len(train_loader.dataset)
    # epoch_val_acc = val_acc/len(valid_loader.dataset)
    # epoch_tr_loss.append(epoch_train_loss)
    # epoch_vl_loss.append(epoch_val_loss)
    # epoch_tr_acc.append(epoch_train_acc)
    # epoch_vl_acc.append(epoch_val_acc)
    # print(f'Epoch {epoch+1}')
    # print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    # print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    # if epoch_val_loss <= valid_loss_min:
    #     torch.save(model.state_dict(), 'state_dict.pt')
    #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
    #     valid_loss_min = epoch_val_loss
    # print(25*'==')