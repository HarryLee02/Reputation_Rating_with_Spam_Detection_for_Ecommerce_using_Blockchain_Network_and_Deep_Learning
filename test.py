import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

import numpy as np

class CustomTrainableEmbedding:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))

    def forward(self, word_indices):
        """
        Retrieve embeddings by indexing into the embedding matrix.
        :param word_indices: List of word indices.
        :return: Corresponding word embeddings.
        """
        embeddings = self.embedding_matrix[word_indices]
        return embeddings
    
    def update(self, word_indices, gradient, learning_rate=0.01):
        """
        Updates embeddings for given word indices using gradient descent.
        :param word_indices: Indices of words whose embeddings are being updated.
        :param gradient: Gradient matrix computed from backpropagation.
        :param learning_rate: Learning rate for updates.
        """
        self.embedding_matrix[word_indices] -= learning_rate * gradient

class CustomWordEmbedding:
    def __init__(self, vocab, embed_dim):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        # Random initialization of embeddings
        self.embedding_matrix = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embed_dim))
    
    def lookup(self, word):
        """
        Retrieves the embedding vector for a given word using direct lookup.
        :param word: Word to look up.
        :return: Embedding vector.
        """
        if word in self.vocab:
            word_index = self.vocab[word]
            return self.embedding_matrix[word_index]
        else:
            return np.zeros(self.embed_dim)  # Return a zero vector for unknown words
    
    def get_embedding_matrix_mult(self, one_hot_vector):
        return np.dot(one_hot_vector, self.embedding_matrix)

    def visualize_embeddings(self, num_words=50):
        reduced_matrix = self.embedding_matrix[:num_words]
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(reduced_matrix)
        
        plt.figure(figsize=(8, 8))
        for i, word in enumerate(list(self.vocab.keys())[:num_words]):
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
            plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], word, fontsize=8)
        plt.title("Word Embedding Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.show()

# Example Usage:
if __name__ == "__main__":

    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_layers = 2
    output_dim = 1
    embedding_dim = 64
    hidden_dim = 100

    vocab = {"cat": 0, "dog": 1, "fish": 2, "apple": 3}  # Sample vocabulary
    embed_dim = 4

    embedding = CustomWordEmbedding(vocab, embed_dim)

    print("Embedding for 'dog':", embedding.lookup("dog"))

    one_hot = np.array([0, 1, 0, 0])  # One-hot encoding for "dog"
    print("Embedding for 'dog' via matrix multiplication:", embedding.get_embedding_matrix_mult(one_hot))

    embedding.visualize_embeddings(num_words=4)
