import numpy as np
import torch as T
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import pandas
device = T.device('cpu')

# -----------------------------------------------------------

class MyEmbedding(T.nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super(MyEmbedding, self).__init__()
    self.weight = T.nn.Parameter(T.zeros((vocab_size, embed_dim), dtype=T.float32))
    T.nn.init.uniform_(self.weight, -0.1, +0.1)
    # T.nn.init.normal_(self.weight)  # mean = 0, stddev = 1

  def forward(self, x):
    return self.weight[x]

# -----------------------------------------------------------

class LSTM_Net(T.nn.Module):
  def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
    """
    Initialize LSTM network
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of word embeddings (default: 128)
        hidden_dim: Dimension of LSTM hidden state (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.3)
    """
    super(LSTM_Net, self).__init__()
    # Embedding layer:
    self.embed = MyEmbedding(vocab_size, embed_dim)
    
    # LSTM layer with multiple layers
    self.lstm = T.nn.LSTM(
        input_size=embed_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0,  # dropout only between LSTM layers
        bidirectional=False  # using unidirectional for simplicity
    )
    
    # Additional dropout after LSTM
    self.do1 = T.nn.Dropout(dropout)
    
    # Final classification layer
    self.fc1 = T.nn.Linear(hidden_dim, 1)
 
  def forward(self, x):
    # x shape: [batch_size, seq_len]
    
    # Get word embeddings
    z = self.embed(x)  # [batch_size, seq_len, embed_dim]
    
    # lstm_out shape: [batch_size, seq_len, hidden_dim]
    lstm_out, (h_n, c_n) = self.lstm(z)
    
    # Take the last output of the sequence
    z = lstm_out[:,-1]  # [batch_size, hidden_dim]
    
    # Apply dropout
    z = self.do1(z)
    
    # Final classification
    z = T.sigmoid(self.fc1(z))
    return z

# -----------------------------------------------------------

class Vietnamese_Dataset(T.utils.data.Dataset):
  def __init__(self, src_file, vocab, max_len=100):
    # CSV file with 'comment' and 'label' columns
    df = pd.read_csv(src_file)
    self.comments = df['comment'].values
    self.labels = df['label'].values
    self.vocab = vocab
    self.max_len = max_len
    self.tokenizer = RegexpTokenizer(r'\w+')
    
    x_data = []
    y_data = []
    
    for comment, label in zip(self.comments, self.labels):
      # Tokenize 
      tokens = self.tokenizer.tokenize(comment)
      
      # tokens to IDs
      token_ids = []
      for token in tokens:
        if token in self.vocab.index:
          token_ids.append(self.vocab.loc[token, 'value'])
        else:
          token_ids.append(0)  # Unknown token
      
      # Pad or trunc
      if len(token_ids) < max_len:
        token_ids = [0] * (max_len - len(token_ids)) + token_ids
      else:
        token_ids = token_ids[-max_len:]
        
      x_data.append(token_ids)
      y_data.append(label)
    
    # arrays -> tensors
    self.x_data = T.tensor(np.array(x_data), dtype=T.int64).to(device)
    self.y_data = T.tensor(np.array(y_data), dtype=T.float32).to(device)
    self.y_data = self.y_data.reshape(-1, 1)  # float32 2D

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    tokens = self.x_data[idx]
    trgts = self.y_data[idx] 
    return (tokens, trgts)

# -----------------------------------------------------------

def load_vocab(vocab_path):
    vocab = pd.read_json(vocab_path,orient='index')
    vocab.columns=['freq','value']
    return vocab

def accuracy(model, dataset):
  num_correct = 0; num_wrong = 0
  ldr = T.utils.data.DataLoader(dataset,
    batch_size=1, shuffle=False)
  for (batch_idx, batch) in enumerate(ldr):
    X = batch[0]  # inputs
    Y = batch[1]  # target sentiment label 0 or 1

    with T.no_grad():
      oupt = model(X)  # single [0.0, 1.0]
    if oupt < 0.5 and Y == 0:
      num_correct += 1
    elif oupt > 0.5 and Y == 1:
      num_correct += 1
    else:
      num_wrong += 1
    
  acc = (num_correct * 100.0) / (num_correct + num_wrong)
  return acc

# -----------------------------------------------------------

def save_model(model, vocab_size, save_path):
    """Save model and its configuration"""
    model_info = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embed_dim': model.embed.weight.shape[1],
        'hidden_dim': model.lstm.hidden_size
    }
    T.save(model_info, save_path)
    print(f"Model saved to {save_path}")

def load_model(model_path, device):
    """Load model and its configuration"""
    model_info = T.load(model_path, map_location=device)
    model = LSTM_Net(
        vocab_size=model_info['vocab_size'],
        embed_dim=model_info['embed_dim'],
        hidden_dim=model_info['hidden_dim']
    ).to(device)
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    return model

def main():
  print("\nBegin PyTorch Vietnamese LSTM demo ")

  # 1. load dataset, tokenize + pad in Dataset class
  print("\nLoading preprocessed train and test data ")
  train_file = "vispamdetection_dataset\\dataset\\train.csv"
  test_file = "vispamdetection_dataset\\dataset\\test.csv"
  
  # 2. load vocab
  vocab = load_vocab("vocab/vocab_with_freq.json")
  vocab_size = len(vocab)
  print(f"Vocabulary size: {vocab_size}")
  
  train_ds = Vietnamese_Dataset(train_file, vocab) 
  test_ds = Vietnamese_Dataset(test_file, vocab)

  bat_size = 32  # Increased batch size
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True, drop_last=False)
  n_train = len(train_ds)
  n_test = len(test_ds)
  print("Num train = %d Num test = %d " % (n_train, n_test))

# -----------------------------------------------------------

  # 3. lstm
  print("\nCreating Multi-layer LSTM binary classifier ")
  # hyperparameters
  embed_dim = 128  
  hidden_dim = 128
  num_layers = 2
  dropout = 0.3

  net = LSTM_Net(
      vocab_size=vocab_size,
      embed_dim=embed_dim,
      hidden_dim=hidden_dim,
      num_layers=num_layers,
      dropout=dropout
  ).to(device)

  # 4. train model
  loss_func = T.nn.BCELoss()
  lrn_rate = 0.001
  optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate, weight_decay=1e-5)  # Added weight decay
  max_epochs = 30

  print("\nModel Configuration:")
  print(f"Number of LSTM layers: {num_layers}")
  print(f"Embedding dimension: {embed_dim}")
  print(f"Hidden dimension: {hidden_dim}")
  print(f"Dropout rate: {dropout}")
  print(f"Batch size: {bat_size}")
  print(f"Learning rate: {lrn_rate}")
  print(f"Max epochs: {max_epochs}")

  print("\nStarting training ")

  net.train()
  for epoch in range(0, max_epochs):
    tot_err = 0.0  # for one epoch
    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch[0]  # [bs,seq_len]
      Y = batch[1]
      optimizer.zero_grad()
      oupt = net(X)
      loss_val = loss_func(oupt, Y) 
      tot_err += loss_val.item()
      loss_val.backward()  # compute gradients
      
      # Gradient clipping
      T.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
      
      optimizer.step()     # update weights
  
    print("epoch = %4d  |" % epoch, end="")
    print("   loss = %10.4f  |" % tot_err, end="")
    net.eval()
    train_acc = accuracy(net, train_ds)
    print("  acc = %8.2f%%" % train_acc)
    net.train()

  print("Training complete")

# -----------------------------------------------------------

  # 5. evaluate
  net.eval()
  test_acc = accuracy(net, test_ds)
  print("\nAccuracy on test data = %8.2f%%" % test_acc)

  # 6. save model
  print("\nSaving trained model state")
  model_save_path = "models/vietnamese_multilayer_lstm_model.pt"
  save_model(net, vocab_size, model_save_path)

  
  print("\nModel saved! End PyTorch Vietnamese LSTM sentiment demo")

if __name__ == "__main__":
  main()