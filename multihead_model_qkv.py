import numpy as np
import torch as T
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import emoji
from wordfreq import zipf_frequency
from sklearn.metrics import confusion_matrix

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if hasattr(T, 'set_float32_matmul_precision'):
    T.set_float32_matmul_precision('high')

# -----------------------------------------------------------
class CustomTokenizer:
    def __init__(self):
        # Word tokenizer
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        
        # Number pattern
        self.number_pattern = re.compile(r'\d+(?:\.\d+)?')
    
    def is_emoji(self, char):

        return emoji.is_emoji(char)
    
    def tokenize(self, text):
        # Convert text to lowercase
        text = text.lower()
        
        tokens = []
        current_word = []
        i = 0
        while i < len(text):
            char = text[i]
            
            if self.is_emoji(char):
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                tokens.append(char)

            elif char.isdigit():
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                num = []
                while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                    num.append(text[i])
                    i += 1
                tokens.append(''.join(num))
                continue

            elif char.isalnum():
                current_word.append(char)
            else:
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
        
            i += 1
        
        if current_word:
            tokens.append(''.join(current_word))
        
        return tokens

# -----------------------------------------------------------
class VocabUpdater:
    def __init__(self, vocab):
        self.vocab = vocab  # pandas DataFrame with index as word, columns: ['freq', 'value']
        self.placeholder_pattern = re.compile(r'^token\d+$') # token12345
        self.vowels = set('aeiouyăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')
    
    def normalize_repeated_chars(self, word):
        """
        Normalize words with repeated characters to their base form.
        Examples: 'tốtt' -> 'tốt', 'ngonnn' -> 'ngon'
        """
        if len(word) < 3:
            return word
        
        # Find sequences of 2 or more repeated characters
        normalized = word
        for i in range(len(word) - 1):
            if word[i] == word[i+1]:
                # Find the end of the repeated sequence
                j = i + 2
                while j < len(word) and word[j] == word[i]:
                    j += 1
                # Replace the repeated sequence with just one character
                normalized = normalized[:i+1] + normalized[j:]
                break
        
        return normalized
    
    def is_gibberish(self, word):
        # min 2 chars, contains a vowel, not all digits, not a placeholder
        if len(word) < 2:
            return True
        if self.placeholder_pattern.match(word):
            return True
        if word.isdigit():
            return True
        if not any(v in word for v in self.vowels):
            return True
        return False

    def update_with_new_word(self, new_word):
        # First normalize the word to handle repeated characters
        normalized_word = self.normalize_repeated_chars(new_word)
        
        # Check if normalized version already exists in vocab
        if normalized_word in self.vocab.index:
            return False  # Already in vocab
        
        # Check if gibberish
        if self.is_gibberish(new_word):
            return False
        
        # If the normalized word is different from original, check if it's gibberish too
        if normalized_word != new_word and self.is_gibberish(normalized_word):
            return False
        
        word_to_add = normalized_word if normalized_word != new_word else new_word
        
        placeholders = [w for w in self.vocab.index if self.placeholder_pattern.match(w)]
        if not placeholders:
            return False
        
        placeholder = placeholders[0]
        value = self.vocab.loc[placeholder, 'value']
        freq = zipf_frequency(word_to_add, 'vi')
        self.vocab = self.vocab.drop(placeholder)
        self.vocab.loc[word_to_add] = [freq, value]
        return True

    def get_vocab(self):
        return self.vocab

# read line from dataset
# call customtokenizer -> output token vector
# check token in vocab, update new token 

# -----------------------------------------------------------
class MyEmbedding(T.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(MyEmbedding, self).__init__()
        self.embedding = T.nn.Embedding(vocab_size, embed_dim)
        T.nn.init.uniform_(self.embedding.weight, -0.2, +0.2)

    def forward(self, x):
        return self.embedding(x)

# -----------------------------------------------------------
class MultiHeadAttention(T.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = T.nn.Linear(embed_dim, embed_dim)
        self.k_proj = T.nn.Linear(embed_dim, embed_dim)
        self.v_proj = T.nn.Linear(embed_dim, embed_dim)
        self.out_proj = T.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # softmax((q @ a @ q^T )/sqrt(dim)) @ v
        
        # survey them cac loai head khac -> danh gia thuc nghiem
        # ve heatmap 
        # qkv -> detect binh thuong
        # q a q^T -> detect spam

        scores = T.matmul(q, k.transpose(-2, -1)) / T.sqrt(T.tensor(self.head_dim, dtype=T.float32))
        attn_weights = T.softmax(scores, dim=-1)
        
        context = T.matmul(attn_weights, v)
        
        # Reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(context)

# -----------------------------------------------------------
class MultiHeadSentimentModel(T.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=4, num_heads=4):
        super(MultiHeadSentimentModel, self).__init__()
        # Embedding layer
        self.embed = MyEmbedding(vocab_size, embed_dim)
        self.attention_layers = T.nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        # Layer normalization for each attention layer
        self.layer_norms = T.nn.ModuleList([
            T.nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        self.dropout = T.nn.Dropout(0.2)
        # Final classification layer
        self.fc = T.nn.Linear(embed_dim, 1)
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        # word embeddings
        z = self.embed(x)  # [batch_size, seq_len, embed_dim]
        # Apply multiple layers of attention
        for i, (attention, norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            normalized = norm(z)
            attended = attention(normalized)
            z = z + attended
            z = self.dropout(z)
        pooled = T.mean(z, dim=1)  # [batch_size, embed_dim]
        # Final classification
        output = T.sigmoid(self.fc(pooled))  # [batch_size, 1]
        return output

# -----------------------------------------------------------
class Vietnamese_Dataset(T.utils.data.Dataset):
    def __init__(self, src_file, vocab, max_len=128, update_vocab=False):
        df = pd.read_csv(src_file)
        self.comments = df['comment'].values
        self.labels = df['label'].values
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = CustomTokenizer()
        self.update_vocab = update_vocab
        if self.update_vocab:
            self.vocab_updater = VocabUpdater(self.vocab)
            # First pass: update vocab with all new words
            for comment in self.comments:
                tokens = self.tokenizer.tokenize(comment)
                for token in tokens:
                    if token not in self.vocab.index:
                        self.vocab_updater.update_with_new_word(token)
            self.vocab = self.vocab_updater.get_vocab()
        x_data = []
        y_data = []
        # Second pass: build tensors using updated vocab
        for comment, label in zip(self.comments, self.labels):
            tokens = self.tokenizer.tokenize(comment)
            token_ids = []
            for token in tokens:
                if token in self.vocab.index:
                    token_ids.append(self.vocab.loc[token, 'value'])
                else:
                    token_ids.append(0)
            if len(token_ids) < max_len:
                token_ids = [0] * (max_len - len(token_ids)) + token_ids
            else:
                token_ids = token_ids[-max_len:]
            x_data.append(token_ids)
            y_data.append(label)
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
    vocab = pd.read_json(vocab_path, orient='index')
    vocab.columns = ['freq', 'value']
    return vocab

def accuracy(model, dataset):
    num_correct = 0
    num_wrong = 0
    ldr = T.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
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

def save_model(model, vocab_size, save_path):
    model_info = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embed_dim': model.embed.embedding.weight.shape[1],
        'num_layers': len(model.attention_layers),
        'num_heads': model.attention_layers[0].num_heads
    }
    T.save(model_info, save_path)
    print(f"Model saved to {save_path}")

def load_model(model_path, device):
    """Load model and its configuration"""
    model_info = T.load(model_path, map_location=device)
    model = MultiHeadSentimentModel(
        vocab_size=model_info['vocab_size'],
        embed_dim=model_info['embed_dim'],
        num_layers=model_info['num_layers'],
        num_heads=model_info['num_heads']
    ).to(device)
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    return model

def compute_confusion_matrix(model, dataset):
    model.eval()
    all_preds = []
    all_labels = []
    
    ldr = T.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    with T.no_grad():
        for (batch_idx, batch) in enumerate(ldr):
            X = batch[0]
            Y = batch[1]
            oupt = model(X)
            preds = (oupt > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())
    
    return confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("\nBegin PyTorch Vietnamese Multi-Head Attention demo")
    gc.enable()
    print("\nLoading preprocessed train and test data")
    train_file = "/kaggle/input/vietnamese-feedback-training/reviews.csv"
    test_file = "/kaggle/input/vietnamese-feedback-testing/test.csv"
    # 2. load vocab
    vocab = load_vocab("/kaggle/input/vocab-json/vocab_with_freq.json")
    # --- Unified vocab update for both train and test ---
    vocab_updater = VocabUpdater(vocab)
    tokenizer = CustomTokenizer()
    for file in [train_file, test_file]:
        df = pd.read_csv(file)
        comments = df['comment'].values
        for comment in comments:
            tokens = tokenizer.tokenize(comment)
            for token in tokens:
                if token not in vocab_updater.vocab.index:
                    vocab_updater.update_with_new_word(token)
    unified_vocab = vocab_updater.get_vocab()
    unified_vocab.to_json('/kaggle/working/vocab_with_freq_updated_unified.json', orient='index', force_ascii=False)
    print("Unified vocab saved to /kaggle/working/vocab_with_freq_updated_unified.json")
    vocab_size = len(unified_vocab)
    print(f"Unified Vocabulary size: {vocab_size}")
    train_ds = Vietnamese_Dataset(train_file, unified_vocab, max_len=128, update_vocab=False)
    test_ds = Vietnamese_Dataset(test_file, unified_vocab, max_len=128, update_vocab=False)

    # Increase batch size for GPU
    bat_size = 32 if device.type == 'cuda' else 16
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True, drop_last=False)
    n_train = len(train_ds)
    n_test = len(test_ds)

    print("Num train = %d Num test = %d" % (n_train, n_test))

    # 3. create model
    print("\nCreating Multi-Head Attention model")
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    max_epochs = 40

    model = MultiHeadSentimentModel(vocab_size, embed_dim, num_layers, num_heads).to(device)
    print(f"Model created with {num_layers} layers, {num_heads} heads per layer (total {num_layers * num_heads} heads)")

    # 4. train model
    loss_func = T.nn.BCELoss()
    lrn_rate = 1e-4
    optimizer = T.optim.Adam(model.parameters(), lr=lrn_rate)

    print("\nbatch size = " + str(bat_size))
    print("loss func = " + str(loss_func))
    print("optimizer = Adam")
    print("learn rate = %0.4f" % lrn_rate)
    print("max_epochs = %d" % max_epochs)

    # lists to store training metrics 
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("\nStarting training")
    model.train()
    for epoch in range(0, max_epochs):
        tot_err = 0.0
        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch[0].to(device)
            Y = batch[1].to(device)
            optimizer.zero_grad()
            oupt = model(X)
            loss_val = loss_func(oupt, Y)
            tot_err += loss_val.item()
            loss_val.backward()
            optimizer.step()

            # Clear cache periodically
            if batch_idx % 100 == 0:
                if device.type == 'cuda':
                    T.cuda.empty_cache()

        # Calculate metrics for this epoch
        epoch_loss = tot_err / len(train_ldr)
        train_losses.append(epoch_loss)
        
        model.eval()
        train_acc = accuracy(model, train_ds)
        test_acc = accuracy(model, test_ds)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print("epoch = %4d  |" % epoch, end="")
        print("   loss = %10.4f  |" % epoch_loss, end="")
        print("  train acc = %8.2f%%  |" % train_acc, end="")
        print("  test acc = %8.2f%%" % test_acc)
        model.train()

        # Clear memory after each epoch
        if device.type == 'cuda':
            T.cuda.empty_cache()
        gc.collect()

    print("Training complete")

    # save training metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': range(max_epochs),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies
    })
    metrics_df.to_csv('/kaggle/working/training_metrics.csv', index=False)

    # save training curves plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    sns.lineplot(data=train_losses, color='blue', linewidth=2)
    plt.title('Training Loss Over Time', fontsize=12, pad=15)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    sns.lineplot(data=train_accuracies, color='green', label='Training Accuracy', linewidth=2)
    sns.lineplot(data=test_accuracies, color='red', label='Test Accuracy', linewidth=2)
    plt.title('Model Accuracy Over Time', fontsize=12, pad=15)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 5. evaluate
    model.eval()
    test_acc = accuracy(model, test_ds)
    print("\nAccuracy on test data = %8.2f%%" % test_acc)

    # Compute and plot confusion matrix
    print("\nComputing confusion matrix...")
    cm = compute_confusion_matrix(model, test_ds)
    plot_confusion_matrix(cm, '/kaggle/working/confusion_matrix.png')
    print("Confusion matrix saved to /kaggle/working/confusion_matrix.png")

    # 6. save model
    print("\nSaving trained model state")
    model_save_path = "/kaggle/working/multihead_vietnamese_model.pt"
    save_model(model, vocab_size, model_save_path)

    print("\nEnd PyTorch Vietnamese Multi-Head Attention demo")

if __name__ == "__main__":
    main() 

# fine-tuning khi co tu moi vao vocab
# sua toan trong multi head attention
