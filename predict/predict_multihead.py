import torch as T
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score, classification_report
import re
import emoji

device = T.device('cpu')

class MyEmbedding(T.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(MyEmbedding, self).__init__()
        self.embedding = T.nn.Embedding(vocab_size, embed_dim)
        T.nn.init.uniform_(self.embedding.weight, -0.2, +0.2)

    def forward(self, x):
        return self.embedding(x)

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
        
        # Linear projections and reshape
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = T.matmul(q, k.transpose(-2, -1)) / T.sqrt(T.tensor(self.head_dim, dtype=T.float32))
        attn_weights = T.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = T.matmul(attn_weights, v)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(context)

class MultiHeadSentimentModel(T.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=4, num_heads=4):
        super(MultiHeadSentimentModel, self).__init__()
        
        # Embedding layer
        self.embed = MyEmbedding(vocab_size, embed_dim)
        
        # Multiple layers
        self.attention_layers = T.nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.layer_norms = T.nn.ModuleList([
            T.nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Dropout for regularization
        self.dropout = T.nn.Dropout(0.2)
        
        # Final classification layer
        self.fc = T.nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        
        # Get word embeddings
        embeddings = self.embed(x)  # [batch_size, seq_len, embed_dim]
        
        # Apply multiple layers of attention
        for i, (attention, norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            normalized = norm(embeddings)
            
            attended = attention(normalized)
            
            embeddings = embeddings + attended
            
            # Apply dropout
            embeddings = self.dropout(embeddings)
        
        pooled = T.mean(embeddings, dim=1)  # [batch_size, embed_dim]
        
        output = T.sigmoid(self.fc(pooled))  # [batch_size, 1]
        
        return output

class CustomTokenizer:
    def __init__(self):
        # Word pattern
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        
        # Number pattern
        self.number_pattern = re.compile(r'\d+(?:\.\d+)?')
    
    def is_emoji(self, char):
        # Use emoji library to check
        return emoji.is_emoji(char)
    
    def tokenize(self, text):
        # Convert to lowercase
        text = text.lower()
        
        tokens = []
        current_word = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Check if current character is an emoji
            if self.is_emoji(char):
                # If we have accumulated a word, add it first
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                # Add the emoji as a separate token
                tokens.append(char)
            # Check if current character is a number
            elif char.isdigit():
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                # Start collecting the number
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
        
        # Add any remaining word
        if current_word:
            tokens.append(''.join(current_word))
        
        return tokens

def load_vocab(vocab_path):
    vocab = pd.read_json(vocab_path, orient='index')
    vocab.columns = ['freq', 'value']
    return vocab

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

def predict_sentiment(comment, model, vocab, max_len=100):
    """Predict sentiment for a single Vietnamese comment"""
    tokenizer = CustomTokenizer()
    tokens = tokenizer.tokenize(comment)
    
    # Convert tokens to IDs
    token_ids = []
    for token in tokens:
        if token in vocab.index:
            token_ids.append(vocab.loc[token, 'value'])
        else:
            token_ids.append(0)  # Unknown token
    
    # Pad or truncate
    if len(token_ids) < max_len:
        token_ids = [0] * (max_len - len(token_ids)) + token_ids
    else:
        token_ids = token_ids[-max_len:]
    
    review = T.tensor(token_ids, dtype=T.int64).to(device)
    review = review.unsqueeze(0) 
    
    # prediction
    with T.no_grad():
        prediction = model(review)
    
    return prediction.item()

def evaluate_model(model, test_file, vocab, max_samples=None):
    """Evaluate model on test dataset"""
    # Load test dataset
    df = pd.read_csv(test_file)
    if max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    predictions = []
    true_labels = []
    
    print("\nEvaluating model on test dataset...")
    for idx, row in df.iterrows():
        comment = row['comment']
        true_label = row['label']
        
        # Get prediction
        pred_score = predict_sentiment(comment, model, vocab)
        pred_label = 1 if pred_score > 0.5 else 0
        
        predictions.append(pred_label)
        true_labels.append(true_label)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} comments...")
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions))
    
    return accuracy, predictions, true_labels

def main():
    # load vocab
    print("Loading vocabulary...")
    # vocab = load_vocab("vocab/new_vocab.json")
    vocab = load_vocab("vocab_with_freq_updated_unified.json")

    # Load model
    print("Loading model...")
    # model = load_model("models/multihead_vietnamese_model_new.pt", device)
    model = load_model("multihead_vietnamese_model_copy.pt", device)
    
    # evaluate on test dataset
    test_file = "vispamdetection_dataset/dataset/train.csv"
    evaluate_model(model, test_file, vocab)
    
    # test
    print("\nTesting with example comments:")
    print("0 = no spam, 1 = spam")
    print("-" * 50)
    
    test_comments = [
        "🤮🤮🤮🤮🤮",
        "Món đồ ko đc tốt, ko nên mua nhé mn",
        "Quá tệ🤮 sản phẩm không đáng để mua mọi người ạ",
        "Video và ảnh mang tính chất minh hoạ,",
        "tốt",
        "Rất tốt, nên mua nhé.",
        "Túi rất chắc chắn và nhiều ngăn, dáng khoẻ khoắn hợp cả nam lẫn nữ. Thời gian giao hàng rất nhanh mình ưng ý."
    ]
    
    for comment in test_comments:
        sentiment = predict_sentiment(comment, model, vocab)
        print(f"\nComment: {comment}")
        print(f"Sentiment score: {sentiment:.4f}")
        print(f"Prediction: {'Spam' if sentiment > 0.5 else 'No Spam'}")

if __name__ == "__main__":
    main()