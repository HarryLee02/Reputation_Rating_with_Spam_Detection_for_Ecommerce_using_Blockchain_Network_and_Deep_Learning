import torch as T
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score, classification_report

device = T.device('cpu')

class MyEmbedding(T.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(MyEmbedding, self).__init__()
        self.weight = T.nn.Parameter(T.zeros((vocab_size, embed_dim), dtype=T.float32))
        T.nn.init.uniform_(self.weight, -0.1, +0.1)

    def forward(self, x):
        return self.weight[x]

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

def load_vocab(vocab_path):
    vocab = pd.read_json(vocab_path, orient='index')
    vocab.columns = ['freq', 'value']
    return vocab

def load_model(model_path, device):
    """Load model and its configuration"""
    model_info = T.load(model_path, map_location=device)
    model = LSTM_Net(
        vocab_size=model_info['vocab_size'],
        embed_dim=model_info['embed_dim'],
        hidden_dim=model_info['hidden_dim'],
        num_layers=1,  # Set to 2 layers to match the saved model
        dropout=0.2    # Set to 0.3 to match the saved model
    ).to(device)
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    return model

def predict_sentiment(comment, model, vocab, max_len=50):
    """Predict sentiment for a single Vietnamese comment"""
    tokenizer = RegexpTokenizer(r'\w+')
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
    
    # Convert to tensor and add batch dimension
    review = T.tensor(token_ids, dtype=T.int64).to(device)
    review = review.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
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
        
        # Print progress
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
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = load_vocab("vocab/vocab_with_freq.json")
    
    # Load model
    print("Loading model...")
    model = load_model("models/vietnamese_lstm_model.pt", device)
    # model = load_model("models/vietnamese_multilayer_lstm_model.pt", device)

    
    # Evaluate on test dataset
    test_file = "vispamdetection_dataset/reviews.csv/reviews.csv"
    evaluate_model(model, test_file, vocab)  # Test on all samples
    
    # Test with some example comments
    print("\nTesting with example comments:")
    print("0 = no spam, 1 = spam")
    print("-" * 50)
    
    test_comments = [
        # "Sản phẩm rất tốt, chất lượng cao",
        # "Dịch vụ tệ, không đáng tiền",
        # "Sản phẩm bình thường, không có gì đặc biệt",
        # "Rất hài lòng với dịch vụ, sẽ quay lại",
        # "Chất lượng kém, không nên mua",
        # "Nsnwnwnxnwkkxmmxmsmwmsmsxnndnwnxnsnwnznn ncn1msnxnkwf",
        "Video và ảnh mang tính chất minh hoạ,",
        "Mới nhận được hàng chất lượng khá ôn so với giá tiền jjjjjjjjjjjjj",
        "Hàng đẹp, ví sờ mềm rất thích tay, shop giao hàng nhanh, đóng gói cẩn thận.",
        "Túi rất chắc chắn và nhiều ngăn, dáng khoẻ khoắn hợp cả nam lẫn nữ. Thời gian giao hàng rất nhanh mình ưng ý."
    ]
    
    for comment in test_comments:
        sentiment = predict_sentiment(comment, model, vocab)
        print(f"\nComment: {comment}")
        print(f"Sentiment score: {sentiment:.4f}")
        print(f"Prediction: {'Spam' if sentiment > 0.5 else 'No Spam'}")

if __name__ == "__main__":
    main() 