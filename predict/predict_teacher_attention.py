import torch as T
import pandas as pd
import numpy as np
from teacher_attention_model import TeacherAttentionModel, CustomTokenizer, load_vocab, load_model
from sklearn.metrics import accuracy_score, classification_report

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def predict_sentiment(text, model, vocab, tokenizer, max_len=128):
    """
    Predict sentiment for a single Vietnamese text using the teacher attention model
    
    Args:
        text (str): Vietnamese text to classify
        model: Trained TeacherAttentionModel
        vocab: Vocabulary DataFrame
        tokenizer: CustomTokenizer instance
        max_len (int): Maximum sequence length
    
    Returns:
        float: Sentiment score between 0 and 1 (0 = negative, 1 = positive)
    """
    model.eval()
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    # Convert tokens to IDs
    token_ids = []
    for token in tokens:
        if token in vocab.index:
            token_ids.append(vocab.loc[token, 'value'])
        else:
            token_ids.append(0)  # Unknown token
    
    # Pad or truncate to max_len
    if len(token_ids) < max_len:
        token_ids = [0] * (max_len - len(token_ids)) + token_ids
    else:
        token_ids = token_ids[-max_len:]
    
    # Convert to tensor
    x = T.tensor([token_ids], dtype=T.int64).to(device)
    
    # Make prediction
    with T.no_grad():
        output = model(x)
        sentiment_score = output.item()
    
    return sentiment_score

def predict_batch(texts, model, vocab, tokenizer, max_len=128):
    """
    Predict sentiment for a batch of Vietnamese texts
    
    Args:
        texts (list): List of Vietnamese texts to classify
        model: Trained TeacherAttentionModel
        vocab: Vocabulary DataFrame
        tokenizer: CustomTokenizer instance
        max_len (int): Maximum sequence length
    
    Returns:
        list: List of sentiment scores between 0 and 1
    """
    model.eval()
    
    # Process all texts
    all_token_ids = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = []
        for token in tokens:
            if token in vocab.index:
                token_ids.append(vocab.loc[token, 'value'])
            else:
                token_ids.append(0)
        
        if len(token_ids) < max_len:
            token_ids = [0] * (max_len - len(token_ids)) + token_ids
        else:
            token_ids = token_ids[-max_len:]
        
        all_token_ids.append(token_ids)
    
    # Convert to tensor
    x = T.tensor(all_token_ids, dtype=T.int64).to(device)
    
    # Make predictions
    with T.no_grad():
        outputs = model(x)
        sentiment_scores = outputs.cpu().numpy().flatten()
    
    return sentiment_scores.tolist()

def evaluate_model(model, test_file, vocab, tokenizer, max_samples=None):
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
        pred_score = predict_sentiment(comment, model, vocab, tokenizer)
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
    print("Vietnamese Sentiment Analysis with Teacher Attention Model")
    print("=" * 60)
    
    # Load the trained model
    model_path = "models/teacher_attention_vietnamese_model.pt"
    vocab_path = "vocab_with_freq_updated_unified_teacher.json"
    
    try:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, device)
        print("Model loaded successfully!")
        
        print(f"Loading vocabulary from {vocab_path}...")
        vocab = load_vocab(vocab_path)
        print(f"Vocabulary loaded! Size: {len(vocab)}")
        
        # Initialize tokenizer
        tokenizer = CustomTokenizer()
        
        # Example Vietnamese texts for testing
        test_texts = [
            "Sản phẩm rất tốt, chất lượng cao, giao hàng nhanh chóng!",
            "Dịch vụ tệ hại, nhân viên thái độ không tốt, không nên mua",
            "Sản phẩm bình thường, giá cả hợp lý",
            "Tuyệt vời! Rất hài lòng với dịch vụ này",
            "Chất lượng kém, không đáng tiền",
            "🤮🤮🤮🤮🤮",
            "Món đồ ko đc tốt, ko nên mua nhé mn",
            "Quá tệ🤮 sản phẩm không đáng để mua mọi người ạ",
            "Video và ảnh mang tính chất minh hoạ,",
            "tốt",
            "Rất tốt, nên mua nhé.",
            "Túi rất chắc chắn và nhiều ngăn, dáng khoẻ khoắn hợp cả nam lẫn nữ. Thời gian giao hàng rất nhanh mình ưng ý."
        ]
        
        print("\n" + "=" * 60)
        print("Testing with example texts:")
        print("=" * 60)
        
        for i, text in enumerate(test_texts, 1):
            sentiment_score = predict_sentiment(text, model, vocab, tokenizer)
            sentiment_label = "Spam" if sentiment_score > 0.5 else "No Spam"
            
            print(f"\nText {i}: {text}")
            print(f"Sentiment Score: {sentiment_score:.4f}")
            print(f"Prediction: {sentiment_label}")
        
        # Evaluate on test dataset
        print("\n" + "=" * 60)
        print("Evaluating on test dataset:")
        print("=" * 60)
        
        test_file = "vispamdetection_dataset/reviews.csv/reviews.csv"
        try:
            # Uncomment the line below to run full evaluation
            evaluate_model(model, test_file, vocab, tokenizer)
            print("Evaluation function ready. Uncomment the line above to run full evaluation.")
        except Exception as e:
            print(f"Error during evaluation: {e}")
        
        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive mode - Enter Vietnamese text to analyze (type 'quit' to exit):")
        print("=" * 60)
        
        while True:
            user_input = input("\nEnter Vietnamese text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            try:
                sentiment_score = predict_sentiment(user_input, model, vocab, tokenizer)
                sentiment_label = "Spam" if sentiment_score > 0.5 else "No Spam"
                
                print(f"Sentiment Score: {sentiment_score:.4f}")
                print(f"Prediction: {sentiment_label}")
                
                # Additional interpretation
                if sentiment_score > 0.8:
                    print("Interpretation: Very likely spam")
                elif sentiment_score > 0.6:
                    print("Interpretation: Likely spam")
                elif sentiment_score > 0.4:
                    print("Interpretation: Uncertain")
                elif sentiment_score > 0.2:
                    print("Interpretation: Likely legitimate")
                else:
                    print("Interpretation: Very likely legitimate")
                    
            except Exception as e:
                print(f"Error processing text: {e}")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find model file. Please make sure {model_path} exists.")
        print("You need to train the model first using teacher_attention_model.py")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main() 