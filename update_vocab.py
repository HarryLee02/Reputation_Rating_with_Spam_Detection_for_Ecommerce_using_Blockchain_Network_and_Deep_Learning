import json
import pandas as pd
import emoji
import re

# Load dataset.csv
df = pd.read_csv("vispamdetection_dataset/reviews.csv/reviews.csv")

# Load vocab.json
with open("vocab/single_word_vocab.json", "r", encoding="utf-8") as file:
    vocab_data = json.load(file)

# Function to extract emojis from text
def extract_emojis(text):
    return [char for char in text if emoji.is_emoji(char)]

# Identify emojis in comments and update vocab.json
new_vocab = vocab_data.copy()  # Preserve original structure
token_keys = [key for key in vocab_data if key.startswith("token")]  # Placeholder keys

for comment in df["comment"]:  # Iterate through comments
    emojis_in_comment = extract_emojis(comment)
    
    for emj in emojis_in_comment:
        if emj not in new_vocab:  # If emoji not already in vocab
            if token_keys:  # Check if placeholders exist
                placeholder = token_keys.pop(0)  # Get first placeholder key
                new_vocab[emj] = new_vocab.pop(placeholder)  # Replace placeholder with emoji

# Save updated vocab.json
with open("vocab/updated_vocab.json", "w", encoding="utf-8") as file:
    json.dump(new_vocab, file, ensure_ascii=False, indent=4)

print("Vocab updated successfully!")
