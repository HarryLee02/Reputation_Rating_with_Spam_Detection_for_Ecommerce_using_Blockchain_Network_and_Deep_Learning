import json
import re
from wordfreq import word_frequency, zipf_frequency

with open('vocab/updated_vocab.json', 'r', encoding='utf-8') as f:
    old_vocab = json.load(f)

new_vocab = old_vocab.copy() 
for keys,value in old_vocab.items():

    zipf = zipf_frequency(keys, 'vi')
    # print(f"Word: {word}, Zipf: {zipf}")
    new_vocab[keys] = [zipf,value]

with open('vocab/vocab_with_freq.json', 'w', encoding='utf-8') as f:
    json.dump(new_vocab, f, ensure_ascii=False, indent=2)