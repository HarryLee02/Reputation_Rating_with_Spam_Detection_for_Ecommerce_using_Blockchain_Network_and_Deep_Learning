import json
import re

with open('vocab/vocab.json', 'r', encoding='utf-8') as f:
    old_vocab = json.load(f)

new_vocab = {}

for compound_word in old_vocab.keys():
    words = re.split(r'[;,\-\s]+', compound_word)
    # words = compound_word.split(' ')
    for word in words:
        word = word.lower()
        if word not in new_vocab and word.isdigit() == False:
            new_vocab[word] = None  # Placeholder value


# 28 first index are gibberish words
sorted_new_vocab = {key: idx + 1 for idx, key in enumerate(sorted(new_vocab.keys())[28:])}

tmp_size = len(sorted_new_vocab)
# add everytime so be careful
for i in range(1,10000):
    key = "token" + str(tmp_size+i)
    sorted_new_vocab[key] = tmp_size+i

with open('vocab/single_word_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(sorted_new_vocab, f, ensure_ascii=False, indent=2)

print("Sorted and re-indexed new vocabulary has been saved to vocab/single_word_vocab.json")
