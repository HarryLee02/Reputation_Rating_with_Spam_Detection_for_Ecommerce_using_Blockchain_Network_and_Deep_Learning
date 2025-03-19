import json

text_dict = {}

with open("words.txt", "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)

        sources = data["source"]
        if sources != ["wiktionary"] and sources != ["tudientv"]:
            text = data["text"]
            if text not in text_dict:
                text_dict[text] = len(text_dict) + 1

with open("vocab.json", "w", encoding="utf-8") as outfile:
    json.dump(text_dict, outfile, ensure_ascii=False, indent=4)

print("Processing complete! The new JSON file is 'vocab.json'.")
