import pandas as pd
import re
import contractions
import json

class CustomTokenizer:
    def __init__(self):
        self.vocab = {}
        self.id_counter = 1  # Start indexing from 1 (reserve 0 for padding)
    
    def preprocess_text(self, text):
        # Expand contractions
        expanded_text = contractions.fix(text)
        return expanded_text

    def tokenize(self, text):
        # Split text by spaces and punctuation (simplified example)
        tokens = re.findall(r'\w+|\S+', text.lower())  # Matches words or punctuation
        return tokens

    def build_vocab(self, texts):
        for text in texts:
            expanded_text = self.preprocess_text(text)
            tokens = self.tokenize(expanded_text)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.id_counter
                    self.id_counter += 1

    def encode(self, text):
        # Convert text to tokens, then tokens to IDs
        expanded_text = self.preprocess_text(text)
        tokens = self.tokenize(expanded_text)
        return [self.vocab.get(token, 0) for token in tokens]  # 0 for OOV tokens

    def pad_sequences(self, sequences, max_length):
        # Pad or truncate sequences to a fixed length
        return [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in sequences]

# Example usage
df = pd.DataFrame({'crimeaditionalinfo': ["can't find the suspect", "I'm at the scene", "it's over there"]})

# Initialize and build tokenizer
tokenizer = CustomTokenizer()
tokenizer.build_vocab(df['crimeaditionalinfo'])

# Tokenize and encode each entry
df['tokenized'] = df['crimeaditionalinfo'].apply(tokenizer.tokenize)
df['encoded'] = df['crimeaditionalinfo'].apply(tokenizer.encode)

# Display results
print("Vocabulary:", tokenizer.vocab)
print(df[['crimeaditionalinfo', 'tokenized', 'encoded']])



# Save the vocabulary dictionary as JSON
with open('tokenizer_vocab.json', 'w') as f:
    json.dump(tokenizer.vocab, f)
