from transformers import BertTokenizer

# Load the multilingual BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize and encode a sample text
sample_text = "bhai can't find the suspect"
tokenized = tokenizer.tokenize(sample_text)
encoded = tokenizer.encode(sample_text, add_special_tokens=True)

print("Tokenized:", tokenized)
print("Encoded:", encoded)
