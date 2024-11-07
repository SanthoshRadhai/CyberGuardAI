import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Load CSV file
file_path = "train.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Extract the specific column
column_name = 'crimeaditionalinfo'

# Function to get embeddings
def get_embeddings(text):
    # Tokenize and get token IDs with attention mask
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract embeddings from the last hidden state
    embeddings = outputs.last_hidden_state
    return embeddings.squeeze().numpy()  # Converts to a 2D array (tokens x embedding dim)

# Apply embedding function to each entry
df['embeddings'] = df[column_name].apply(get_embeddings)

# Display results
print(df[['crimeaditionalinfo', 'embeddings']])

# Save the tokenizer to a directory (this will create a tokenizer.json file along with other files)
tokenizer.save_pretrained("my_tokenizer")
