import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the pretrained embedding model
class ByteEmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(ByteEmbeddingModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 256  # Byte-level vocabulary size (0 to 255)
embedding_dim = 100  # Embedding dimension, must match the trained model

model = ByteEmbeddingModel(vocab_size, embedding_dim).to(device)
model.load_state_dict(torch.load('byte_embedding_model.pth'))
model.eval()

# Function to byte-encode a word
def byte_encode(word):
    # Convert word to bytes and clip to range (0, 255)
    byte_ids = [min(ord(char), 255) for char in word]
    return byte_ids

# Function to get the embedding for a word
def get_word_embedding(word, model, device):
    byte_ids = byte_encode(word)
    max_sequence_length = 500  # Adjust based on your model
    byte_ids = byte_ids[:max_sequence_length] + [0] * (max_sequence_length - len(byte_ids))  # Padding
    word_tensor = torch.tensor(byte_ids).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        embedding = model(word_tensor)
    return embedding.squeeze().cpu().numpy()

# Function to find the closest words based on cosine similarity
def find_closest_words(input_word, model, device, top_n=5):
    # Load your dataset to get words (make sure it includes the 'crimeaditionalinfo' column or vocabulary)
    file_path = 'train.csv'  # Replace with your dataset file path
    df = pd.read_csv(file_path)
    words = set(df['crimeaditionalinfo'].astype(str).apply(lambda x: x.split()).sum())

    # Get the embedding for the input word
    input_embedding = get_word_embedding(input_word, model, device)

    word_embeddings = []
    for word in words:
        word_embedding = get_word_embedding(word, model, device)
        word_embeddings.append(word_embedding)

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity([input_embedding], word_embeddings)[0]

    # Get top_n closest words
    closest_indices = cosine_similarities.argsort()[-top_n:][::-1]
    closest_words = [list(words)[i] for i in closest_indices]

    return closest_words, cosine_similarities[closest_indices]

# Example usage
input_word = "fraud"  # Replace with the word you want to search for
closest_words, similarities = find_closest_words(input_word, model, device, top_n=5)

print(f"Closest words to '{input_word}':")
for word, similarity in zip(closest_words, similarities):
    print(f"{word}: {similarity:.4f}")
