import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load the dataset
file_path = 'train.csv'  # Replace with your dataset file path
df = pd.read_csv(file_path)

# Ensure the text column is properly formatted
df['sentence'] = df['crimeaditionalinfo'].astype(str)  # Replace 'sentence_column' with the actual column name in your CSV

# Step 2: Preprocess the data (remove unwanted characters, convert to lowercase)
df['cleaned_sentence'] = df['sentence'].str.replace('[^a-zA-Z0-9\s]', '', regex=True).str.lower()

# Save the cleaned sentences to a text file for training SentencePiece
df['cleaned_sentence'].to_csv('sentences.txt', index=False, header=False)

# Step 3: Train the SentencePiece tokenizer with a dynamic vocabulary size
spm.SentencePieceTrainer.train(input='sentences.txt', model_prefix='tokenizer_model', vocab_size=57400, character_coverage=1.0, model_type='unigram')

# Step 4: Load the SentencePiece model
sp = spm.SentencePieceProcessor(model_file='tokenizer_model.model')

# Step 5: Tokenize the dataset and convert sentences to token IDs
tokenized_data = df['cleaned_sentence'].apply(lambda x: sp.encode_as_ids(x))

# Optional: Pad or truncate sentences to a fixed length (e.g., 50 tokens)
max_sequence_length = 500
tokenized_data = tokenized_data.apply(lambda x: x[:max_sequence_length] + [0] * (max_sequence_length - len(x)))

# Step 6: Convert the tokenized data into a PyTorch tensor
input_tensor = torch.tensor(tokenized_data.tolist())

# Step 7: Define the sentence embedding model
class SentenceEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SentenceEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Step 8: Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move it to the selected device (GPU or CPU)
embedding_dim = 100  # Adjust embedding dimension as needed
vocab_size = sp.get_piece_size()  # Get the dynamic vocabulary size
model = SentenceEmbeddingModel(vocab_size, embedding_dim).to(device)

# Step 9: Initialize the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Modify loss based on your task

# Step 10: Prepare DataLoader for training
dataset = TensorDataset(input_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 11: Train the model (dummy example, modify loss and task as needed)
epochs = 100   # Set the number of epochs for training
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Get the batch of token IDs and move it to the GPU
        input_ids = batch[0].to(device)

        # Forward pass through the model (generate embeddings)
        embeddings = model(input_ids)

        # Example loss: minimize the difference between the embeddings
        loss = criterion(embeddings, embeddings)  # Modify loss depending on your task

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.10f}")

# Step 12: Generate embedding vectors for a sample sentence
sample_sentence = "This is a test sentence."
sample_token_ids = sp.encode_as_ids(sample_sentence)  # Tokenize the sample sentence

# Convert to tensor, move it to the GPU, and pass through the model to get embeddings
sample_tensor = torch.tensor(sample_token_ids).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
sample_embeddings = model(sample_tensor)

# Print the embedding result
print("Embedding for the sample sentence:")
print(sample_embeddings.cpu())  # Move back to CPU for printing

# Step 13: Show the dynamic vocabulary size
print("Dynamic Vocabulary Size:", vocab_size)

# Step 14: Save the trained model and tokenizer
torch.save(model.state_dict(), 'sentence_embedding_model.pth')  # Save the PyTorch model

print("Model saved successfully.")