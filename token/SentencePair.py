import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load your dataset and preprocess the sentences
file_path = 'train.csv'  # Replace with your dataset file path
df = pd.read_csv(file_path)

# Ensure that the 'crimeaditionalinfo' column exists and process the sentences
df['sentence'] = df['crimeaditionalinfo'].astype(str)
df['cleaned_sentence'] = df['sentence'].str.replace('[^a-zA-Z0-9\s]', '', regex=True).str.lower()

# Save the cleaned sentences to a text file for training the tokenizer
with open('sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in df['cleaned_sentence']:
        f.write(sentence + "\n")

# Step 2: Train a SentencePiece model using the cleaned sentences
spm.SentencePieceTrainer.train(input='sentences.txt', model_prefix='tokenizer_model', vocab_size=57400, character_coverage=1.0, model_type='bpe')

# Step 3: Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file='tokenizer_model.model')

# Step 4: Tokenize the cleaned sentences using the SentencePiece model
tokenized_data = [sp.encode_as_ids(sentence) for sentence in df['cleaned_sentence']]

# Step 5: Pad or truncate to a fixed length (e.g., 500 tokens)
max_sequence_length = 500
tokenized_data = [x[:max_sequence_length] + [0] * (max_sequence_length - len(x)) for x in tokenized_data]
input_tensor = torch.tensor(tokenized_data)

# Step 6: Define the model for sentence embeddings
class SentenceEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SentenceEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Step 7: Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move it to the selected device
embedding_dim = 100  # Adjust embedding dimension as needed
vocab_size = sp.get_piece_size()  # Get the vocabulary size from the SentencePiece model
model = SentenceEmbeddingModel(vocab_size, embedding_dim).to(device)

# Step 8: Initialize the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Modify loss based on your task

# Step 9: Prepare DataLoader for training
dataset = TensorDataset(input_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 10: Train the model
epochs = 10  # Set the number of epochs for training
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

# Step 11: Generate an embedding vector for a sample sentence
sample_sentence = "This is a test sentence."
sample_token_ids = sp.encode_as_ids(sample_sentence)  # Tokenize the sample sentence

# Convert to tensor, move it to the GPU, and pass through the model to get embeddings
sample_tensor = torch.tensor(sample_token_ids).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
sample_embeddings = model(sample_tensor)

# Print the embedding result
print("Embedding for the sample sentence:")
print(sample_embeddings.cpu())  # Move back to CPU for printing

# Step 12: Save the trained model
torch.save(model.state_dict(), 'sentence_embedding_model.pth')  # Save the PyTorch model
print("Model saved successfully.")
