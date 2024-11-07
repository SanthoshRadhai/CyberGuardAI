import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and preprocess your dataset
file_path = 'train.csv'  # Replace with your dataset file path
df = pd.read_csv(file_path)

# Ensure that the 'crimeaditionalinfo' column exists
df['sentence'] = df['crimeaditionalinfo'].astype(str)

# Step 2: Byte-level encoding function
def byte_encode(sentence):
    # Convert each character to its byte value and ensure the range is valid
    byte_ids = [min(ord(char), 255) for char in sentence]  # Clip to 255 if necessary
    return byte_ids

# Convert sentences to byte-level encoded lists
encoded_data = [byte_encode(sentence) for sentence in df['sentence']]

# Step 3: Pad or truncate to a fixed length (e.g., 500 bytes)
max_sequence_length = 500
encoded_data = [x[:max_sequence_length] + [0] * (max_sequence_length - len(x)) for x in encoded_data]
input_tensor = torch.tensor(encoded_data, dtype=torch.long)

# Step 4: Define the model for byte-level embeddings
class ByteEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(ByteEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Step 5: Set device and initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The vocab size for bytes is 256 (from 0 to 255), and embedding_dim is the vector size
vocab_size = 256
embedding_dim = 100  # Adjust embedding dimension as needed
model = ByteEmbeddingModel(vocab_size, embedding_dim).to(device)

# Step 6: Initialize the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Modify loss based on your task

# Step 7: Prepare DataLoader for training
dataset = TensorDataset(input_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 8: Train the model
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

# Step 9: Generate an embedding vector for a sample sentence
sample_sentence = "This is a test sentence."
sample_byte_ids = byte_encode(sample_sentence)[:max_sequence_length]  # Encode and truncate
sample_byte_ids += [0] * (max_sequence_length - len(sample_byte_ids))  # Pad

# Convert to tensor, move it to the GPU, and pass through the model to get embeddings
sample_tensor = torch.tensor(sample_byte_ids).unsqueeze(0).to(device)  # Add batch dimension
sample_embeddings = model(sample_tensor)

# Print the embedding result
print("Embedding for the sample sentence:")
print(sample_embeddings.cpu())  # Move back to CPU for printing

# Step 10: Save the trained model
torch.save(model.state_dict(), 'byte_embedding_model.pth')
print("Model saved successfully.")
