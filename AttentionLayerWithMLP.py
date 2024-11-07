import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader

# Load the SentencePiece model for tokenization
sp = spm.SentencePieceProcessor(model_file='tokenizer_model.model')

# Define a dataset class to load and preprocess the data
class TextDataset(Dataset):
    def __init__(self, csv_file, sp_model, max_seq_length=500):
        self.data = pd.read_csv(csv_file)
        self.sp_model = sp_model
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target_class_1 = self.data.iloc[idx, 0]  # First target class
        target_class_2 = self.data.iloc[idx, 1]  # Second target class
        input_text = self.data.iloc[idx, 2]  # Input text (3rd column)

        # Tokenize the input text using the SentencePiece model
        tokenized_input = self.sp_model.encode_as_ids(input_text)
        
        # Pad or truncate to fixed sequence length
        tokenized_input = tokenized_input[:self.max_seq_length] + [0] * (self.max_seq_length - len(tokenized_input))
        
        return torch.tensor(tokenized_input), torch.tensor([target_class_1, target_class_2])

# Define the self-attention model
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.embedding_dim**0.5
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        attended_values = torch.matmul(attention_weights, V)
        output = self.output_layer(attended_values)
        
        return output, attention_weights

# Define the overall model
class AttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SelfAttention(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        attn_output, attention_weights = self.attention(x)
        pooled_output = attn_output.mean(dim=1)  # Global average pooling
        output = self.fc(pooled_output)
        return output, attention_weights

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embedding_dim = 100
vocab_size = sp.get_piece_size()  # Vocabulary size from SentencePiece model
num_classes = 2  # Adjust for your task (e.g., 2 target classes)
batch_size = 32
learning_rate = 0.001
epochs = 5
max_seq_length = 500

# Create the dataset and data loader
train_dataset = TextDataset('your_dataset.csv', sp, max_seq_length=max_seq_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = AttentionModel(vocab_size, embedding_dim, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        # Move data to device (GPU or CPU)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, attn_weights = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels.float())  # Ensure labels are float for BCEWithLogitsLoss
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Optionally print attention weights for the first batch
    if epoch == 0:
        print("Attention Weights (First Batch):")
        print(attn_weights[0].cpu().detach().numpy())

# Save the trained model
torch.save(model.state_dict(), 'self_attention_model.pth')
