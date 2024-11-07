import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pickle

class AttentionMLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_category_classes, num_sub_category_classes):
        super(AttentionMLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.fc_category = nn.Linear(embedding_dim, num_category_classes)
        self.fc_sub_category = nn.Linear(embedding_dim, num_sub_category_classes)

    def forward(self, input_ids):
        # Embedding layer
        embedded = self.embedding(input_ids)

        # Attention mechanism
        attn_output, _ = self.attention_layer(embedded, embedded, embedded)

        # Global pooling (taking the mean of the embeddings)
        pooled_output = torch.mean(attn_output, dim=1)

        # Output for category
        category_output = self.fc_category(pooled_output)

        # Output for sub-category
        sub_category_output = self.fc_sub_category(pooled_output)

        return category_output, sub_category_output

class NLPAttentionPipeline:
    def __init__(self, file_path, max_sequence_length=500, vocab_size=57400, embedding_dim=96):  # Changed embedding_dim to 96
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        
        # Load only the first 1000 rows of the dataset
        self.df = pd.read_csv(file_path).head(1000)
        print(f"Loaded dataset shape: {self.df.shape}")  # Debug print

        # Encode target columns 'category' and 'sub_category'
        self.category_encoder = LabelEncoder()
        self.sub_category_encoder = LabelEncoder()
        self.df['category'] = self.category_encoder.fit_transform(self.df['category'])
        self.df['sub_category'] = self.sub_category_encoder.fit_transform(self.df['sub_category'])
        print(f"Category classes: {self.category_encoder.classes_}")  # Debug print
        print(f"Sub-category classes: {self.sub_category_encoder.classes_}")  # Debug print
        
        # Save the encoders to disk
        self.save_encoders()
        
        # Store the encoded targets
        self.category_targets = self.df['category'].values
        self.sub_category_targets = self.df['sub_category'].values
        print(f"Category targets sample: {self.category_targets[:10]}")  # Debug print
        print(f"Sub-category targets sample: {self.sub_category_targets[:10]}")  # Debug print
        
        # Preprocess text data
        self.df['cleaned_sentence'] = self.df['crimeaditionalinfo'].astype(str).str.replace('[^a-zA-Z0-9\s]', '', regex=True).str.lower()
        print(f"Cleaned sentences sample: {self.df['cleaned_sentence'].head()}")  # Debug print
        
        # Load pretrained SentencePiece tokenizer model
        self.sp = spm.SentencePieceProcessor(model_file='tokenizer_model.model')
        print(f"Tokenizer loaded successfully!")  # Debug print
        
        # Initialize model, optimizer, and loss functions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = self.sp.get_piece_size()  # Dynamic vocab size from trained tokenizer
        self.num_category_classes = len(self.category_encoder.classes_)
        self.num_sub_category_classes = len(self.sub_category_encoder.classes_)
        print(f"Vocab size from tokenizer: {self.vocab_size}")  # Debug print
        print(f"Number of category classes: {self.num_category_classes}")
        print(f"Number of sub-category classes: {self.num_sub_category_classes}")
        
        self.model = AttentionMLPModel(self.vocab_size, self.embedding_dim, self.num_category_classes, self.num_sub_category_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion_category = nn.CrossEntropyLoss()
        self.criterion_sub_category = nn.CrossEntropyLoss()
        print(f"Model initialized successfully!")  # Debug print
    
    def _tokenize_data(self):
        # Tokenize and pad sequences
        tokenized_data = self.df['cleaned_sentence'].apply(lambda x: self.sp.encode_as_ids(x))
        tokenized_data = tokenized_data.apply(lambda x: x[:self.max_sequence_length] + [0] * (self.max_sequence_length - len(x)))
        
        return (
            torch.tensor(tokenized_data.tolist(), dtype=torch.long),
            torch.tensor(self.category_targets, dtype=torch.long),
            torch.tensor(self.sub_category_targets, dtype=torch.long)
        )
    
    def train(self, epochs=10, batch_size=64):
        input_tensor, category_targets, sub_category_targets = self._tokenize_data()
        dataset = TensorDataset(input_tensor, category_targets, sub_category_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for input_ids, category_targets, sub_category_targets in dataloader:
                input_ids = input_ids.to(self.device)
                category_targets = category_targets.to(self.device)
                sub_category_targets = sub_category_targets.to(self.device)

                self.optimizer.zero_grad()
                category_output, sub_category_output = self.model(input_ids)

                loss_category = self.criterion_category(category_output, category_targets)
                loss_sub_category = self.criterion_sub_category(sub_category_output, sub_category_targets)
                loss = loss_category + loss_sub_category

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    def save_model(self):
        torch.save(self.model.state_dict(), 'attention_mlp_model.pth')
        print("Model saved successfully.")
    
    def save_encoders(self):
        with open('category_encoder.pkl', 'wb') as f:
            pickle.dump(self.category_encoder, f)
        with open('sub_category_encoder.pkl', 'wb') as f:
            pickle.dump(self.sub_category_encoder, f)
        print("Encoders saved successfully.")

# Example usage
pipeline = NLPAttentionPipeline(file_path='train.csv')
pipeline.train(epochs=10)  # Train for 10 epochs
pipeline.save_model()  # Save the trained model
