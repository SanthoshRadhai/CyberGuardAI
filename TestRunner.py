import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define the AttentionMLPModel class (if not imported from another file)
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


# Define the NLPAttentionPrediction class for inference
class NLPAttentionPrediction:
    def __init__(self, model_path, tokenizer_model_path, category_encoder_path, sub_category_encoder_path):
        self.model = AttentionMLPModel(vocab_size=32000, embedding_dim=96, num_category_classes=13, num_sub_category_classes=31)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load SentencePiece tokenizer
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        
        # Load LabelEncoders
        with open(category_encoder_path, 'rb') as f:
            self.category_encoder = pickle.load(f)
        with open(sub_category_encoder_path, 'rb') as f:
            self.sub_category_encoder = pickle.load(f)

    def predict(self, text):
        # Clean and tokenize the input text
        cleaned_text = self._clean_text(text)
        tokenized_text = self.sp.encode_as_ids(cleaned_text)
        
        # Pad the tokenized text to max_sequence_length (e.g., 500)
        tokenized_text = tokenized_text[:500] + [0] * (500 - len(tokenized_text))
        input_tensor = torch.tensor([tokenized_text], dtype=torch.long)

        # Get predictions
        with torch.no_grad():
            category_output, sub_category_output = self.model(input_tensor)
        
        # Get the predicted category and sub-category labels
        category_pred = torch.argmax(category_output, dim=1).item()
        sub_category_pred = torch.argmax(sub_category_output, dim=1).item()
        
        # Decode the predictions using the LabelEncoders
        category_label = self.category_encoder.inverse_transform([category_pred])[0]
        sub_category_label = self.sub_category_encoder.inverse_transform([sub_category_pred])[0]

        return category_label, sub_category_label

    def _clean_text(self, text):
        # Clean the text (e.g., remove punctuation, convert to lowercase)
        return text.strip().lower()


# Example usage
if __name__ == "__main__":
    # Paths to the saved model, tokenizer, and encoders
    model_path = 'attention_mlp_model.pth'
    tokenizer_model_path = 'tokenizer_model.model'
    category_encoder_path = 'category_encoder.pkl'
    sub_category_encoder_path = 'sub_category_encoder.pkl'

    # Initialize the prediction pipeline
    nlp_predictor = NLPAttentionPrediction(model_path, tokenizer_model_path, category_encoder_path, sub_category_encoder_path)
    
    # Sample text for prediction
    sample_text = "Example crime-related text goes here."
    
    # Get predictions
    category, sub_category = nlp_predictor.predict(sample_text)
    
    # Output the predictions
    print(f"Predicted Category: {category}")
    print(f"Predicted Sub-Category: {sub_category}")
