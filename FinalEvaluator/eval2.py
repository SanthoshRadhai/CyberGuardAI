import torch
import torch.nn as nn
import sentencepiece as spm
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 

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
        # Check if the text is a string and handle cases like NaN or non-string input
        if isinstance(text, str):
            return text.strip().lower()
        else:
            # If not a string, return an empty string or placeholder
            return ""


# Example usage for evaluation
if __name__ == "__main__":
    # Paths to the saved model, tokenizer, and encoders
    model_path = 'attention_mlp_model.pth'  # Update with your trained model path
    tokenizer_model_path = 'tokenizer_model.model'  # Update with your tokenizer model path
    category_encoder_path = 'category_encoder.pkl'  # Update with your category encoder path
    sub_category_encoder_path = 'sub_category_encoder.pkl'  # Update with your sub-category encoder path

    # Load the dataset (only first 1000 rows)
    csv_path = 'test.csv'  # Update with your actual test CSV file path
    data = pd.read_csv(csv_path, nrows=1000)

    # Initialize the prediction pipeline
    nlp_predictor = NLPAttentionPrediction(model_path, tokenizer_model_path, category_encoder_path, sub_category_encoder_path)
    
    # Perform predictions on the data
    predicted_categories = []
    predicted_sub_categories = []

    for index, row in data.iterrows():
        crime_info = row['crimeaditionalinfo']
        
        # Predict category and sub-category
        category_pred, sub_category_pred = nlp_predictor.predict(crime_info)
        
        predicted_categories.append(category_pred)
        predicted_sub_categories.append(sub_category_pred)

    # Ensure true labels are consistent and clean (e.g., as strings)
    true_categories = list(map(str, data['category'].values))
    true_sub_categories = list(map(str, data['sub_category'].values))

    # Ensure predicted labels are consistent and clean (e.g., as strings)
    predicted_categories = list(map(str, predicted_categories))
    predicted_sub_categories = list(map(str, predicted_sub_categories))

    # Calculate accuracy
    category_accuracy = accuracy_score(true_categories, predicted_categories)
    sub_category_accuracy = accuracy_score(true_sub_categories, predicted_sub_categories)

    print(f"Category Accuracy: {category_accuracy}")
    print(f"Sub-category Accuracy: {sub_category_accuracy}")

    # Optionally, generate a classification report for more details
    category_report = classification_report(true_categories, predicted_categories)
    sub_category_report = classification_report(true_sub_categories, predicted_sub_categories)

    print("Category Classification Report:")
    print(category_report)

    print("Sub-category Classification Report:")
    print(sub_category_report)

    # Plotting confusion matrices
    # Category Confusion Matrix
    category_cm = confusion_matrix(true_categories, predicted_categories)
    plt.figure(figsize=(10, 7))
    sns.heatmap(category_cm, annot=True, fmt='d', cmap='Blues', xticklabels=nlp_predictor.category_encoder.classes_,
                yticklabels=nlp_predictor.category_encoder.classes_)
    plt.title("Category Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Sub-category Confusion Matrix
    sub_category_cm = confusion_matrix(true_sub_categories, predicted_sub_categories)
    plt.figure(figsize=(10, 7))
    sns.heatmap(sub_category_cm, annot=True, fmt='d', cmap='Blues', xticklabels=nlp_predictor.sub_category_encoder.classes_,
                yticklabels=nlp_predictor.sub_category_encoder.classes_)
    plt.title("Sub-category Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
