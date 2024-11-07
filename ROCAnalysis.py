import torch
import torch.nn as nn
import sentencepiece as spm
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

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
        self.model = AttentionMLPModel(vocab_size=32000, embedding_dim=96, num_category_classes=15, num_sub_category_classes=36)
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
        
        return category_output, sub_category_output

    def _clean_text(self, text):
        # Check if the text is a string and handle cases like NaN or non-string input
        if isinstance(text, str):
            return text.strip().lower()
        else:
            # If not a string, return an empty string or placeholder
            return ""

# Paths to the saved model, tokenizer, and encoders
model_path = 'attention_mlp_model.pth'
tokenizer_model_path = 'tokenizer_model.model'
category_encoder_path = 'category_encoder.pkl'
sub_category_encoder_path = 'sub_category_encoder.pkl'

# Load the dataset
csv_path = 'test.csv'
data = pd.read_csv(csv_path, nrows=29000)

# Initialize the prediction pipeline
nlp_predictor = NLPAttentionPrediction(model_path, tokenizer_model_path, category_encoder_path, sub_category_encoder_path)

# Perform predictions on the data
kf = KFold(n_splits=5, shuffle=True, random_state=42)
true_categories = []
predicted_category_probs = []

for train_index, test_index in kf.split(data):
    train_data, test_data = data.iloc[train_index], data.iloc[test_index]
    true_labels = []
    pred_probs = []

    for _, row in test_data.iterrows():
        crime_info = row['crimeaditionalinfo']
        category_output, _ = nlp_predictor.predict(crime_info)
        
        # Convert logits to probabilities using softmax
        category_prob = torch.softmax(category_output, dim=1).numpy()
        pred_probs.append(category_prob[0])  # Only one element as batch size is 1

        true_labels.append(row['category'])

    true_categories.extend(true_labels)
    predicted_category_probs.extend(pred_probs)

# Binarize the true labels
lb = LabelBinarizer()
true_categories_bin = lb.fit_transform(true_categories)

# Plot ROC curves for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(lb.classes_)):
    fpr[i], tpr[i], _ = roc_curve(true_categories_bin[:, i], np.array(predicted_category_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(14, 10))
for i in range(len(lb.classes_)):
    plt.plot(fpr[i], tpr[i], label=f"Class {lb.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()
