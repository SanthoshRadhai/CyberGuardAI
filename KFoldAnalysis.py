import torch
import torch.nn as nn
import sentencepiece as spm
import pickle
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your model class and prediction class here
class AttentionMLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_category_classes, num_sub_category_classes):
        super(AttentionMLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.fc_category = nn.Linear(embedding_dim, num_category_classes)
        self.fc_sub_category = nn.Linear(embedding_dim, num_sub_category_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        attn_output, _ = self.attention_layer(embedded, embedded, embedded)
        pooled_output = torch.mean(attn_output, dim=1)
        category_output = self.fc_category(pooled_output)
        sub_category_output = self.fc_sub_category(pooled_output)
        return category_output, sub_category_output

class NLPAttentionPrediction:
    def __init__(self, model_path, tokenizer_model_path, category_encoder_path, sub_category_encoder_path):
        self.model = AttentionMLPModel(vocab_size=32000, embedding_dim=96, num_category_classes=15, num_sub_category_classes=36)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        with open(category_encoder_path, 'rb') as f:
            self.category_encoder = pickle.load(f)
        with open(sub_category_encoder_path, 'rb') as f:
            self.sub_category_encoder = pickle.load(f)

    def predict(self, text):
        cleaned_text = self._clean_text(text)
        tokenized_text = self.sp.encode_as_ids(cleaned_text)
        tokenized_text = tokenized_text[:500] + [0] * (500 - len(tokenized_text))
        input_tensor = torch.tensor([tokenized_text], dtype=torch.long)
        with torch.no_grad():
            category_output, sub_category_output = self.model(input_tensor)
        category_pred = torch.argmax(category_output, dim=1).item()
        sub_category_pred = torch.argmax(sub_category_output, dim=1).item()
        category_label = self.category_encoder.inverse_transform([category_pred])[0]
        sub_category_label = self.sub_category_encoder.inverse_transform([sub_category_pred])[0]
        return category_label, sub_category_label

    def _clean_text(self, text):
        if isinstance(text, str):
            return text.strip().lower()
        else:
            return ""

# Load dataset
csv_path = 'test.csv'
data = pd.read_csv(csv_path, nrows=29000)

# Initialize prediction pipeline
model_path = 'attention_mlp_model.pth'
tokenizer_model_path = 'tokenizer_model.model'
category_encoder_path = 'category_encoder.pkl'
sub_category_encoder_path = 'sub_category_encoder.pkl'
nlp_predictor = NLPAttentionPrediction(model_path, tokenizer_model_path, category_encoder_path, sub_category_encoder_path)

# 5-Fold Cross-Validation
kf = KFold(n_splits=5)
category_accuracies = []
sub_category_accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    true_categories = []
    true_sub_categories = []
    predicted_categories = []
    predicted_sub_categories = []

    for index, row in test_data.iterrows():
        crime_info = row['crimeaditionalinfo']
        true_categories.append(str(row['category']))
        true_sub_categories.append(str(row['sub_category']))

        category_pred, sub_category_pred = nlp_predictor.predict(crime_info)
        predicted_categories.append(str(category_pred))
        predicted_sub_categories.append(str(sub_category_pred))

    # Calculate accuracy for the current fold
    category_accuracy = accuracy_score(true_categories, predicted_categories)
    sub_category_accuracy = accuracy_score(true_sub_categories, predicted_sub_categories)
    category_accuracies.append(category_accuracy)
    sub_category_accuracies.append(sub_category_accuracy)

    print(f"Fold {fold + 1}: Category Accuracy = {category_accuracy}, Sub-category Accuracy = {sub_category_accuracy}")

# Average accuracies over all folds
average_category_accuracy = sum(category_accuracies) / len(category_accuracies)
average_sub_category_accuracy = sum(sub_category_accuracies) / len(sub_category_accuracies)

print(f"Average Category Accuracy: {average_category_accuracy}")
print(f"Average Sub-category Accuracy: {average_sub_category_accuracy}")

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), category_accuracies, marker='o', label='Category Accuracy', color='blue')
plt.plot(range(1, 6), sub_category_accuracies, marker='o', label='Sub-category Accuracy', color='orange')
plt.axhline(average_category_accuracy, color='blue', linestyle='--', label='Avg Category Accuracy')
plt.axhline(average_sub_category_accuracy, color='orange', linestyle='--', label='Avg Sub-category Accuracy')
plt.title("K-Fold Cross-Validation Accuracies")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the plot
plt.savefig('kfold_accuracies.png')
plt.show()
