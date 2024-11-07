

# Cybercrime Classification Model

This repository provides a complete pipeline for training and evaluating a neural network model that classifies cybercrimes based on text descriptions. The pipeline includes the generation of tokenization files, creation of an embedding matrix, training of the model with attention layers, testing, and evaluation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technical Implementation](#technical-implementation)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Step 1: Tokenization](#step-1-tokenization)
   - [Step 2: Training the Embedding Matrix](#step-2-training-the-embedding-matrix)
   - [Step 3: Training the Model](#step-3-training-the-model)
   - [Step 4: Testing the Model](#step-4-testing-the-model)
   - [Step 5: Model Evaluation](#step-5-model-evaluation)

## Project Overview

This project aims to build a model that classifies cybercrimes based on text descriptions of criminal activities. The workflow includes:

1. **Tokenization**: Using SentencePiece to tokenize the text data and generate vocabulary files.
2. **Embedding Matrix**: Generating an embedding matrix from the tokenized text data to represent the words in a dense format.
3. **Model Training**: Training a neural network with attention layers using the embedding matrix, to classify the text descriptions into predefined cybercrime categories.
4. **Testing**: Running the trained model on a separate test dataset to evaluate its performance.
5. **Evaluation**: Evaluating the performance of the trained model using metrics such as accuracy, precision, recall, and F1-score.

The model is specifically designed to classify **cybercrimes** like UPI fraud, phishing, identity theft, online scams, and other related activities based on text descriptions. This can be applied to automated crime reporting systems, cybersecurity monitoring, law enforcement applications, or data analytics in cybercrime prevention.

## Technical Implementation

This section outlines the implementation of the text classification model using **SentencePiece** for tokenization and a **multi-head attention** layer for capturing contextual relationships within the text. The model is designed to efficiently process a wide range of languages and domains by breaking down text into subword units, handling out-of-vocabulary words, and leveraging a multi-head attention mechanism for improved classification accuracy.

### 1. **SentencePiece Tokenization**

**SentencePiece** is a language-independent subword tokenizer and detokenizer widely utilized in natural language processing (NLP) tasks. It is particularly beneficial in contexts where conventional word-based tokenization methods are inadequate, such as in languages without clear word boundaries or with highly diverse vocabularies. SentencePiece processes the input text as a continuous sequence of characters, resulting in more efficient and robust tokenization for downstream machine learning models.

#### Key Aspects of SentencePiece:
1. **Unsupervised Training**: 
   SentencePiece employs an unsupervised learning approach to tokenize text data without relying on predefined rules or language-specific heuristics. It learns to segment the text into meaningful subword units by analyzing the entire corpus as a continuous stream of characters.
   
2. **Subword Units**:
   Instead of splitting text at spaces, SentencePiece breaks down sentences into smaller, data-driven subword units. It keeps commonly used words whole while decomposing rare or complex words into frequently occurring subwords.
   
3. **Handling Out-of-Vocabulary (OOV) Words**:
   SentencePiece can represent any out-of-vocabulary word as a sequence of known subword tokens, making the model adaptable to new or rare words and improving its robustness.
   
4. **Language and Domain Adaptability**:
   SentencePiece is highly effective for languages with complex morphological structures or domain-specific applications, such as medical or legal text processing.

5. **Integration with Embedding Models**:
   SentencePiece integrates seamlessly with models like transformers, generating embeddings for subword tokens that capture semantic information. In this project, a custom embedding layer with a dimension of 100 is used to convert tokenized text into high-dimensional vectors suitable for further analysis and classification.

### 2. **Multi-Head Attention Layer**

The core of the model is a **multi-head attention layer**, which captures different aspects of contextual information in parallel. This is implemented using `nn.MultiheadAttention` in PyTorch with 8 attention heads.

#### Attention Mechanism Details:
- **Contextual Relationships**: The attention mechanism allows the model to focus on different parts of the input sequence simultaneously, capturing relationships between tokens. This is particularly useful for understanding long-range dependencies in text.
- **Parallel Attention**: The multi-head attention layer computes attention scores based on the input token embeddings, allowing the model to attend to various parts of the sentence in parallel, improving contextual understanding.

### 3. **Model Architecture**

- **Text Preprocessing & Tokenization**: 
  The input text is cleaned by removing non-alphanumeric characters and converting the text to lowercase. It is then tokenized using the pretrained SentencePiece model, breaking the text into subword units.

- **Embedding Layer**: 
  The model starts with an embedding layer that maps the tokenized input into dense vector representations. Each token is represented as a vector of size 96, which is the embedding dimension.

- **Multi-Head Attention Layer**:
  The multi-head attention layer uses 8 attention heads to capture contextual information and learn complex relationships between words in the sequence.

- **Global Pooling**:
  The output from the attention layer is aggregated using global pooling, where the mean of the embeddings is computed across the sequence dimension to produce a fixed-size representation of the entire text.

- **Fully Connected Layers**:
  The pooled embeddings are passed through two fully connected layers:
  - One for predicting the main category (e.g., the primary crime classification).
  - Another for predicting the sub-category (e.g., more specific crime types).

- **Output**: 
  Both layers output logits that are used for classification and are processed with cross-entropy loss during training.

### 4. **Training Process**

- The model is trained using **cross-entropy loss** for both category and sub-category tasks. The **Adam optimizer** is used for efficient model training, updating the parameters based on backpropagated error during training.

### 5. **Label Encoding**

- The target labels for both category and sub-category are encoded using **LabelEncoder** from Scikit-learn, converting categorical string labels into numerical representations for use in the neural network.

### 6. **Model Save & Inference**

- After training, the model's state dictionary is saved using `torch.save`, which allows for loading the trained model for future inference. Additionally, the label encoders are serialized using `pickle` for consistent label mapping during inference.

### 7. **Model Evaluation**

- Model performance can be evaluated by computing accuracy, precision, recall, and F1-score on a validation or test set.

## Potential Applications

This model can be applied across various industries:
- **Banking**: Classifying transactions for fraud detection or managing customer complaints related to cybercrimes.
- **Law Enforcement**: Categorizing reports or complaints based on urgency and crime type.
- **Healthcare**: Classifying medical records to prioritize cases or identifying fraud in health insurance claims.
- **E-commerce**: Classifying reviews and comments for sentiment analysis or fraud detection.
- **Education**: Grading essays or classifying academic content.

## Project Structure

The project structure is as follows:

```
/project
  ├── EmbeddingMatrixFinal.py           # Script to generate the embedding matrix
  ├── AttentionMatrixFinal.py           # Script to train the attention-based model
  ├── TestRunner.py                    # Script to test the trained model
  └── Evaluation.py                    # Script to evaluate the trained model
  ├── tokenizer.model                  # SentencePiece tokenizer model (generated)
  └── tokenizer.vocab                  # SentencePiece vocabulary (generated)
  └── train.csv                        # Training dataset with text descriptions of cybercrimes
  └── test.csv                         # Testing dataset with text descriptions of cybercrimes
```

## Installation

Ensure you have Python 3.x installed, and install the necessary dependencies using pip:

```bash
pip install torch pandas sentencepiece scikit-learn
```

## Usage

### Step 1: Tokenization

First, generate the SentencePiece tokenizer and vocabulary files:

```bash
python Tokenizer.py
```

### Step 2: Training the Embedding Matrix

Generate the embedding matrix:

```bash
python EmbeddingMatrixFinal.py
```

### Step 3: Training the Model

Train the model:

```bash
python AttentionMatrixFinal.py
```

### Step 4: Testing the Model

Test the trained model:

```bash
python TestRunner.py
```

### Step 5: Model Evaluation

Evaluate the model:

```bash
python Evaluation.py
```



