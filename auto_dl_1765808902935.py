import pandas as pd
import numpy as np
import os
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding, Input, MultiHeadAttention, Flatten
)

# --- HARD RULE: DO NOT PRINT LOGS OR COMMENTS OUTSIDE JSON ---
# Any internal print statements are for debugging during development and will be removed or commented out.
# The final script must only print the JSON object.

# Dataset Requirements
data_path = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'

# Check if dataset file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

df = pd.read_csv(data_path)

# Automatically detect text column
text_cols_candidates = ["Email Text", "text", "body", "message"]
text_col = None
for col in text_cols_candidates:
    if col in df.columns:
        text_col = col
        break
if text_col is None:
    raise ValueError(f"Could not find a valid text column. Looked for: {', '.join(text_cols_candidates)}")

# Automatically detect label column
label_cols_candidates = ["Email Type", "label", "labels", "target", "Category"]
label_col = None
for col in label_cols_candidates:
    if col in df.columns:
        label_col = col
        break
if label_col is None:
    raise ValueError(f"Could not find a valid label column. Looked for: {', '.join(label_cols_candidates)}")

# Drop rows with NaN in the detected text or label columns
df.dropna(subset=[text_col, label_col], inplace=True)

# Perform label mapping
phishing_variants = ["phishing", "phishing email", "spam", "malicious"]
safe_variants = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

def map_label(label):
    label_lower = str(label).lower()
    if label_lower in phishing_variants:
        return 1
    elif label_lower in safe_variants:
        return 0
    return np.nan # Mark for dropping if not in valid lists

df['mapped_label'] = df[label_col].apply(map_label)

# Drop rows where labels are outside the specified lists (mapped to NaN)
df.dropna(subset=['mapped_label'], inplace=True)

X = df[text_col].astype(str).tolist()
y = df['mapped_label'].astype(int).tolist()

# After mapping, dataset MUST contain both 0 and 1
if len(set(y)) < 2:
    raise ValueError("Dataset mapping failed: need both phishing and safe labels")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Text Processing - Keras Tokenizer
vocab_size = 10000  # Max number of words to keep
embedding_dim = 128  # Dimension of the dense embedding
max_len = 200      # Max length of sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post', truncating='post')

# Helper function to train and evaluate models
def train_and_evaluate_model(model_name_for_logs, model_fn, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    tf.keras.backend.clear_session() # Clear session for each model

    model = model_fn()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0 # Suppress verbose output
    )
    training_time = time.time() - start_time

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist() # Convert numpy array to list for JSON

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf_matrix,
        "train_accuracy_history": [float(val) for val in history.history['accuracy']],
        "validation_accuracy_history": [float(val) for val in history.history['val_accuracy']],
        "training_time": float(training_time)
    }
    return results

# Deep Learning Models to Train (Mandatory)

# 1. LSTM Model
def create_lstm_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    return model

# 2. BiLSTM Model
def create_bilstm_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64)), # Reduced units for BiLSTM due to more parameters
        Dense(1, activation='sigmoid')
    ])
    return model

# 3. GRU Model
def create_gru_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GRU(128),
        Dense(1, activation='sigmoid')
    ])
    return model

# 4. 1D CNN Model
def create_cnn_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    return model

# 5. Hybrid CNN + Dense Model
def create_hybrid_cnn_dense_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(), # Flatten to connect to dense layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# Additional Custom Model ("auto_model")
def create_auto_model():
    # Architecture: CNN for local features -> MultiHeadAttention for global dependencies -> Global Pooling -> Dense Classifier
    # This combines convolutional feature extraction with self-attention mechanism to capture
    # both local patterns and long-range dependencies efficiently.
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
    
    # 1D Convolutional layer to extract local features
    conv_out = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    
    # MultiHeadAttention layer to capture global dependencies
    # key_dim and value_dim should match the last dimension of the input tensor to attention (conv_out's filters)
    attention_out = MultiHeadAttention(num_heads=8, key_dim=128, value_dim=128)(conv_out, conv_out)
    
    # Global pooling to reduce dimensionality after attention
    pooled_out = GlobalMaxPooling1D()(attention_out)
    
    # Dense layers for classification
    dense_out = Dense(64, activation='relu')(pooled_out)
    dropout_out = Dropout(0.5)(dense_out)
    
    outputs = Dense(1, activation='sigmoid')(dropout_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

auto_model_description = (
    "Hybrid model featuring a 1D Convolutional layer to extract local features, "
    "followed by MultiHeadAttention for capturing global dependencies and interactions "
    "between these features, then a GlobalMaxPooling layer, and finally a dense classifier."
)

# Execute training and evaluation for all models
all_results = {}

all_results["lstm"] = train_and_evaluate_model(
    "LSTM", create_lstm_model, X_train_padded, y_train, X_test_padded, y_test
)
all_results["bilstm"] = train_and_evaluate_model(
    "BiLSTM", create_bilstm_model, X_train_padded, y_train, X_test_padded, y_test
)
all_results["gru"] = train_and_evaluate_model(
    "GRU", create_gru_model, X_train_padded, y_train, X_test_padded, y_test
)
all_results["cnn"] = train_and_evaluate_model(
    "1D CNN", create_cnn_model, X_train_padded, y_train, X_test_padded, y_test
)
all_results["hybrid"] = train_and_evaluate_model(
    "Hybrid CNN + Dense", create_hybrid_cnn_dense_model, X_train_padded, y_train, X_test_padded, y_test
)

auto_model_specific_results = train_and_evaluate_model(
    "auto_model", create_auto_model, X_train_padded, y_train, X_test_padded, y_test
)
auto_model_specific_results["description"] = auto_model_description
all_results["auto_model"] = auto_model_specific_results

# Final Output (ONLY JSON)
print(json.dumps(all_results, indent=None), flush=True)

