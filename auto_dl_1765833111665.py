import pandas as pd
import numpy as np
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping

# --- Dataset Requirements ---
dataset_path = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Automatically detect text and label columns
text_cols_candidates = ["Email Text", "text", "body", "message"]
label_cols_candidates = ["Email Type", "label", "labels", "target", "Category"]

text_column = None
for col in text_cols_candidates:
    if col in df.columns:
        text_column = col
        break
if text_column is None:
    raise ValueError(f"No text column found from candidates: {text_cols_candidates}")

label_column = None
for col in label_cols_candidates:
    if col in df.columns:
        label_column = col
        break
if label_column is None:
    raise ValueError(f"No label column found from candidates: {label_cols_candidates}")

# Drop rows with NaN text or label immediately
df.dropna(subset=[text_column, label_column], inplace=True)

# Perform label mapping
phishing_variants = ["phishing", "phishing email", "spam", "malicious"]
safe_variants = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

def map_label(label):
    label_lower = str(label).lower()
    if label_lower in phishing_variants:
        return 1
    elif label_lower in safe_variants:
        return 0
    return np.nan # Mark for dropping

df['mapped_label'] = df[label_column].apply(map_label)

# Drop rows where mapping resulted in NaN (labels outside specified lists)
df.dropna(subset=['mapped_label'], inplace=True)

# Convert mapped_label to int
df['mapped_label'] = df['mapped_label'].astype(int)

# Validate dataset contains both 0 and 1
if not (0 in df['mapped_label'].unique() and 1 in df['mapped_label'].unique()):
    raise ValueError("Dataset mapping failed: need both phishing and safe labels")

X = df[text_column].values
y = df['mapped_label'].values

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Text Processing ---
max_len = 200
embedding_dim = 128
oov_tok = "<unk>"
padding_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(oov_token=oov_tok) # num_words is not set here to ensure all words are tokenized initially
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1 # Actual vocab size based on training data

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

# --- Model Training and Evaluation Function ---
def train_and_evaluate_model(model_name, model, X_train_p, y_train_p, X_test_p, y_test_p, callbacks_list):
    start_time = time.time()
    history = model.fit(
        X_train_p, y_train_p,
        epochs=10, # Hard limit as per requirements
        validation_data=(X_test_p, y_test_p),
        callbacks=callbacks_list,
        verbose=0 # Suppress training output
    )
    training_time = time.time() - start_time

    # Predict probabilities, then convert to binary predictions
    y_pred_probs = model.predict(X_test_p, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten() # Ensure 1D array for metrics

    # Calculate metrics
    accuracy = accuracy_score(y_test_p, y_pred)
    precision = precision_score(y_test_p, y_pred)
    recall = recall_score(y_test_p, y_pred)
    f1 = f1_score(y_test_p, y_pred)
    cm = confusion_matrix(y_test_p, y_pred).tolist() # Convert to list for JSON

    # Extract history (ensure keys exist, as EarlyStopping might cut short)
    train_accuracy_history = history.history.get('accuracy', [])
    val_accuracy_history = history.history.get('val_accuracy', [])

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "train_accuracy_history": [float(x) for x in train_accuracy_history],
        "validation_accuracy_history": [float(x) for x in val_accuracy_history],
        "training_time": float(training_time)
    }

# EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=0
)
callbacks = [early_stopping_callback]

results = {}

# --- Deep Learning Models to Train ---

# 1. LSTM Model
lstm_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results["lstm"] = train_and_evaluate_model("LSTM", lstm_model, X_train_padded, y_train, X_test_padded, y_test, callbacks)

# 2. BiLSTM Model
bilstm_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Bidirectional(LSTM(128)),
    Dense(1, activation='sigmoid')
])
bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results["bilstm"] = train_and_evaluate_model("BiLSTM", bilstm_model, X_train_padded, y_train, X_test_padded, y_test, callbacks)

# 3. GRU Model
gru_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GRU(128),
    Dense(1, activation='sigmoid')
])
gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results["gru"] = train_and_evaluate_model("GRU", gru_model, X_train_padded, y_train, X_test_padded, y_test, callbacks)

# 4. 1D CNN Model
cnn_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results["cnn"] = train_and_evaluate_model("CNN", cnn_model, X_train_padded, y_train, X_test_padded, y_test, callbacks)

# 5. Hybrid CNN + Dense Model
hybrid_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
hybrid_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results["hybrid"] = train_and_evaluate_model("Hybrid", hybrid_model, X_train_padded, y_train, X_test_padded, y_test, callbacks)

# --- Additional Custom Model ("auto_model") ---
auto_model_description = (
    "A custom model combining a 1D Convolutional layer for local feature extraction, "
    "followed by MaxPooling to reduce dimensionality, and then a Bidirectional GRU layer "
    "to capture sequential context from both directions, ending with a Dense output layer."
)
auto_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(GRU(64, return_sequences=False)),
    Dense(1, activation='sigmoid')
])
auto_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
auto_model_metrics = train_and_evaluate_model("auto_model", auto_model, X_train_padded, y_train, X_test_padded, y_test, callbacks)
results["auto_model"] = {
    "description": auto_model_description,
    **auto_model_metrics
}

# --- Final Output ---
print("<<<JSON_START>>>")
print(json.dumps(results, indent=2))
print("<<<JSON_END>>>")
