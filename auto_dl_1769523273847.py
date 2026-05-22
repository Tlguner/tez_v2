import pandas as pd
import numpy as np
import os
import time
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping

# --- Dataset Requirements ---

file_path = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file not found at: {file_path}")

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)
print(f"Initial dataset shape: {df.shape}")

# Automatically detect text column
text_column_candidates = ["text_combined", "Email Text", "email_text", "text", "body", "message", "content", "email", "mail", "raw_text", "sentence", "clean_text"]
text_col = None
for col in text_column_candidates:
    if col in df.columns:
        text_col = col
        break

if not text_col:
    # Fallback: choose the column with dtype object/string that has the largest average string length.
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    if object_cols.empty:
        raise ValueError("Could not find a text column (no object/string columns found).")

    max_avg_len = -1
    for col in object_cols:
        # Ensure column values are treated as strings for length calculation
        str_series = df[col].astype(str).replace('nan', '')
        avg_len = str_series.apply(len).mean()
        if avg_len > max_avg_len:
            max_avg_len = avg_len
            text_col = col

if not text_col:
    raise ValueError("Could not find a text column")

print(f"Detected text column: '{text_col}'")

# Automatically detect label column
label_column_candidates = ["label", "Email Type", "labels", "target", "Category", "class", "y", "is_phishing", "spam", "phish"]
label_col = None
for col in label_column_candidates:
    if col in df.columns:
        label_col = col
        break

if not label_col:
    # Fallback: choose a column with <= 10 unique values (after dropping NaNs) and not equal to the chosen text column.
    for col in df.columns:
        if col != text_col:
            # Drop NaNs before counting unique values
            if df[col].dropna().nunique() <= 10:
                label_col = col
                break

if not label_col:
    raise ValueError("Could not find a label column")

print(f"Detected label column: '{label_col}'")

# Perform label mapping
phishing_labels = ["phishing", "phishing email", "spam", "malicious"]
safe_labels = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

df['mapped_label'] = np.nan # Temporary column for mapped labels

if pd.api.types.is_numeric_dtype(df[label_col]):
    if not set(df[label_col].dropna().unique()).issubset({0, 1}):
        raise ValueError("Numeric label column must contain only 0 and 1")
    df['mapped_label'] = df[label_col].astype(int)
    print("Label column is numeric and contains only 0/1. Using as-is.")
else: # string/object dtype
    print("Label column is string/object. Performing mapping...")
    # Convert to lowercase for case-insensitive matching
    df['mapped_label'] = df[label_col].astype(str).str.lower().apply(
        lambda x: 1 if x in phishing_labels else (0 if x in safe_labels else np.nan)
    )

# Drop rows with unmappable labels (NaN in 'mapped_label')
initial_rows = len(df)
df.dropna(subset=['mapped_label'], inplace=True)
dropped_rows = initial_rows - len(df)
if dropped_rows > 0:
    print(f"Dropped {dropped_rows} rows with unmappable labels.")

# Fill NaN text with empty string ""
df[text_col] = df[text_col].fillna("")

# Ensure mapped_label is integer type
df['mapped_label'] = df['mapped_label'].astype(int)

# Dataset MUST contain both 0 and 1
if not (0 in df['mapped_label'].unique() and 1 in df['mapped_label'].unique()):
    raise ValueError("Dataset mapping failed: need both phishing and safe labels")

print(f"Final dataset shape after processing: {df.shape}")
print(f"Label distribution: \n{df['mapped_label'].value_counts()}")

X = df[text_col].values
y = df['mapped_label'].values

# --- Train/Test Split ---
print("\nPerforming stratified 80/20 train/test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# --- Text Processing ---
print("\nApplying Keras Tokenizer and padding sequences...")
max_len = 200
num_words = 10000 # Top 10,000 most frequent words

tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

vocab_size = len(tokenizer.word_index) + 1 # +1 for 0-padding
embedding_dim = 100 # Dimension for word embeddings

print(f"Vocabulary size: {vocab_size}")
print(f"Max sequence length: {max_len}")
print(f"Embedding dimension: {embedding_dim}")

# --- Deep Learning Models ---

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
results = {}

def train_and_evaluate_model(model, X_train_padded, y_train, X_test_padded, y_test, model_name, description=None):
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    
    # Use 10% of the training data as validation for early stopping
    history = model.fit(X_train_padded, y_train,
                        epochs=10,
                        validation_split=0.1,
                        callbacks=[early_stopping],
                        verbose=1)
    training_time = time.time() - start_time

    # Evaluate
    y_pred_probs = model.predict(X_test_padded, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"{model_name} Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Training Time: {training_time:.2f} seconds")

    model_result = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "train_accuracy_history": history.history.get('accuracy', []),
        "validation_accuracy_history": history.history.get('val_accuracy', []),
        "training_time": training_time
    }
    if description:
        model_result["description"] = description
    return model_result

# 1. LSTM
model_lstm = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results['lstm'] = train_and_evaluate_model(model_lstm, X_train_padded, y_train, X_test_padded, y_test, "LSTM")

# 2. BiLSTM
model_bilstm = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Bidirectional(LSTM(128)),
    Dense(1, activation='sigmoid')
])
model_bilstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results['bilstm'] = train_and_evaluate_model(model_bilstm, X_train_padded, y_train, X_test_padded, y_test, "BiLSTM")

# 3. GRU
model_gru = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GRU(128),
    Dense(1, activation='sigmoid')
])
model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results['gru'] = train_and_evaluate_model(model_gru, X_train_padded, y_train, X_test_padded, y_test, "GRU")

# 4. 1D CNN
model_cnn = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results['cnn'] = train_and_evaluate_model(model_cnn, X_train_padded, y_train, X_test_padded, y_test, "1D CNN")

# 5. Hybrid CNN + Dense
model_hybrid = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model_hybrid.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results['hybrid'] = train_and_evaluate_model(model_hybrid, X_train_padded, y_train, X_test_padded, y_test, "Hybrid CNN + Dense")

# --- Additional Custom Model ("auto_model") ---
# A hybrid model combining CNN for feature extraction and GRU for sequence processing.
# This architecture is different from the mandatory models by using GRU after CNN features.
model_auto = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    GRU(64, return_sequences=False), # GRU processes the output of CNN-pooled features
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model_auto.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
results['auto_model'] = train_and_evaluate_model(
    model_auto, X_train_padded, y_train, X_test_padded, y_test, "auto_model",
    description="Hybrid model: Embedding -> Conv1D (kernel=3) -> MaxPooling1D -> GRU (64 units) -> Dense (32 units) -> Dropout -> Dense (sigmoid). Combines CNN for local feature extraction with GRU for sequence understanding of these features."
)

# --- Final Output ---
print("\n<<<JSON_START>>>")
print(json.dumps(results, indent=2))
print("<<<JSON_END>>>")
