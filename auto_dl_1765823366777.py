import pandas as pd
import numpy as np
import json
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding, Flatten,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping

# --- HARD RULE: Keras Imports Required ---
# The above explicit imports satisfy this requirement.

# --- Dataset Requirements ---
DATASET_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'

# Automatically detect text and label columns
TEXT_COL_CANDIDATES = ["Email Text", "text", "body", "message"]
LABEL_COL_CANDIDATES = ["Email Type", "label", "labels", "target", "Category"]

# Label mapping
PHISHING_VARIANTS = ["phishing", "phishing email", "spam", "malicious"]
SAFE_VARIANTS = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

# --- CRITICAL TRAINING LIMITS ---
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2

def load_and_preprocess_data(path):
    """
    Loads data, detects columns, maps labels, and handles NaNs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)

    # Detect text column
    text_col = None
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"No text column found from candidates: {TEXT_COL_CANDIDATES}")

    # Detect label column
    label_col = None
    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No label column found from candidates: {LABEL_COL_CANDIDATES}")

    # Drop rows with NaN in detected text or label columns
    df.dropna(subset=[text_col, label_col], inplace=True)

    # Convert label column to lowercase for case-insensitive matching
    df[label_col] = df[label_col].astype(str).str.lower()

    # Apply label mapping
    def map_label(label):
        if label in PHISHING_VARIANTS:
            return 1  # Phishing
        elif label in SAFE_VARIANTS:
            return 0  # Ham
        else:
            return np.nan # Mark for dropping

    df['mapped_label'] = df[label_col].apply(map_label)

    # Drop rows with labels outside the specified lists
    df.dropna(subset=['mapped_label'], inplace=True)

    # Convert mapped_label to int
    df['mapped_label'] = df['mapped_label'].astype(int)

    # Validate that both 0 and 1 are present
    if not (df['mapped_label'].isin([0, 1]).all() and 0 in df['mapped_label'].unique() and 1 in df['mapped_label'].unique()):
        raise ValueError("Dataset mapping failed: need both phishing and safe labels")

    return df[text_col], df['mapped_label']

def train_and_evaluate_model(model, X_train_seq, y_train, X_test_seq, y_test, model_name, batch_size=32):
    """
    Trains a given Keras model and evaluates it, returning required metrics.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        X_train_seq, y_train,
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        validation_data=(X_test_seq, y_test),
        callbacks=[early_stopping],
        verbose=0 # Suppress verbose output
    )
    end_time = time.time()
    training_time = end_time - start_time

    y_pred_probs = model.predict(X_test_seq, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "train_accuracy_history": history.history['accuracy'],
        "validation_accuracy_history": history.history['val_accuracy'],
        "training_time": training_time
    }
    return results

# --- Main Script Execution ---
if __name__ == "__main__":
    results = {}

    try:
        # Load and preprocess data
        texts, labels = load_and_preprocess_data(DATASET_PATH)

        # Train/Test Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Text Processing (Keras Tokenizer)
        num_words = 10000 # Max number of words to keep
        max_len = 200     # Pad sequences to this length
        embedding_dim = 128 # Dimension for word embeddings

        tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
        tokenizer.fit_on_texts(X_train_raw)

        X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
        X_test_seq = tokenizer.texts_to_sequences(X_test_raw)

        X_train_seq = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
        X_test_seq = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

        vocab_size = len(tokenizer.word_index) + 1 # +1 for padding (index 0)

        # --- Deep Learning Models to Train ---

        # 1. LSTM Model
        lstm_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            LSTM(128),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        results['lstm'] = train_and_evaluate_model(lstm_model, X_train_seq, y_train, X_test_seq, y_test, "LSTM")

        # 2. BiLSTM Model
        bilstm_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            Bidirectional(LSTM(64)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        results['bilstm'] = train_and_evaluate_model(bilstm_model, X_train_seq, y_train, X_test_seq, y_test, "BiLSTM")

        # 3. GRU Model
        gru_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            GRU(128),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        results['gru'] = train_and_evaluate_model(gru_model, X_train_seq, y_train, X_test_seq, y_test, "GRU")

        # 4. 1D CNN Model
        cnn_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            GlobalMaxPooling1D(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        results['cnn'] = train_and_evaluate_model(cnn_model, X_train_seq, y_train, X_test_seq, y_test, "1D CNN")

        # 5. Hybrid CNN + Dense Model
        hybrid_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(), # Flatten for Dense layers
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        results['hybrid'] = train_and_evaluate_model(hybrid_model, X_train_seq, y_train, X_test_seq, y_test, "Hybrid CNN + Dense")

        # --- Additional Custom Model ("auto_model") ---
        # Architecture: Embedding -> Conv1D -> GRU (return sequences) -> MultiHeadAttention -> GlobalMaxPooling1D -> Dense -> Output
        auto_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            MultiHeadAttention(num_heads=4, key_dim=64), # Query, Key, Value from GRU output
            GlobalMaxPooling1D(), # Summarize the attention output
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        auto_model_description = (
            "A custom model featuring an Embedding layer, followed by a Conv1D layer for local feature extraction. "
            "A GRU layer processes the sequence while retaining sequence information. "
            "MultiHeadAttention then processes the GRU output to focus on relevant parts. "
            "GlobalMaxPooling1D summarizes the attention output, followed by a Dense layer and a sigmoid output."
        )
        auto_model_metrics = train_and_evaluate_model(auto_model, X_train_seq, y_train, X_test_seq, y_test, "auto_model")
        auto_model_metrics["description"] = auto_model_description
        results['auto_model'] = auto_model_metrics

    except FileNotFoundError as e:
        print("<<<JSON_START>>>")
        print(json.dumps({"error": str(e)}, indent=2))
        print("<<<JSON_END>>>")
        exit()
    except ValueError as e:
        print("<<<JSON_START>>>")
        print(json.dumps({"error": str(e)}, indent=2))
        print("<<<JSON_END>>>")
        exit()
    except Exception as e:
        print("<<<JSON_START>>>")
        print(json.dumps({"error": f"An unexpected error occurred: {str(e)}"}, indent=2))
        print("<<<JSON_END>>>")
        exit()

    # Final Output
    print("<<<JSON_START>>>")
    print(json.dumps(results, indent=2))
    print("<<<JSON_END>>>")
