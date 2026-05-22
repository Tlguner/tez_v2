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
    GlobalMaxPooling1D, Dense, Dropout, Embedding, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping

# Define constants
DATASET_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/test_dataset/phishing_email.csv'
TEXT_COL_CANDIDATES = ["Email Text", "text", "body", "message"]
LABEL_COL_CANDIDATES = ["Email Type", "label", "labels", "target", "Category"]
PHISHING_LABELS = ["phishing", "phishing email", "spam", "malicious"]
SAFE_LABELS = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]
MAX_LEN = 200
EMBEDDING_DIM = 128
MAX_VOCAB_SIZE = 10000
RANDOM_SEED = 42

print("Deep Learning Model Training & Evaluation Script")
print("------------------------------------------------")

# --- Dataset Loading and Preprocessing ---
print("\n--- Step 1: Loading and Preprocessing Dataset ---")

def load_and_preprocess_data(path):
    """
    Loads data from the specified path, detects text and label columns,
    performs label mapping, and handles NaNs and invalid labels.
    """
    if not os.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)
    print(f"Original dataset loaded with {len(df)} rows.")

    # Detect text column
    text_col = None
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            text_col = col
            break
    if not text_col:
        raise ValueError(f"No valid text column found. Tried: {TEXT_COL_CANDIDATES}")
    print(f"Detected text column: '{text_col}'")

    # Detect label column
    label_col = None
    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            label_col = col
            break
    if not label_col:
        raise ValueError(f"No valid label column found. Tried: {LABEL_COL_CANDIDATES}")
    print(f"Detected label column: '{label_col}'")

    df = df[[text_col, label_col]].copy()
    df.columns = ['text', 'label']

    # Drop rows with NaN in text or label
    initial_rows = len(df)
    df.dropna(subset=['text', 'label'], inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to NaN values in 'text' or 'label'.")

    # Perform label mapping
    phishing_variants_lower = [l.lower() for l in PHISHING_LABELS]
    safe_variants_lower = [l.lower() for l in SAFE_LABELS]

    def map_label(label):
        label_lower = str(label).lower()
        if label_lower in phishing_variants_lower:
            return 1 # Phishing/spam
        elif label_lower in safe_variants_lower:
            return 0 # Ham/safe
        return np.nan # Mark for dropping

    df['mapped_label'] = df['label'].apply(map_label)
    
    initial_rows_after_nan_drop = len(df)
    df.dropna(subset=['mapped_label'], inplace=True)
    if len(df) < initial_rows_after_nan_drop:
        print(f"Dropped {initial_rows_after_nan_drop - len(df)} rows due to labels outside the valid lists.")
    
    df['mapped_label'] = df['mapped_label'].astype(int)

    # Check for both classes
    if df['mapped_label'].nunique() < 2:
        raise ValueError("Dataset mapping failed: need both phishing and safe labels (0 and 1).")

    print(f"Dataset successfully preprocessed with {len(df)} rows.")
    print("Class distribution after mapping:")
    print(df['mapped_label'].value_counts())
    return df['text'].tolist(), df['mapped_label'].tolist()

all_results = {}
data_loaded_successfully = False

try:
    X, y = load_and_preprocess_data(DATASET_PATH)
    data_loaded_successfully = True
except (FileNotFoundError, ValueError) as e:
    error_message = f"Error during data loading: {e}"
    print(error_message)
    all_results['error'] = error_message
    # Do not exit here; proceed to print JSON with error info

if data_loaded_successfully:
    # --- Train/Test Split ---
    print("\n--- Step 2: Performing Train/Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # --- Text Processing with Keras Tokenizer ---
    print("\n--- Step 3: Text Processing (Keras Tokenizer) ---")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(X_train)

    word_index = tokenizer.word_index
    # +1 for 0-padding, cap at MAX_VOCAB_SIZE
    vocab_size = min(len(word_index) + 1, MAX_VOCAB_SIZE)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Padded training sequences shape: {X_train_padded.shape}")
    print(f"Padded testing sequences shape: {X_test_padded.shape}")


    # --- Model Training and Evaluation Function ---
    print("\n--- Step 4: Defining Model Training and Evaluation Function ---")
    def train_evaluate_model(model_name, build_model_func, X_train, y_train, X_test, y_test, vocab_size, embedding_dim, max_len):
        """
        Builds, compiles, trains, and evaluates a deep learning model.
        Records metrics and training history.
        """
        print(f"\n--- Training {model_name} ---")
        tf.keras.backend.clear_session() # Clear prior models
        model = build_model_func(vocab_size, embedding_dim, max_len)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Optional: print model summary for mandatory models only
        if model_name in ["LSTM", "BiLSTM", "GRU", "1D CNN", "Hybrid CNN + Dense", "auto_model"]:
            model.summary()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        )

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=10, # Hard limit
            batch_size=32,
            validation_split=0.2, # Use a split from X_train for validation
            callbacks=[early_stopping],
            verbose=1
        )
        training_time = time.time() - start_time

        print(f"--- Evaluating {model_name} ---")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)

        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON

        results = {
            "accuracy": float(accuracy),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "train_accuracy_history": [float(val) for val in history.history['accuracy']],
            "validation_accuracy_history": [float(val) for val in history.history['val_accuracy']],
            "training_time_seconds": float(training_time)
        }
        print(f"{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Training Time: {training_time:.2f} seconds")
        return results

    # --- Model Architectures ---
    print("\n--- Step 5: Defining Model Architectures ---")

    def build_lstm_model(vocab_size, embedding_dim, max_len):
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            LSTM(128),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_bilstm_model(vocab_size, embedding_dim, max_len):
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Bidirectional(LSTM(128)),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_gru_model(vocab_size, embedding_dim, max_len):
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            GRU(128),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_cnn_model(vocab_size, embedding_dim, max_len):
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_hybrid_cnn_dense_model(vocab_size, embedding_dim, max_len):
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_auto_model(vocab_size, embedding_dim, max_len):
        """
        AI-designed custom model: Stacked Conv1D layers with MaxPooling,
        followed by a GRU layer, then Dense layers.
        """
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            GRU(units=64, return_sequences=False), # GRU processes the sequence output of CNN before flattening
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model

    # --- Train and Evaluate All Models ---
    print("\n--- Step 6: Training and Evaluating Models ---")

    # Mandatory models
    all_results['lstm'] = train_evaluate_model("LSTM", build_lstm_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size, EMBEDDING_DIM, MAX_LEN)
    all_results['bilstm'] = train_evaluate_model("BiLSTM", build_bilstm_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size, EMBEDDING_DIM, MAX_LEN)
    all_results['gru'] = train_evaluate_model("GRU", build_gru_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size, EMBEDDING_DIM, MAX_LEN)
    all_results['cnn'] = train_evaluate_model("1D CNN", build_cnn_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size, EMBEDDING_DIM, MAX_LEN)
    all_results['hybrid'] = train_evaluate_model("Hybrid CNN + Dense", build_hybrid_cnn_dense_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size, EMBEDDING_DIM, MAX_LEN)

    # Custom model
    auto_model_description = "A custom model with an Embedding layer, followed by two stacked Conv1D layers (kernel_size=5 and kernel_size=3) with MaxPooling, then a GRU layer (64 units), a Dense layer (64 units, ReLU), Dropout (0.5), and a final sigmoid output Dense layer."
    auto_model_results = train_evaluate_model("auto_model", build_auto_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size, EMBEDDING_DIM, MAX_LEN)
    auto_model_results['description'] = auto_model_description
    all_results['auto_model'] = auto_model_results

# --- Final Output ---
print("\n--- Step 7: Generating JSON Output ---")
print("<<<JSON_START>>>")
print(json.dumps(all_results, indent=2))
print("<<<JSON_END>>>")