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

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- Configuration Constants ---
DATASET_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/test_dataset/phishing_email.csv'
MAX_LEN = 200
VOCAB_SIZE = 10000  # Maximum number of words to keep in the vocabulary
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 32

# --- Dataset Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    """
    Loads data from the specified CSV path, detects text and label columns,
    performs label mapping, and handles NaNs according to requirements.
    """
    print(f"Loading dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    print(f"Original dataset shape: {df.shape}")

    # 1. Automatically detect text column
    text_column_candidates = ["text_combined", "Email Text", "email_text", "text", "body", "message", "content", "email", "mail", "raw_text", "sentence", "clean_text"]
    text_col = None
    for col in text_column_candidates:
        if col in df.columns:
            text_col = col
            print(f"Detected text column by preferred name: '{text_col}'")
            break

    if text_col is None:
        # Fallback: choose the column with dtype object/string that has the largest average string length.
        object_cols = df.select_dtypes(include=['object', 'string']).columns
        if not object_cols.empty:
            # Ensure all columns are treated as strings for length calculation
            avg_lengths = {col: df[col].astype(str).apply(len).mean() for col in object_cols}
            text_col = max(avg_lengths, key=avg_lengths.get)
            print(f"Fallback: Detected text column by largest average string length: '{text_col}'")
        else:
            raise ValueError("Could not find a text column (no object/string columns found).")

    # 2. Automatically detect label column
    label_column_candidates = ["label", "Email Type", "labels", "target", "Category", "class", "y", "is_phishing", "spam", "phish"]
    label_col = None
    for col in label_column_candidates:
        if col in df.columns:
            label_col = col
            print(f"Detected label column by preferred name: '{label_col}'")
            break

    if label_col is None:
        # Fallback: choose a column with <= 10 unique values (after dropping NaNs) and not equal to the chosen text column.
        candidate_label_cols = [c for c in df.columns if c != text_col]
        for col in candidate_label_cols:
            if df[col].nunique(dropna=True) <= 10:
                label_col = col
                print(f"Fallback: Detected label column by unique value count (<=10): '{label_col}'")
                break
        if label_col is None:
            raise ValueError("Could not find a label column.")

    # Drop rows with NaN label only
    original_rows_count = df.shape[0]
    df.dropna(subset=[label_col], inplace=True)
    if df.shape[0] < original_rows_count:
        print(f"Dropped {original_rows_count - df.shape[0]} rows due to NaN in label column '{label_col}'.")

    # Fill NaN text with empty string ""
    df[text_col] = df[text_col].fillna("")
    print(f"Filled NaN in text column '{text_col}' with empty strings.")

    # 3. Perform label mapping
    labels_series = df[label_col]
    df['mapped_label'] = None # Initialize mapped_label column

    if pd.api.types.is_numeric_dtype(labels_series):
        if set(labels_series.unique()) <= {0, 1}:
            print("Numeric label column contains only 0 and 1. Using labels as-is (0 = safe, 1 = phishing).")
            df['mapped_label'] = labels_series.astype(int)
        else:
            raise ValueError("Numeric label column must contain only 0 and 1.")
    else: # String/object dtype
        print("Mapping string/object label column to numeric (0=safe, 1=phishing).")
        phishing_variants = ["phishing", "phishing email", "spam", "malicious"]
        safe_variants = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

        label_map = {}
        for p_v in phishing_variants:
            label_map[p_v.lower()] = 1
        for s_v in safe_variants:
            label_map[s_v.lower()] = 0

        df['mapped_label'] = df[label_col].astype(str).str.lower().map(label_map)

        # Drop rows with unmappable labels
        original_rows_count = df.shape[0]
        df.dropna(subset=['mapped_label'], inplace=True)
        if df.shape[0] < original_rows_count:
            print(f"Dropped {original_rows_count - df.shape[0]} rows with unmappable string labels.")
        df['mapped_label'] = df['mapped_label'].astype(int)

    # After mapping: Dataset MUST contain both 0 and 1
    if not (0 in df['mapped_label'].unique() and 1 in df['mapped_label'].unique()):
        raise ValueError("Dataset mapping failed: need both phishing (1) and safe (0) labels after processing.")

    texts = df[text_col].tolist()
    labels = df['mapped_label'].values

    print(f"Final dataset size: {len(texts)} samples.")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return texts, labels

# --- Model Training and Evaluation Function ---
def train_and_evaluate_model(model_name, model_builder_func, X_train, y_train, X_test, y_test,
                             vocab_size, embedding_dim, max_len, epochs, batch_size):
    """
    Builds, trains, and evaluates a deep learning model.
    """
    print(f"\n--- Training {model_name} ---")
    model = model_builder_func(vocab_size, embedding_dim, max_len)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Optional: print model summary for debugging
    # model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    start_time = time.time()
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2, # 20% of the training data used for validation
                        callbacks=[early_stopping],
                        verbose=1)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"{model_name} training time: {training_time:.2f} seconds")

    # Evaluate on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON output

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix,
        "train_accuracy_history": history.history['accuracy'],
        "validation_accuracy_history": history.history['val_accuracy'],
        "training_time": training_time
    }
    print(f"{model_name} - Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return results

# --- Model Architectures ---

def build_lstm_model(vocab_size, embedding_dim, max_len):
    """ Builds a simple LSTM model. """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_bilstm_model(vocab_size, embedding_dim, max_len):
    """ Builds a Bidirectional LSTM model. """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128)),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gru_model(vocab_size, embedding_dim, max_len):
    """ Builds a simple GRU model. """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GRU(128),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_cnn_model(vocab_size, embedding_dim, max_len):
    """ Builds a 1D Convolutional Neural Network (CNN) model. """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(), # Reduces dimension by taking max over time step
        Dense(1, activation='sigmoid')
    ])
    return model

def build_hybrid_cnn_dense_model(vocab_size, embedding_dim, max_len):
    """ Builds a hybrid CNN + Dense model. """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=4), # Downsample after CNN
        Flatten(), # Flatten for Dense layers
        Dense(64, activation='relu'),
        Dropout(0.5), # Regularization
        Dense(1, activation='sigmoid')
    ])
    return model

def build_auto_model(vocab_size, embedding_dim, max_len):
    """
    Custom model: A hybrid architecture combining Conv1D, MaxPooling1D, GRU,
    and Dense layers with Dropout for robust feature extraction and sequence processing.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'), # Local feature extraction
        MaxPooling1D(pool_size=2), # Spatial downsampling
        GRU(64, return_sequences=False), # Process sequence, output single vector
        Dense(64, activation='relu'), # Further processing
        Dropout(0.5), # Regularization to prevent overfitting
        Dense(1, activation='sigmoid') # Output layer for binary classification
    ])
    return model

# --- Main Execution ---
def main():
    final_results = {}
    try:
        # Load and preprocess data
        texts, labels = load_and_preprocess_data(DATASET_PATH)

        # Train/Test Split (stratified 80/20)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        print(f"\nTrain samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")
        print(f"Train label distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Test label distribution: {pd.Series(y_test).value_counts().to_dict()}")

        # Text Processing using Keras Tokenizer
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
        tokenizer.fit_on_texts(X_train_raw)

        X_train_sequences = tokenizer.texts_to_sequences(X_train_raw)
        X_test_sequences = tokenizer.texts_to_sequences(X_test_raw)

        # Pad sequences to MAX_LEN
        X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

        print(f"\nPadded training data shape: {X_train_padded.shape}")
        print(f"Padded test data shape: {X_test_padded.shape}")
        print(f"Actual vocabulary size (after tokenizer fit): {len(tokenizer.word_index) + 1} (limited to {VOCAB_SIZE} most frequent words)")

        # Mandatory Models to Train
        model_configs = {
            "lstm": build_lstm_model,
            "bilstm": build_bilstm_model,
            "gru": build_gru_model,
            "cnn": build_cnn_model,
            "hybrid": build_hybrid_cnn_dense_model,
        }

        for model_name, builder_func in model_configs.items():
            results = train_and_evaluate_model(
                model_name, builder_func, X_train_padded, y_train, X_test_padded, y_test,
                VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, EPOCHS, BATCH_SIZE
            )
            final_results[model_name] = results

        # Additional Custom Model ("auto_model")
        auto_model_results = train_and_evaluate_model(
            "auto_model", build_auto_model, X_train_padded, y_train, X_test_padded, y_test,
            VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, EPOCHS, BATCH_SIZE
        )
        auto_model_results["description"] = "A hybrid model combining a Conv1D layer for local feature extraction, followed by a MaxPooling1D layer for downsampling, then a GRU layer for sequence understanding, and finally a Dense layer with Dropout for classification."
        final_results["auto_model"] = auto_model_results

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        # Exit gracefully as per requirement if critical errors occur
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Final Output (JSON)
    print("\n<<<JSON_START>>>")
    print(json.dumps(final_results, indent=2))
    print("<<<JSON_END>>>")

if __name__ == "__main__":
    main()
