import pandas as pd
import numpy as np
import json
import time
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping

# --- HARD RULES: Keras Imports Required ---
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding
)
# --- End Keras Imports Required ---

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Dataset Requirements ---
DATA_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'

PREFERRED_TEXT_COLUMNS = [
    "text_combined", "Email Text", "email_text", "text", "body", "message",
    "content", "email", "mail", "raw_text", "sentence", "clean_text"
]
PREFERRED_LABEL_COLUMNS = [
    "label", "Email Type", "labels", "target", "Category", "class", "y",
    "is_phishing", "spam", "phish"
]

PHISHING_LABELS = ["phishing", "phishing email", "spam", "malicious"]
SAFE_LABELS = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

# --- Text Processing Constants ---
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128 # A common choice for embedding dimension

# --- Training Limits ---
NUM_EPOCHS = 10
PATIENCE = 2

def load_and_preprocess_data(data_path):
    """
    Loads the dataset, detects text and label columns, performs label mapping,
    and handles missing values.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Error loading CSV from {data_path}: {e}")

    initial_rows = len(df)
    print(f"Initial dataset loaded with {initial_rows} rows.")

    # 1. Detect Text Column
    text_column = None
    for col in PREFERRED_TEXT_COLUMNS:
        if col in df.columns:
            text_column = col
            print(f"Detected text column: '{text_column}' (from preferred list)")
            break
    
    if text_column is None:
        # Fallback: largest average string length among object/string columns
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(string_cols) == 0:
            raise ValueError("Could not find a text column: No object/string columns found.")
        
        col_avg_lengths = {}
        for col in string_cols:
            temp_series = df[col].dropna().astype(str)
            if not temp_series.empty:
                col_avg_lengths[col] = temp_series.apply(len).mean()
            else:
                col_avg_lengths[col] = 0 # Handle empty columns
        
        if col_avg_lengths:
            text_column = max(col_avg_lengths, key=col_avg_lengths.get)
            if col_avg_lengths[text_column] > 0:
                print(f"Detected text column: '{text_column}' (fallback: largest average string length)")
            else:
                raise ValueError("Could not find a text column: All object/string columns are empty or contain only empty strings.")
        else:
            raise ValueError("Could not find a text column: No suitable object/string columns found for fallback.")

    # 2. Detect Label Column
    label_column = None
    for col in PREFERRED_LABEL_COLUMNS:
        if col in df.columns and col != text_column: # Ensure it's not the text column
            label_column = col
            print(f"Detected label column: '{label_column}' (from preferred list)")
            break
    
    if label_column is None:
        # Fallback: <= 10 unique values, not text column
        candidate_labels = []
        for col in df.columns:
            if col != text_column:
                # Drop NaNs before counting unique to mimic how they'd be used
                unique_values = df[col].dropna().nunique()
                if unique_values > 0 and unique_values <= 10:
                    candidate_labels.append((col, unique_values))
        
        if candidate_labels:
            # Pick the one with fewest unique values, as it's more likely to be a label
            label_column = min(candidate_labels, key=lambda x: x[1])[0]
            print(f"Detected label column: '{label_column}' (fallback: <=10 unique values)")
        else:
            raise ValueError("Could not find a label column: No column with <=10 unique non-NaN values found.")

    # 3. Perform Label Mapping
    print(f"Original unique labels in '{label_column}': {df[label_column].dropna().unique()}")
    mapped_labels = []
    dropped_count = 0

    if pd.api.types.is_numeric_dtype(df[label_column]):
        if not set(df[label_column].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Numeric label column '{label_column}' must contain only 0 and 1. Found: {df[label_column].dropna().unique()}")
        df['mapped_label'] = df[label_column].astype(int) # Use as-is
    else: # String/Object dtype
        lower_phishing_labels = [s.lower() for s in PHISHING_LABELS]
        lower_safe_labels = [s.lower() for s in SAFE_LABELS]

        for index, row in df.iterrows():
            label_val = str(row[label_column]).lower().strip() if pd.notna(row[label_column]) else None
            
            if label_val in lower_phishing_labels:
                mapped_labels.append(1)
            elif label_val in lower_safe_labels:
                mapped_labels.append(0)
            else:
                mapped_labels.append(np.nan) # Mark for dropping
                if pd.notna(label_val): # Only count non-NaN as unmappable
                    dropped_count += 1
        df['mapped_label'] = mapped_labels
        
    df_original_size = len(df)
    df.dropna(subset=['mapped_label'], inplace=True)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows due to unmappable string labels.")
    elif df_original_size - len(df) > 0:
        print(f"Dropped {df_original_size - len(df)} rows due to NaN labels after numeric mapping (if any).")

    # 4. Fill NaN text with empty string ""
    df[text_column] = df[text_column].fillna("").astype(str)

    # 5. Drop rows with NaN label only (already handled by df.dropna(subset=['mapped_label']))
    # This ensures that any rows whose label was unmappable are removed.

    # Final Check: Dataset MUST contain both 0 and 1
    if not (0 in df['mapped_label'].unique() and 1 in df['mapped_label'].unique()):
        raise ValueError("Dataset mapping failed: need both phishing (1) and safe (0) labels after preprocessing.")
    
    print(f"Final dataset size after preprocessing: {len(df)} rows. {len(df) - initial_rows} rows dropped.")
    print(f"Final label distribution: {df['mapped_label'].value_counts()}")

    return df, text_column, 'mapped_label'


def train_and_evaluate_model(model_builder_fn, model_name, X_train, y_train, X_test, y_test,
                             vocab_size, embedding_dim, max_len):
    """
    Trains a deep learning model, evaluates it, and returns performance metrics.
    """
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()

    model = model_builder_fn(vocab_size, embedding_dim, max_len)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary() # Optional: print model summary

    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=32,
        validation_split=0.1, # Using a small validation split from training data
        callbacks=[early_stopping],
        verbose=1 # Print training logs
    )

    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds.")

    # Evaluate on test set
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": conf_mat.tolist(),
        "train_accuracy_history": history.history['accuracy'],
        "validation_accuracy_history": history.history['val_accuracy'],
        "training_time_seconds": training_time
    }

# --- Deep Learning Model Definitions ---

def build_lstm(vocab_size, embedding_dim, max_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(128, dropout=0.5, recurrent_dropout=0.5), # Added recurrent_dropout for LSTM/GRU
        Dense(1, activation='sigmoid')
    ])
    return model

def build_bilstm(vocab_size, embedding_dim, max_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5)),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gru(vocab_size, embedding_dim, max_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GRU(128, dropout=0.5, recurrent_dropout=0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_cnn(vocab_size, embedding_dim, max_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_hybrid_cnn_dense(vocab_size, embedding_dim, max_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_auto_model(vocab_size, embedding_dim, max_len):
    """
    Custom 'auto_model' architecture: CNN followed by GRU, then GlobalAveragePooling
    and Dense layers.
    This combines local feature extraction (CNN) with sequential processing (GRU),
    and then a global summary (GAP) before classification.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), # GRU processes sequence after CNN
        GlobalAveragePooling1D(), # Summarize the GRU outputs
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.description = "CNN-GRU with GlobalAveragePooling: Extracts local features with Conv1D, processes sequence with GRU, then summarizes with GlobalAveragePooling before Dense classification."
    return model

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Data Loading and Preprocessing
        df, text_col, label_col = load_and_preprocess_data(DATA_PATH)
        texts = df[text_col].tolist()
        labels = df[label_col].values

        # 2. Train/Test Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"\nTrain set size: {len(X_train_raw)}, Test set size: {len(X_test_raw)}")
        print(f"Train label distribution: {pd.Series(y_train).value_counts()}")
        print(f"Test label distribution: {pd.Series(y_test).value_counts()}")

        # 3. Text Processing (Keras Tokenizer)
        tokenizer = Tokenizer(num_words=None, oov_token="<unk>") # Let tokenizer determine vocab size
        tokenizer.fit_on_texts(X_train_raw)

        vocab_size = len(tokenizer.word_index) + 1 # +1 for 0-padding
        print(f"\nVocabulary size: {vocab_size}")

        X_train_sequences = tokenizer.texts_to_sequences(X_train_raw)
        X_test_sequences = tokenizer.texts_to_sequences(X_test_raw)

        X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        print(f"Padded sequence shape: {X_train_padded.shape}")

        # Store results for JSON output
        all_results = {}

        # 4. Train and Evaluate Mandatory Models
        model_configs = {
            "lstm": build_lstm,
            "bilstm": build_bilstm,
            "gru": build_gru,
            "cnn": build_cnn,
            "hybrid": build_hybrid_cnn_dense
        }

        for model_name, builder_fn in model_configs.items():
            result = train_and_evaluate_model(
                builder_fn, model_name.upper(),
                X_train_padded, y_train, X_test_padded, y_test,
                vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
            )
            all_results[model_name] = result

        # 5. Train and Evaluate Custom "auto_model"
        auto_model_builder = build_auto_model
        auto_model_name = "auto_model"
        auto_model_result = train_and_evaluate_model(
            auto_model_builder, auto_model_name.upper(),
            X_train_padded, y_train, X_test_padded, y_test,
            vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
        )
        auto_model_result["description"] = auto_model_builder(vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH).description
        all_results[auto_model_name] = auto_model_result

        # 6. Final Output
        print("\n<<<JSON_START>>>")
        print(json.dumps(all_results, indent=2))
        print("<<<JSON_END>>>")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

