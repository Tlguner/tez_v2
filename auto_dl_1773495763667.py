import pandas as pd
import numpy as np
import tensorflow as tf
import json
import time
import os
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Dataset Requirements ---

DATA_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'
MAX_LEN = 200 # Maximum length for padded sequences
EMBEDDING_DIM = 128 # Dimension for word embeddings

def load_and_preprocess_data(file_path):
    """
    Loads data, detects text and label columns, performs mapping,
    and returns processed text and labels.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Original DataFrame columns: {df.columns.tolist()}")

    # 1. Detect Text Column
    text_column_candidates = ["text_combined", "Email Text", "email_text", "text", "body", "message", "content", "email", "mail", "raw_text", "sentence", "clean_text"]
    text_column = None
    for col in text_column_candidates:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        # Fallback: choose the column with dtype object/string that has the largest average string length
        object_cols = df.select_dtypes(include='object').columns
        if not object_cols.empty:
            avg_lengths = df[object_cols].astype(str).apply(lambda x: x.str.len().mean())
            text_column = avg_lengths.idxmax()
            print(f"Fallback: Auto-detected text column: '{text_column}' based on average string length.")
        else:
            raise ValueError("Could not find a text column. No object/string columns found.")
    else:
        print(f"Preferred text column: '{text_column}' detected.")

    # 2. Detect Label Column
    label_column_candidates = ["label", "Email Type", "labels", "target", "Category", "class", "y", "is_phishing", "spam", "phish"]
    label_column = None
    for col in label_column_candidates:
        if col in df.columns:
            label_column = col
            break

    if label_column is None:
        # Fallback: choose a column with <= 10 unique values (after dropping NaNs) and not equal to the chosen text column.
        candidate_label_cols = [col for col in df.columns if col != text_column]
        for col in candidate_label_cols:
            if df[col].nunique(dropna=True) <= 10:
                label_column = col
                print(f"Fallback: Auto-detected label column: '{label_column}' based on unique value count (<=10).")
                break
        if label_column is None:
            raise ValueError("Could not find a label column.")
    else:
        print(f"Preferred label column: '{label_column}' detected.")

    X = df[text_column].fillna("") # Fill NaN text with empty string
    y = df[label_column].copy()

    # Drop rows with NaN label only (after getting a copy to avoid SettingWithCopyWarning)
    initial_rows = df.shape[0]
    df.dropna(subset=[label_column], inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows due to NaN in label column.")

    y = df[label_column]
    X = df[text_column].fillna("") # Re-align X after dropping rows


    # 3. Perform Label Mapping
    valid_phishing_labels = ["phishing", "phishing email", "spam", "malicious"]
    valid_safe_labels = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

    if pd.api.types.is_numeric_dtype(y):
        if not y.isin([0, 1]).all():
            raise ValueError("Numeric label column must contain only 0 and 1.")
        # Assuming 0 is safe and 1 is phishing as per requirement for existing numeric labels
        print("Numeric label column detected with 0/1 values. Using as-is (0=safe, 1=phishing).")
    elif pd.api.types.is_string_dtype(y) or pd.api.types.is_object_dtype(y):
        print("String/object label column detected. Performing mapping...")
        y_mapped = pd.Series(index=y.index, dtype=int)
        lower_y = y.astype(str).str.lower()

        # Map phishing/spam variants -> 1
        phish_mask = lower_y.apply(lambda x: any(phish_term in x for phish_term in valid_phishing_labels))
        y_mapped[phish_mask] = 1

        # Map ham/safe/legitimate variants -> 0
        safe_mask = lower_y.apply(lambda x: any(safe_term in x for safe_term in valid_safe_labels))
        y_mapped[safe_mask] = 0

        # Identify unmappable labels
        unmapped_mask = ~(phish_mask | safe_mask)
        if unmapped_mask.any():
            print(f"Warning: Dropping {unmapped_mask.sum()} rows with unmappable labels. Examples: {y[unmapped_mask].unique()[:5]}")
            X = X[~unmapped_mask]
            y_mapped = y_mapped[~unmapped_mask]

        y = y_mapped
    else:
        raise ValueError(f"Unsupported label column dtype: {y.dtype}")

    # Final check: Dataset MUST contain both 0 and 1
    if not (0 in y.unique() and 1 in y.unique()):
        raise ValueError("Dataset mapping failed: need both phishing (1) and safe (0) labels.")

    print(f"Final dataset shape after processing: {X.shape}")
    print(f"Label distribution: \n{y.value_counts()}")

    return X, y

# --- Model Training and Evaluation ---

def build_compile_train_evaluate_model(model_name, model_builder, X_train, y_train, X_test, y_test, vocab_size):
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()

    model = model_builder(vocab_size)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.1, # 10% of training data for validation
                        callbacks=[early_stopping],
                        verbose=0) # Set to 1 or 2 for more verbose output during training

    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds.")

    # Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"{model_name} - Confusion Matrix:\n{np.array(conf_matrix)}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf_matrix,
        "train_accuracy_history": [float(h) for h in history.history['accuracy']],
        "validation_accuracy_history": [float(h) for h in history.history['val_accuracy']],
        "training_time_seconds": float(training_time)
    }

# --- Model Architectures ---

def build_lstm_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_bilstm_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(128)),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gru_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(128),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_cnn_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_hybrid_cnn_dense_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_auto_model(vocab_size):
    """
    Custom AI-designed model: CNN-LSTM Hybrid
    Combines Conv1D for local feature extraction, followed by LSTM for sequential understanding.
    """
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

# --- Main Execution ---

if __name__ == "__main__":
    try:
        # Load and Preprocess Data
        X, y = load_and_preprocess_data(DATA_PATH)

        # Train/Test Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        print(f"\nTrain set size: {len(X_train_raw)}, Test set size: {len(X_test_raw)}")
        print(f"Train label distribution:\n{y_train.value_counts()}")
        print(f"Test label distribution:\n{y_test.value_counts()}")


        # Text Processing (Keras Tokenizer)
        tokenizer = Tokenizer(num_words=None, oov_token="<unk>") # Use None to keep all words
        tokenizer.fit_on_texts(X_train_raw)
        vocab_size = len(tokenizer.word_index) + 1 # +1 for 0-padding

        X_train_sequences = tokenizer.texts_to_sequences(X_train_raw)
        X_test_sequences = tokenizer.texts_to_sequences(X_test_raw)

        X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

        print(f"\nVocabulary size: {vocab_size}")
        print(f"Padded X_train shape: {X_train_padded.shape}")
        print(f"Padded X_test shape: {X_test_padded.shape}")

        results = {}

        # Train and Evaluate Models
        results["lstm"] = build_compile_train_evaluate_model("LSTM", build_lstm_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size)
        results["bilstm"] = build_compile_train_evaluate_model("BiLSTM", build_bilstm_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size)
        results["gru"] = build_compile_train_evaluate_model("GRU", build_gru_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size)
        results["cnn"] = build_compile_train_evaluate_model("1D CNN", build_cnn_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size)
        results["hybrid"] = build_compile_train_evaluate_model("Hybrid CNN+Dense", build_hybrid_cnn_dense_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size)

        # Custom AI-designed model
        auto_model_results = build_compile_train_evaluate_model("auto_model", build_auto_model, X_train_padded, y_train, X_test_padded, y_test, vocab_size)
        auto_model_results["description"] = "A hybrid CNN-LSTM model: Embedding -> Conv1D (64 filters, kernel 3) -> MaxPooling1D (pool 2) -> LSTM (64 units) -> Dense (32 units, relu) -> Dropout (0.3) -> Dense (1 unit, sigmoid)."
        results["auto_model"] = auto_model_results

        print("\n<<<JSON_START>>>")
        print(json.dumps(results, indent=2))
        print("<<<JSON_END>>>")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

