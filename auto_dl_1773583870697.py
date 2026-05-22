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
    GlobalMaxPooling1D, Dense, Dropout, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
DATASET_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/test_dataset/phishing_email.csv'
MAX_LEN = 200 # Fixed sequence length for padding
EMBEDDING_DIM = 100 # Dimension for word embeddings
MAX_WORDS = 10000 # Maximum number of words to keep in the vocabulary

# --- Dataset Requirements for Column Detection and Mapping ---
text_col_candidates = [
    "text_combined", "Email Text", "email_text", "text", "body", "message",
    "content", "email", "mail", "raw_text", "sentence", "clean_text"
]
label_col_candidates = [
    "label", "Email Type", "labels", "target", "Category", "class",
    "y", "is_phishing", "spam", "phish"
]

phishing_labels_map = [s.lower() for s in ["phishing", "phishing email", "spam", "malicious"]]
safe_labels_map = [s.lower() for s in ["ham", "safe", "safe email", "legitimate", "normal", "benign"]]

def load_and_preprocess_data(file_path):
    """
    Loads data from the specified CSV, automatically detects text and label columns,
    performs label mapping, and handles NaNs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset from: {file_path}")
        print(f"Initial dataset shape: {df.shape}")
        print(f"Initial columns: {df.columns.tolist()}")
    except Exception as e:
        raise IOError(f"Error loading CSV from {file_path}: {e}")

    # --- Detect Text Column ---
    text_col = None
    for col_name in text_col_candidates:
        if col_name in df.columns:
            text_col = col_name
            print(f"Detected text column: '{text_col}' from preferred candidates.")
            break

    if text_col is None:
        # Fallback: choose the column with dtype object/string that has the largest average string length.
        object_or_string_cols = [col for col in df.columns if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])]
        
        if not object_or_string_cols:
            raise ValueError("Could not find a text column (no object/string columns found in dataframe).")

        max_avg_len = -1
        best_text_col_fallback = None
        for col in object_or_string_cols:
            # Fill NaN with empty string for length calculation only, to avoid errors and ensure accurate length.
            current_avg_len = df[col].fillna('').astype(str).apply(len).mean()
            if current_avg_len > max_avg_len:
                max_avg_len = current_avg_len
                best_text_col_fallback = col
        
        if best_text_col_fallback is not None:
            text_col = best_text_col_fallback
            print(f"Detected text column: '{text_col}' using fallback (largest average string length).")
        else:
            raise ValueError("Could not find a text column (fallback strategy failed to identify one).")

    # --- Detect Label Column ---
    label_col = None
    for col_name in label_col_candidates:
        if col_name in df.columns and col_name != text_col: # Ensure it's not the text column itself
            label_col = col_name
            print(f"Detected label column: '{label_col}' from preferred candidates.")
            break

    if label_col is None:
        # Fallback: choose a column with <= 10 unique values (after dropping NaNs) and not equal to the chosen text column.
        candidate_label_cols = []
        for col in df.columns:
            if col != text_col: # Exclude the already identified text column
                # Use dropna to count unique values for non-null entries
                unique_values_count = df[col].dropna().nunique()
                if unique_values_count <= 10:
                    candidate_label_cols.append(col)
        
        if len(candidate_label_cols) > 0:
            label_col = candidate_label_cols[0] # Pick the first one found
            print(f"Detected label column: '{label_col}' using fallback (<=10 unique values, not the text column).")
        else:
            raise ValueError("Could not find a label column (fallback strategy failed to identify one).")

    # --- Perform Label Mapping ---
    df_processed = df[[text_col, label_col]].copy()
    df_processed.rename(columns={text_col: 'text_content', label_col: 'original_label'}, inplace=True)

    mapped_labels = pd.Series(index=df_processed.index, dtype=int)

    if pd.api.types.is_numeric_dtype(df_processed['original_label']):
        unique_vals = df_processed['original_label'].dropna().unique()
        # Check if all unique values are exactly 0 or 1
        if not (np.isin(unique_vals, [0, 1]).all()):
            raise ValueError(
                f"Numeric label column '{label_col}' must contain only 0 and 1, "
                f"but found: {unique_vals}. Please ensure 0 represents safe and 1 represents phishing/spam."
            )
        mapped_labels = df_processed['original_label'].astype(int)
        print("Numeric label column detected. Using labels as-is (0=safe, 1=phishing).")
    else: # String/Object dtype
        print("String label column detected. Applying mapping...")
        def map_string_label(label):
            if pd.isna(label):
                return np.nan # Keep NaN for unmappable/missing
            label_lower = str(label).lower()
            if label_lower in phishing_labels_map:
                return 1
            elif label_lower in safe_labels_map:
                return 0
            else:
                return np.nan # Unmappable
        
        mapped_labels = df_processed['original_label'].apply(map_string_label)
    
    df_processed['label'] = mapped_labels
    
    # Drop rows with unmappable labels (which become NaN after mapping) or original NaN labels
    initial_rows = len(df_processed)
    df_processed.dropna(subset=['label'], inplace=True)
    rows_dropped_labels = initial_rows - len(df_processed)
    if rows_dropped_labels > 0:
        print(f"Dropped {rows_dropped_labels} rows due to unmappable or NaN labels.")

    # Fill NaN text with empty string ""
    df_processed['text_content'] = df_processed['text_content'].fillna('').astype(str)
    
    # After mapping: Dataset MUST contain both 0 and 1
    if not (0 in df_processed['label'].unique() and 1 in df_processed['label'].unique()):
        raise ValueError("Dataset mapping failed: need both phishing (1) and safe (0) labels after mapping and dropping.")
    
    print(f"Final dataset shape after processing: {df_processed.shape}")
    print(f"Label distribution: \n{df_processed['label'].value_counts()}")

    return df_processed['text_content'], df_processed['label']

def create_model(model_type, vocab_size, embedding_dim, max_len):
    """
    Creates and returns a Keras Sequential model based on the specified type.
    """
    model = Sequential()
    
    if model_type == "LSTM":
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(LSTM(128))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "BiLSTM":
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "GRU":
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(GRU(128))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "CNN":
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "Hybrid":
        # CNN + Dense
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "auto_model":
        # AI-designed custom model: A deeper CNN with a GRU layer
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(Conv1D(128, 7, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, 5, activation='relu', padding='same'))
        model.add(GRU(64, return_sequences=False)) # Use return_sequences=False to pass directly to Dense
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains the given Keras model and evaluates its performance on the test set.
    """
    print(f"\n--- Training {model_name} Model ---")
    start_time = time.time()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=10, # Max 10 epochs as per requirements
        batch_size=32,
        validation_split=0.2, # 20% of training data used for validation
        callbacks=[early_stopping],
        verbose=1 # Print training logs per epoch
    )

    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds.")

    # Evaluate on the unseen test set
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0) # Handle cases where no positive predictions
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON serialization

    print(f"--- {model_name} Evaluation on Test Set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{np.array(cm)}")

    # Ensure accuracy history is available, handle cases where EarlyStopping might end early
    train_acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "train_accuracy_history": [float(val) for val in train_acc],
        "validation_accuracy_history": [float(val) for val in val_acc],
        "training_time_seconds": float(training_time)
    }


# --- Main Execution ---
if __name__ == "__main__":
    results = {}

    try:
        # 1. Load and Preprocess Data
        X_raw, y = load_and_preprocess_data(DATASET_PATH)

        # 2. Train/Test Split (80/20 stratified)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"\nTrain/Test Split: Train samples = {len(X_train_raw)}, Test samples = {len(X_test_raw)}")
        print(f"Train label distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test label distribution:\n{y_test.value_counts(normalize=True)}")

        # 3. Text Processing (Keras Tokenizer and Padding)
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
        tokenizer.fit_on_texts(X_train_raw)
        
        # Determine actual vocabulary size, capped by MAX_WORDS
        VOCAB_SIZE = len(tokenizer.word_index) + 1 # +1 for padding/0 index
        if VOCAB_SIZE > MAX_WORDS:
            VOCAB_SIZE = MAX_WORDS # Cap at MAX_WORDS if more words were found

        X_train_sequences = tokenizer.texts_to_sequences(X_train_raw)
        X_test_sequences = tokenizer.texts_to_sequences(X_test_raw)

        X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        print(f"\nVocabulary size: {VOCAB_SIZE}")
        print(f"Padded sequence length: {MAX_LEN}")

        # 4. Train and Evaluate Mandatory Deep Learning Models
        model_configs = {
            "lstm": "LSTM",
            "bilstm": "BiLSTM",
            "gru": "GRU",
            "cnn": "CNN",
            "hybrid": "Hybrid"
        }

        for model_key, model_type in model_configs.items():
            tf.keras.backend.clear_session() # Clear session to prevent memory issues and ensure fresh models
            model = create_model(model_type, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)
            model_results = train_and_evaluate_model(model, X_train_padded, y_train, X_test_padded, y_test, model_type)
            results[model_key] = model_results

        # 5. Train and Evaluate Custom "auto_model"
        tf.keras.backend.clear_session()
        auto_model = create_model("auto_model", VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)
        # Short description of the auto_model architecture
        auto_model_description = (
            "Sequential model combining two Conv1D layers with MaxPooling, "
            "followed by a GRU layer to capture sequential dependencies. "
            "A Dense hidden layer with Dropout is used before the final sigmoid output. "
            "This architecture leverages both local feature extraction (CNN) and sequence understanding (GRU)."
        )
        auto_model_results = train_and_evaluate_model(auto_model, X_train_padded, y_train, X_test_padded, y_test, "auto_model")
        auto_model_results["description"] = auto_model_description
        results["auto_model"] = auto_model_results

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\nError during script execution: {e}")
        # Exit with a non-zero code to indicate failure
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # Exit with a non-zero code to indicate failure
        exit(1)

    print("\n<<<JSON_START>>>")
    print(json.dumps(results, indent=2))
    print("<<<JSON_END>>>")
