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

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Dataset Requirements ---
DATASET_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/test_dataset/phishing_email.csv'
MAX_LEN = 200 # Fixed sequence length for padding
EMBEDDING_DIM = 100 # Dimensionality for word embeddings

def load_and_preprocess_data(file_path):
    """
    Loads data from the specified path, automatically detects text and label columns,
    performs label mapping, and handles NaNs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")

    # --- Automatically detect text column ---
    text_col_candidates = ["text_combined", "Email Text", "email_text", "text", "body", "message", "content", "email", "mail", "raw_text", "sentence", "clean_text"]
    text_column = None
    for col in text_col_candidates:
        if col in df.columns:
            text_column = col
            print(f"Detected text column: '{text_column}' from candidates.")
            break

    if text_column is None:
        # Fallback: choose the column with dtype object/string that has the largest average string length.
        object_cols = df.select_dtypes(include=['object', 'string']).columns
        if not object_cols.empty:
            avg_lengths = {}
            for col in object_cols:
                # Ensure it's not entirely NaN or empty strings before calculating length
                temp_series = df[col].dropna().astype(str).str.len()
                if not temp_series.empty:
                    avg_lengths[col] = temp_series.mean()
            if avg_lengths:
                text_column = max(avg_lengths, key=avg_lengths.get)
                print(f"Detected text column: '{text_column}' using fallback (largest average string length).")
            else:
                raise ValueError("Could not find a text column: No suitable object/string columns with text data found.")
        else:
            raise ValueError("Could not find a text column: No object/string columns available.")

    # Fill NaN text with empty string ""
    df[text_column] = df[text_column].fillna("").astype(str)
    print(f"Filled NaN values in text column '{text_column}' with empty strings.")

    # --- Automatically detect label column ---
    label_col_candidates = ["label", "Email Type", "labels", "target", "Category", "class", "y", "is_phishing", "spam", "phish"]
    label_column = None
    for col in label_col_candidates:
        if col in df.columns:
            label_column = col
            print(f"Detected label column: '{label_column}' from candidates.")
            break

    if label_column is None:
        # Fallback: choose a column with <= 10 unique values (after dropping NaNs) and not equal to the chosen text column.
        candidate_label_cols = [col for col in df.columns if col != text_column and df[col].nunique(dropna=True) <= 10]
        if candidate_label_cols:
            # Prefer numeric or boolean-like columns if available
            numeric_like_cols = [col for col in candidate_label_cols if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])]
            if numeric_like_cols:
                label_column = numeric_like_cols[0] # Just pick the first one
                print(f"Detected label column: '{label_column}' using fallback (numeric-like with <=10 unique values).")
            else:
                label_column = candidate_label_cols[0] # Pick the first available
                print(f"Detected label column: '{label_column}' using fallback (<=10 unique values).")
        else:
            raise ValueError("Could not find a label column: No suitable column with <=10 unique values found.")

    # Drop rows with NaN label only
    original_rows = len(df)
    df.dropna(subset=[label_column], inplace=True)
    if len(df) < original_rows:
        print(f"Dropped {original_rows - len(df)} rows due to NaN values in the label column.")

    # --- Perform label mapping ---
    y_raw = df[label_column]

    valid_phishing_labels = ["phishing", "phishing email", "spam", "malicious"]
    valid_safe_labels = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

    dropped_unmappable_count = 0

    if pd.api.types.is_numeric_dtype(y_raw):
        # 1. If label column dtype is numeric:
        unique_vals = y_raw.dropna().unique()
        if not (np.isin(unique_vals, [0, 1])).all():
            raise ValueError(f"Numeric label column '{label_column}' must contain only 0 and 1. Found: {unique_vals}")
        print(f"Numeric label column '{label_column}' already in 0/1 format. Using as-is.")
        df['encoded_label'] = y_raw.astype(int)
    else:
        # 2. If label column dtype is string/object:
        print(f"String label column '{label_column}' detected. Mapping labels...")
        mapped_labels = []
        for label in y_raw:
            label_lower = str(label).lower()
            if label_lower in valid_phishing_labels:
                mapped_labels.append(1)
            elif label_lower in valid_safe_labels:
                mapped_labels.append(0)
            else:
                mapped_labels.append(np.nan) # Mark for dropping
                dropped_unmappable_count += 1
        
        df['encoded_label'] = mapped_labels
        
        if dropped_unmappable_count > 0:
            original_len_before_drop = len(df)
            df.dropna(subset=['encoded_label'], inplace=True)
            print(f"Dropped {original_len_before_drop - len(df)} rows with unmappable labels.")
        
        df['encoded_label'] = df['encoded_label'].astype(int)

    # After mapping: Dataset MUST contain both 0 and 1
    if df['encoded_label'].nunique() < 2:
        raise ValueError("Dataset mapping failed: need both phishing (1) and safe (0) labels after processing. Check your label mapping rules or data.")
    
    print(f"Final label distribution:\n{df['encoded_label'].value_counts()}")

    return df[text_column], df['encoded_label']

# Load and preprocess data
X, y = load_and_preprocess_data(DATASET_PATH)

# --- Train/Test Split ---
print("\nPerforming stratified 80/20 train/test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train/Test Split complete: {len(X_train)} training samples, {len(X_test)} test samples.")
print(f"Training label distribution:\n{y_train.value_counts()}")
print(f"Test label distribution:\n{y_test.value_counts()}")

# --- Text Processing ---
print("\nPerforming text tokenization and padding...")
tokenizer = Tokenizer(num_words=None, oov_token="<unk>") # Let tokenizer determine vocab size
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1 # +1 for <unk> token

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Vocabulary size: {vocab_size}")
print(f"Max sequence length: {MAX_LEN}")
print(f"Shape of X_train_padded: {X_train_padded.shape}")
print(f"Shape of X_test_padded: {X_test_padded.shape}")

# --- Model Training and Evaluation Function ---
def train_evaluate_model(model, name, X_train, y_train, X_test, y_test):
    """
    Trains a given Keras model and evaluates its performance on the test set.
    """
    print(f"\n--- Training {name} Model ---")
    start_time = time.time()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=10, # Max 10 epochs
        batch_size=32,
        validation_split=0.1, # 10% of training data used for validation
        callbacks=[early_stopping],
        verbose=1 # Print training progress
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time for {name}: {training_time:.2f} seconds")

    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int) # Convert probabilities to binary predictions

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist() # Convert numpy array to list for JSON serialization

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "train_accuracy_history": [float(val) for val in history.history['accuracy']],
        "validation_accuracy_history": [float(val) for val in history.history['val_accuracy']],
        "training_time_seconds": float(training_time)
    }
    print(f"Test Accuracy for {name}: {accuracy:.4f}")
    print(f"Test Precision for {name}: {precision:.4f}")
    print(f"Test Recall for {name}: {recall:.4f}")
    print(f"Test F1-Score for {name}: {f1:.4f}")
    print(f"Test Confusion Matrix for {name}:\n{cm}")

    return results

all_model_results = {}

# --- 1. LSTM Model ---
lstm_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
all_model_results['lstm'] = train_evaluate_model(lstm_model, "LSTM", X_train_padded, y_train, X_test_padded, y_test)

# --- 2. BiLSTM Model ---
bilstm_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(128)),
    Dense(1, activation='sigmoid')
])
all_model_results['bilstm'] = train_evaluate_model(bilstm_model, "BiLSTM", X_train_padded, y_train, X_test_padded, y_test)

# --- 3. GRU Model ---
gru_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    GRU(128),
    Dense(1, activation='sigmoid')
])
all_model_results['gru'] = train_evaluate_model(gru_model, "GRU", X_train_padded, y_train, X_test_padded, y_test)

# --- 4. 1D CNN Model ---
cnn_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
all_model_results['cnn'] = train_evaluate_model(cnn_model, "1D CNN", X_train_padded, y_train, X_test_padded, y_test)

# --- 5. Hybrid CNN + Dense Model ---
# This model uses Conv1D for local feature extraction, MaxPooling1D to reduce dimensionality,
# followed by GlobalMaxPooling1D to flatten for a Dense block.
hybrid_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(pool_size=4),
    GlobalMaxPooling1D(), # Acts as a flattening step to pass to Dense layers
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
all_model_results['hybrid'] = train_evaluate_model(hybrid_model, "Hybrid CNN + Dense", X_train_padded, y_train, X_test_padded, y_test)

# --- Additional Custom Model ("auto_model") - CNN-GRU Hybrid ---
auto_model_description = "A custom model combining a 1D Convolutional layer for local feature extraction, followed by MaxPooling, a GRU layer to process sequences, and two Dense layers with Dropout for classification. It aims to capture both local patterns and sequential dependencies in the text data."
auto_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=2),
    GRU(64), # GRU processes the output of the CNN
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
auto_model_results = train_evaluate_model(auto_model, "auto_model (CNN-GRU Hybrid)", X_train_padded, y_train, X_test_padded, y_test)
auto_model_results["description"] = auto_model_description
all_model_results['auto_model'] = auto_model_results

# --- Final Output ---
print("\n<<<JSON_START>>>")
print(json.dumps(all_model_results, indent=2))
print("<<<JSON_END>>>")
