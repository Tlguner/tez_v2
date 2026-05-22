import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import time
import json
import re
import os

# HARD RULES - Dataset Path
DATASET_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/test_dataset/phishing_email.csv'

# Keras Tokenizer parameters
# VOCAB_SIZE will be dynamically updated after tokenizer fitting
MAX_LEN = 200
EMBEDDING_DIM = 128 # Common embedding dimension for all models

# HARD RULES - Training limits
EPOCHS = 10
PATIENCE = 2

# Global variable to store results for all models
all_model_results = {}

# --- Helper function for model training and evaluation ---
def train_evaluate_model(model, model_name, X_train_padded, y_train, X_test_padded, y_test, vocab_size, embedding_dim, max_len):
    """
    Trains and evaluates a given Keras model.
    """
    print(f"\n--- Training {model_name} ---")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Print model summary only for auto_model or first model for brevity in output
    # if model_name == "LSTM" or model_name == "auto_model":
    #     model.summary() 

    start_time = time.time()
    history = model.fit(
        X_train_padded, y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.2, # Used for early stopping
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time

    # Evaluate on the actual test set
    y_pred_probs = model.predict(X_test_padded)
    y_pred = (y_pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist() # Convert numpy array to list for JSON

    train_acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix,
        "train_accuracy_history": train_acc_history,
        "validation_accuracy_history": val_acc_history,
        "training_time": training_time
    }
    return results

# --- Main script execution ---
def main():
    print("Starting deep learning pipeline...")

    # --- Dataset Loading ---
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset loaded from {DATASET_PATH}. Shape: {df.shape}")
    except Exception as e:
        raise IOError(f"Error loading dataset: {e}")

    # --- Text Column Detection ---
    text_column = None
    preferred_text_cols = [
        "text_combined","Email Text","email_text","text","body","message","content","email","mail","raw_text","sentence","clean_text"
    ]
    for col in preferred_text_cols:
        if col in df.columns:
            text_column = col
            print(f"Detected text column: '{text_column}' (preferred match)")
            break
    
    if text_column is None:
        object_or_string_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(object_or_string_cols) > 0:
            max_avg_len = -1
            for col in object_or_string_cols:
                # Calculate average string length, excluding NaNs
                # Handle cases where all values are non-string or NaN
                valid_strings = df[col].dropna().astype(str)
                if not valid_strings.empty:
                    avg_len = valid_strings.apply(len).mean()
                    if avg_len > max_avg_len:
                        max_avg_len = avg_len
                        text_column = col
            if text_column:
                print(f"Detected text column: '{text_column}' (fallback: largest average string length)")
        if text_column is None:
            raise ValueError("Could not find a text column")

    # --- Label Column Detection ---
    label_column = None
    preferred_label_cols = [
        "label","Email Type","labels","target","Category","class","y","is_phishing","spam","phish"
    ]
    for col in preferred_label_cols:
        if col in df.columns and col != text_column: # Ensure it's not the text column
            label_column = col
            print(f"Detected label column: '{label_column}' (preferred match)")
            break

    if label_column is None:
        # Check object, string, and integer columns for potential labels
        potential_label_cols = [
            col for col in df.select_dtypes(include=['object', 'string', 'int']).columns
            if col != text_column # Exclude the text column itself
        ]
        
        for col in potential_label_cols:
            unique_values = df[col].dropna().nunique()
            if unique_values > 0 and unique_values <= 10: # Must have some values, and <= 10 unique
                label_column = col
                print(f"Detected label column: '{label_column}' (fallback: <=10 unique non-text values)")
                break
        if label_column is None:
            raise ValueError("Could not find a label column")

    # --- Data Cleaning and Label Mapping ---
    df_processed = df[[text_column, label_column]].copy()
    df_processed.rename(columns={text_column: 'text', label_column: 'label'}, inplace=True)

    phishing_variants = ["phishing", "phishing email", "spam", "malicious"]
    safe_variants = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

    # Convert labels to lowercase for case-insensitive matching
    df_processed['label_lower'] = df_processed['label'].astype(str).str.lower()

    def map_label(label_lower):
        if pd.isna(label_lower):
            return np.nan
        # Use regex for whole word matching
        if any(re.search(r'\b' + re.escape(variant) + r'\b', label_lower) for variant in phishing_variants):
            return 1
        if any(re.search(r'\b' + re.escape(variant) + r'\b', label_lower) for variant in safe_variants):
            return 0
        return np.nan # Mark for dropping if not recognized

    df_processed['mapped_label'] = df_processed['label_lower'].apply(map_label)

    # Drop rows with NaN text or unmapped labels
    initial_rows = df_processed.shape[0]
    df_processed.dropna(subset=['text', 'mapped_label'], inplace=True)
    df_processed['mapped_label'] = df_processed['mapped_label'].astype(int) # Convert to int after dropping NaNs
    rows_after_drop = df_processed.shape[0]
    print(f"Dropped {initial_rows - rows_after_drop} rows due to NaN text or unmappable labels.")

    # Validate mapped labels
    unique_mapped_labels = df_processed['mapped_label'].unique()
    if not (0 in unique_mapped_labels and 1 in unique_mapped_labels):
        raise ValueError("Dataset mapping failed: need both phishing (1) and safe (0) labels. Found only one or zero unique labels after processing.")

    X = df_processed['text'].values
    y = df_processed['mapped_label'].values
    print(f"Processed dataset size: {len(X)} samples. Label distribution:\n{pd.Series(y).value_counts()}")

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/Test Split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- Text Processing (Keras Tokenizer) ---
    # `num_words` is an initial cap, the actual vocab size will be `len(tokenizer.word_index) + 1`
    tokenizer = Tokenizer(num_words=10000, oov_token="<unk>") 
    tokenizer.fit_on_texts(X_train)
    
    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer.word_index) + 1 # +1 for 0-padding
    print(f"Vocabulary size derived from training data: {VOCAB_SIZE}")

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    print(f"Texts tokenized and padded. Max sequence length: {MAX_LEN}")

    # --- Define and Train Models ---

    # 1. LSTM
    model_lstm = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ], name="LSTM_Model")
    all_model_results['lstm'] = train_evaluate_model(
        model_lstm, "LSTM", X_train_padded, y_train, X_test_padded, y_test, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN
    )

    # 2. BiLSTM
    model_bilstm = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(128)),
        Dense(1, activation='sigmoid')
    ], name="BiLSTM_Model")
    all_model_results['bilstm'] = train_evaluate_model(
        model_bilstm, "BiLSTM", X_train_padded, y_train, X_test_padded, y_test, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN
    )

    # 3. GRU
    model_gru = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(128),
        Dense(1, activation='sigmoid')
    ], name="GRU_Model")
    all_model_results['gru'] = train_evaluate_model(
        model_gru, "GRU", X_train_padded, y_train, X_test_padded, y_test, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN
    )

    # 4. 1D CNN
    model_cnn = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ], name="CNN_Model")
    all_model_results['cnn'] = train_evaluate_model(
        model_cnn, "CNN", X_train_padded, y_train, X_test_padded, y_test, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN
    )

    # 5. Hybrid CNN + Dense
    model_hybrid = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="Hybrid_CNN_Dense_Model")
    all_model_results['hybrid'] = train_evaluate_model(
        model_hybrid, "Hybrid CNN + Dense", X_train_padded, y_train, X_test_padded, y_test, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN
    )

    # --- Additional Custom Model ("auto_model") ---
    # Architecture: Embedding -> Conv1D -> MaxPooling1D -> GRU -> Dense -> Dropout -> Dense
    # This combines CNN for local feature extraction, followed by GRU for sequence processing
    # after downsampling, and then dense layers, making it distinct and powerful.
    model_auto = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        GRU(64, return_sequences=False), # return_sequences=False to pass to Dense
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="Auto_Model")
    
    auto_model_metrics = train_evaluate_model(
        model_auto, "auto_model", X_train_padded, y_train, X_test_padded, y_test, VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN
    )
    all_model_results['auto_model'] = {
        "description": "Sequential model with Embedding -> Conv1D -> MaxPooling1D -> GRU -> Dense -> Dropout -> Dense layers. Combines convolutional feature extraction with recurrent sequence understanding, followed by a final dense classifier.",
        **auto_model_metrics
    }

    # --- Final Output ---
    print("\n<<<JSON_START>>>")
    print(json.dumps(all_model_results, indent=2))
    print("<<<JSON_END>>>")

if __name__ == "__main__":
    main()
