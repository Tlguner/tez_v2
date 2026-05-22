import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
import time

# Keras Imports Required
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding,
    Input, MultiHeadAttention
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import History

# --- Configuration ---
DATASET_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'
TEXT_COLUMN_CANDIDATES = ["Email Text", "text", "body", "message"]
LABEL_COLUMN_CANDIDATES = ["Email Type", "label", "labels", "target", "Category"]

PHISHING_LABELS = ["phishing", "phishing email", "spam", "malicious"]
HAM_LABELS = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
OOV_TOKEN = "<unk>" # Out-of-vocabulary token
TRAINING_EPOCHS = 10
BATCH_SIZE = 32

# --- Dataset Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Detect text column
    text_column = None
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            text_column = col
            break
    if text_column is None:
        raise ValueError(f"Could not find a text column from candidates: {TEXT_COLUMN_CANDIDATES}")

    # Detect label column
    label_column = None
    for col in LABEL_COLUMN_CANDIDATES:
        if col in df.columns:
            label_column = col
            break
    if label_column is None:
        raise ValueError(f"Could not find a label column from candidates: {LABEL_COLUMN_CANDIDATES}")

    df = df[[text_column, label_column]].copy()
    df.columns = ['text', 'label']

    # Drop rows with NaN in text or label
    df.dropna(subset=['text', 'label'], inplace=True)

    # Perform label mapping
    def map_label(label):
        label_lower = str(label).lower()
        if label_lower in PHISHING_LABELS:
            return 1
        elif label_lower in HAM_LABELS:
            return 0
        return np.nan # Mark for dropping if not in valid lists

    df['mapped_label'] = df['label'].apply(map_label)

    # Drop rows with unmapped labels
    df.dropna(subset=['mapped_label'], inplace=True)
    df['mapped_label'] = df['mapped_label'].astype(int)

    # Check for both 0 and 1
    if not (1 in df['mapped_label'].values and 0 in df['mapped_label'].values):
        raise ValueError("Dataset mapping failed: need both phishing and safe labels")

    return df['text'].tolist(), df['mapped_label'].tolist()

# --- Model Training and Evaluation Function ---
def train_and_evaluate_model(model_builder_fn, X_train, y_train, X_test, y_test, vocab_size, embedding_dim, max_len, epochs, batch_size):
    tf.keras.backend.clear_session() # Clear session to avoid name conflicts

    model = model_builder_fn(vocab_size, embedding_dim, max_len)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use a validation split for history tracking
        verbose=0, # Suppress output during training
        callbacks=[History()]
    )
    training_time = time.time() - start_time

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf_matrix,
        "train_accuracy_history": [float(val) for val in history.history['accuracy']],
        "validation_accuracy_history": [float(val) for val in history.history['val_accuracy']],
        "training_time": float(training_time)
    }

# --- Model Architectures ---
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
        Bidirectional(LSTM(64)), # Reduced units for BiLSTM often
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
        tf.keras.layers.Flatten(), # Explicitly import Flatten as well if needed
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_auto_model(vocab_size, embedding_dim, max_len, head_size=64, num_heads=2):
    # CNN-BiLSTM with MultiHeadAttention
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)

    # CNN block
    conv_output = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    pool_output = MaxPooling1D(pool_size=2)(conv_output)
    
    # RNN block - return sequences for attention
    lstm_output = Bidirectional(LSTM(64, return_sequences=True))(pool_output) 
    
    # MultiHeadAttention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(
        query=lstm_output, value=lstm_output, key=lstm_output
    )
    
    # Aggregate attention output
    global_pool_output = GlobalMaxPooling1D()(attention_output)

    # Classifier
    x = Dense(64, activation='relu')(global_pool_output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Main Execution ---
results = {}

try:
    texts, labels = load_and_preprocess_data(DATASET_PATH)

    # Train/Test Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Text Processing - Keras Tokenizer
    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train_text)

    # +1 for padding/0 index, as Keras Embedding layer expects vocab_size to be the maximum index + 1
    vocab_size = len(tokenizer.word_index) + 1 

    X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_text)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # --- Train mandatory models ---
    model_configs = {
        "lstm": build_lstm_model,
        "bilstm": build_bilstm_model,
        "gru": build_gru_model,
        "cnn": build_cnn_model,
        "hybrid": build_hybrid_cnn_dense_model
    }

    for model_key, model_builder_fn in model_configs.items():
        results[model_key] = train_and_evaluate_model(
            model_builder_fn, X_train_padded, y_train, X_test_padded, y_test,
            vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, TRAINING_EPOCHS, BATCH_SIZE
        )

    # --- Additional Custom Model ("auto_model") ---
    auto_model_description = "CNN-BiLSTM with MultiHeadAttention: Embedding -> Conv1D -> MaxPooling1D -> Bidirectional LSTM (return sequences) -> MultiHeadAttention -> GlobalMaxPooling1D -> Dense -> Dropout -> Dense (sigmoid)."

    results["auto_model"] = train_and_evaluate_model(
        build_auto_model, X_train_padded, y_train, X_test_padded, y_test,
        vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, TRAINING_EPOCHS, BATCH_SIZE
    )
    results["auto_model"]["description"] = auto_model_description

except FileNotFoundError as e:
    # Re-raise FileNotFoundError as specified in HARD RULES
    raise e
except ValueError as e:
    # Re-raise ValueError as specified in HARD RULES for mapping failure
    raise e
except Exception as e:
    # Catch any other unexpected errors and re-raise.
    # The requirement is "Do NOT print logs, comments, or text outside JSON",
    # so we don't print any error messages here directly.
    raise e

# Final Output (ONLY JSON)
print(json.dumps(results), flush=True)
