import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Embedding, Input, Flatten,
    MultiHeadAttention, LayerNormalization
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import json
import os
import re

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Dataset Requirements ---
EXACT_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'

# Load dataset
try:
    df = pd.read_csv(EXACT_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {EXACT_PATH}. Please ensure the path is correct.", flush=True)
    exit() # Exit if the core data isn't found

# Automatically detect text and label columns
TEXT_CANDIDATES = ["Email Text", "text", "body", "message"]
LABEL_CANDIDATES = ["Email Type", "label", "labels", "target", "Category"]

text_column = None
for col in TEXT_CANDIDATES:
    if col in df.columns:
        text_column = col
        break
if text_column is None:
    raise ValueError(f"Could not detect text column from candidates: {TEXT_CANDIDATES}")

label_column = None
for col in LABEL_CANDIDATES:
    if col in df.columns:
        label_column = col
        break
if label_column is None:
    raise ValueError(f"Could not detect label column from candidates: {LABEL_CANDIDATES}")

print(f"Detected text column: '{text_column}'", flush=True)
print(f"Detected label column: '{label_column}'", flush=True)

# Rename columns for easier access
df = df.rename(columns={text_column: 'text', label_column: 'label'})

# Drop rows with NaN in 'text' or 'label'
df.dropna(subset=['text', 'label'], inplace=True)

# Perform label mapping
# Phishing/spam → 1
# Ham/safe/legitimate → 0

def map_label(label_val):
    label_lower = str(label_val).lower()
    if 'phishing' in label_lower or 'spam' in label_lower:
        return 1
    elif 'ham' in label_lower or 'safe' in label_lower or 'legitimate' in label_lower:
        return 0
    return np.nan # Mark as unknown

df['label'] = df['label'].apply(map_label)

# Drop rows with unknown labels (NaN after mapping)
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# Dataset validation: must contain both 0 and 1
if not (0 in df['label'].unique() and 1 in df['label'].unique()):
    raise ValueError("Dataset mapping failed: need both phishing and safe labels")

print(f"Dataset loaded and preprocessed. Total samples: {len(df)}", flush=True)
print(f"Label distribution: \n{df['label'].value_counts()}", flush=True)

# --- Train/Test Split ---
X = df['text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}", flush=True)
print(f"Train label distribution: \n{pd.Series(y_train).value_counts()}", flush=True)
print(f"Test label distribution: \n{pd.Series(y_test).value_counts()}", flush=True)

# --- Tokenizer and Padding ---
MAX_LEN = 200 # Max length for padded sequences
VOCAB_SIZE = 10000 # Max number of words to keep, based on word frequency
EMBEDDING_DIM = 128 # Dimension for word embeddings

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

# Update VOCAB_SIZE to actual vocabulary size used by tokenizer + 1 for padding
vocab_size_actual = min(len(tokenizer.word_index) + 1, VOCAB_SIZE)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Vocabulary size: {vocab_size_actual}", flush=True)
print(f"Training sequences padded to length {MAX_LEN}", flush=True)

# --- Model Training and Evaluation Function ---
def train_and_evaluate_model(model_name, model_builder_func, X_train_data, y_train_data, X_test_data, y_test_data, epochs=10, batch_size=32):
    print(f"\n--- Training {model_name} ---", flush=True)
    model = model_builder_func()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary(print_fn=lambda x: print(x, flush=True)) # Ensure model summary is flushed

    start_time = time.time()
    history = model.fit(
        X_train_data, y_train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use 10% of training data for validation during training
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )
    training_time = time.time() - start_time

    # Evaluate on the test set
    y_pred_proba = model.predict(X_test_data)
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test_data, y_pred)
    prec = precision_score(y_test_data, y_pred, zero_division=0) # Handle cases where no positive predictions
    rec = recall_score(y_test_data, y_pred, zero_division=0)
    f1 = f1_score(y_test_data, y_pred, zero_division=0)
    cm = confusion_matrix(y_test_data, y_pred).tolist() # Convert to list for JSON serialization

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "train_accuracy_history": history.history['accuracy'],
        "val_accuracy_history": history.history['val_accuracy'],
        "training_time": training_time
    }
    print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}", flush=True)
    return results

all_results = {}

# --- Deep Learning Models to Train (Mandatory) ---

# LSTM Model
def build_lstm_model():
    model = Sequential([
        Embedding(vocab_size_actual, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model
all_results['lstm'] = train_and_evaluate_model('LSTM', build_lstm_model, X_train_padded, y_train, X_test_padded, y_test)

# BiLSTM Model
def build_bilstm_model():
    model = Sequential([
        Embedding(vocab_size_actual, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(128)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model
all_results['bilstm'] = train_and_evaluate_model('BiLSTM', build_bilstm_model, X_train_padded, y_train, X_test_padded, y_test)

# GRU Model
def build_gru_model():
    model = Sequential([
        Embedding(vocab_size_actual, EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model
all_results['gru'] = train_and_evaluate_model('GRU', build_gru_model, X_train_padded, y_train, X_test_padded, y_test)

# 1D CNN Model
def build_cnn_model():
    model = Sequential([
        Embedding(vocab_size_actual, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model
all_results['cnn'] = train_and_evaluate_model('1D CNN', build_cnn_model, X_train_padded, y_train, X_test_padded, y_test)

# Hybrid CNN + Dense
def build_hybrid_model():
    model = Sequential([
        Embedding(vocab_size_actual, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(), # Flatten to connect convolutional layers to dense layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model
all_results['hybrid'] = train_and_evaluate_model('Hybrid CNN + Dense', build_hybrid_model, X_train_padded, y_train, X_test_padded, y_test)

# --- Additional Custom Model ("auto_model") ---
def build_auto_model():
    # A custom model combining Conv1D, BiGRU, and MultiHeadAttention
    # This architecture aims to leverage local feature extraction (CNN), sequential context (BiGRU),
    # and global dependencies (Attention).
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(vocab_size_actual, EMBEDDING_DIM, input_length=MAX_LEN)(inputs)

    # Conv1D for local feature extraction
    conv_output = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)

    # Bidirectional GRU for sequential context, return sequences for attention
    gru_output = Bidirectional(GRU(64, return_sequences=True))(conv_output)

    # MultiHeadAttention layer (self-attention)
    # The output from GRU acts as query, key, and value for self-attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(gru_output, gru_output, gru_output)
    
    # Add & Normalize: Residual connection followed by Layer Normalization
    normalized_output = LayerNormalization(epsilon=1e-6)(gru_output + attention_output)

    # Global Average Pooling to reduce dimensionality before Dense layers
    pooled_output = GlobalAveragePooling1D()(normalized_output)

    # Dense layers for classification
    x = Dense(64, activation='relu')(pooled_output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

auto_model_description = (
    "A hybrid model featuring an Embedding layer, followed by a Conv1D for local feature extraction. "
    "The Conv1D output is then processed by a Bidirectional GRU to capture sequential dependencies. "
    "A MultiHeadAttention layer performs self-attention on the GRU's output, allowing the model "
    "to weigh different parts of the sequence. A residual connection and LayerNormalization are applied "
    "after attention. Finally, GlobalAveragePooling1D is used before a Dense classifier layer."
)

auto_model_results = train_and_evaluate_model('auto_model', build_auto_model, X_train_padded, y_train, X_test_padded, y_test)
auto_model_results['description'] = auto_model_description
all_results['auto_model'] = auto_model_results

# --- Optional Model (DistilBERT) ---
try:
    import transformers
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    print("\n--- Training DistilBERT (Optional Model) ---", flush=True)

    # Load DistilBERT tokenizer and model
    distilbert_model_name = "distilbert-base-uncased"
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
    
    # Tokenize data for DistilBERT
    # Ensure all inputs are strings before encoding
    X_train_str = [str(text) for text in X_train]
    X_test_str = [str(text) for text in X_test]

    train_encodings = distilbert_tokenizer(X_train_str, truncation=True, padding=True, max_length=MAX_LEN, return_tensors='tf')
    test_encodings = distilbert_tokenizer(X_test_str, truncation=True, padding=True, max_length=MAX_LEN, return_tensors='tf')

    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    ))
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    ))

    # Set up batching
    batch_size = 16 # Typically smaller for BERT-like models
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build and compile the TF DistilBERT model
    # num_labels=2 for binary classification with SparseCategoricalCrossentropy
    distilbert_model = TFAutoModelForSequenceClassification.from_pretrained(distilbert_model_name, num_labels=2) 
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) # Common LR for BERT fine-tuning
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    distilbert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    print("DistilBERT model summary:", flush=True)
    distilbert_model.summary(print_fn=lambda x: print(x, flush=True))

    start_time = time.time()
    history = distilbert_model.fit(
        train_dataset,
        epochs=3, # Often fewer epochs for fine-tuning BERT-like models
        validation_data=test_dataset, # Using test_dataset directly as validation here
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)]
    )
    training_time = time.time() - start_time

    # Evaluate DistilBERT
    raw_predictions = distilbert_model.predict(test_dataset).logits
    # For binary classification (num_labels=2), get probability of class 1
    y_pred_proba_distilbert = tf.nn.softmax(raw_predictions, axis=-1).numpy()[:, 1]
    y_pred_distilbert = (y_pred_proba_distilbert > 0.5).astype(int)

    acc_db = accuracy_score(y_test, y_pred_distilbert)
    prec_db = precision_score(y_test, y_pred_distilbert, zero_division=0)
    rec_db = recall_score(y_test, y_pred_distilbert, zero_division=0)
    f1_db = f1_score(y_test, y_pred_distilbert, zero_division=0)
    cm_db = confusion_matrix(y_test, y_pred_distilbert).tolist()

    all_results['distilbert'] = {
        "accuracy": acc_db,
        "precision": prec_db,
        "recall": rec_db,
        "f1": f1_db,
        "confusion_matrix": cm_db,
        "train_accuracy_history": history.history['accuracy'], 
        "val_accuracy_history": history.history['val_accuracy'],
        "training_time": training_time
    }
    print(f"DistilBERT - Accuracy: {acc_db:.4f}, Precision: {prec_db:.4f}, Recall: {rec_db:.4f}, F1: {f1_db:.4f}", flush=True)

except ImportError:
    print("\n'transformers' library not installed. Skipping DistilBERT.", flush=True)
except Exception as e:
    print(f"\nAn error occurred during DistilBERT training: {e}. Skipping DistilBERT.", flush=True)


# --- Final Output ---
# Ensure numpy types are converted to standard Python types for JSON serialization
def convert_numpy_to_python_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_to_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python_types(value) for key, value in obj.items()}
    return obj

final_results = convert_numpy_to_python_types(all_results)
print(json.dumps(final_results, indent=2), flush=True)
