import pandas as pd
import numpy as np
import json
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Embedding, Input, Flatten
)
# If the custom model uses attention:
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization


# --- Dataset Requirements ---
DATA_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset file not found at: {DATA_PATH}")

# Automatically detect text and label columns
text_cols = ["Email Text", "text", "body", "message"]
label_cols = ["Email Type", "label", "labels", "target", "Category"]

text_col = None
for col in text_cols:
    if col in df.columns:
        text_col = col
        break
if text_col is None:
    raise ValueError(f"Could not find a text column from candidates: {text_cols}")

label_col = None
for col in label_cols:
    if col in df.columns:
        label_col = col
        break
if label_col is None:
    raise ValueError(f"Could not find a label column from candidates: {label_cols}")

# Drop rows with NaN text or label BEFORE mapping
df.dropna(subset=[text_col, label_col], inplace=True)

# Label mapping
phishing_labels_variants = ["phishing", "phishing email", "spam", "malicious"]
safe_labels_variants = ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

# Create a temporary 'mapped_label' column and initialize with an invalid value
df['mapped_label'] = -1

# Case-insensitive mapping
for label_var in phishing_labels_variants:
    df.loc[df[label_col].astype(str).str.lower() == label_var, 'mapped_label'] = 1
for label_var in safe_labels_variants:
    df.loc[df[label_col].astype(str).str.lower() == label_var, 'mapped_label'] = 0

# Drop rows where mapping failed (i.e., 'mapped_label' is still -1)
df = df[df['mapped_label'] != -1]

X = df[text_col].astype(str).values
y = df['mapped_label'].values

# After mapping, check if both 0 and 1 exist
if not (0 in np.unique(y) and 1 in np.unique(y)):
    raise ValueError("Dataset mapping failed: need both phishing and safe labels")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Text Processing ---
vocab_size_limit = 10000 # Limit for tokenizer vocabulary size
oov_tok = "<OOV>"
max_len = 200 # As required

tokenizer = Tokenizer(num_words=vocab_size_limit, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1 # Actual vocab size after fitting, +1 for OOV or padding

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post', truncating='post')

# --- Model Training Parameters ---
embedding_dim = 128
epochs = 8
batch_size = 64

# --- Helper function to build mandatory models ---
def build_mandatory_model(model_type, vocab_size, embedding_dim, max_len):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))

    if model_type == "lstm":
        model.add(LSTM(128))
    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(64)))
    elif model_type == "gru":
        model.add(GRU(128))
    elif model_type == "cnn":
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
    elif model_type == "hybrid": # CNN + Dense
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- Helper function for training and evaluation ---
def train_and_evaluate_model(model, X_train_data, y_train_data, X_test_data, y_test_data):
    start_time = time.time()
    history = model.fit(X_train_data, y_train_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_data, y_test_data),
                        verbose=0) # Suppress verbose output

    training_time = time.time() - start_time

    # Get predictions
    y_pred_proba = model.predict(X_test_data)
    y_pred = (y_pred_proba > 0.5).astype("int32")

    # Calculate metrics
    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred, zero_division=0)
    recall = recall_score(y_test_data, y_pred, zero_division=0)
    f1 = f1_score(y_test_data, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test_data, y_pred).tolist()

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

results = {}

# --- Train and evaluate LSTM ---
lstm_model = build_mandatory_model("lstm", vocab_size, embedding_dim, max_len)
results["lstm"] = train_and_evaluate_model(lstm_model, X_train_padded, y_train, X_test_padded, y_test)

# --- Train and evaluate BiLSTM ---
bilstm_model = build_mandatory_model("bilstm", vocab_size, embedding_dim, max_len)
results["bilstm"] = train_and_evaluate_model(bilstm_model, X_train_padded, y_train, X_test_padded, y_test)

# --- Train and evaluate GRU ---
gru_model = build_mandatory_model("gru", vocab_size, embedding_dim, max_len)
results["gru"] = train_and_evaluate_model(gru_model, X_train_padded, y_train, X_test_padded, y_test)

# --- Train and evaluate 1D CNN ---
cnn_model = build_mandatory_model("cnn", vocab_size, embedding_dim, max_len)
results["cnn"] = train_and_evaluate_model(cnn_model, X_train_padded, y_train, X_test_padded, y_test)

# --- Train and evaluate Hybrid CNN + Dense ---
hybrid_model = build_mandatory_model("hybrid", vocab_size, embedding_dim, max_len)
results["hybrid"] = train_and_evaluate_model(hybrid_model, X_train_padded, y_train, X_test_padded, y_test)

# --- Additional Custom Model ("auto_model") ---
# Architecture: CNN (Conv1D + MaxPooling) -> GRU (return_sequences) -> MultiHeadAttention -> GlobalAveragePooling1D -> Dense (ReLU + Dropout) -> Dense (Sigmoid)
def build_auto_model(vocab_size, embedding_dim, max_len):
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)

    # First block: Conv1D and MaxPooling to extract local features and reduce sequence length
    x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Second block: GRU layer to capture sequential dependencies, returning sequences for attention
    x = GRU(64, return_sequences=True)(x)

    # Third block: MultiHeadAttention for capturing long-range dependencies across the sequence
    # Self-attention where query, key, value are all from the GRU output
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x, x)
    # Add residual connection and layer normalization for stable training
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Fourth block: Global Average Pooling to reduce sequence to a fixed-size vector
    x = GlobalAveragePooling1D()(x)
    # Dense layers for classification
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="auto_model")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

auto_model_description = "CNN (Conv1D + MaxPooling) -> GRU (return_sequences=True) -> MultiHeadAttention (4 heads) -> GlobalAveragePooling1D -> Dense (ReLU + Dropout) -> Dense (Sigmoid)"
auto_model_instance = build_auto_model(vocab_size, embedding_dim, max_len)
auto_model_metrics = train_and_evaluate_model(auto_model_instance, X_train_padded, y_train, X_test_padded, y_test)
auto_model_metrics["description"] = auto_model_description
results["auto_model"] = auto_model_metrics

# --- Final Output ---
print(json.dumps(results), flush=True)
