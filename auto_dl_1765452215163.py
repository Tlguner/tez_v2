import pandas as pd
import numpy as np
import re
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Embedding,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Dataset Requirements ---
DATA_PATH = 'E:/deu/tez/proje/scraped data/phishing_mcp_server/Phishing_Email.csv'
TEXT_COLS = ["Email Text", "text", "body", "message"]
LABEL_COLS = ["Email Type", "label", "labels", "target", "Category"]

def load_and_preprocess_data(file_path):
    """
    Loads the dataset, detects text and label columns, performs label mapping,
    and handles missing/unknown values.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        raise

    # Automatically detect text column
    text_column = None
    for col in TEXT_COLS:
        if col in df.columns:
            text_column = col
            break
    if text_column is None:
        raise ValueError(f"Could not find text column from {TEXT_COLS}")

    # Automatically detect label column
    label_column = None
    for col in LABEL_COLS:
        if col in df.columns:
            label_column = col
            break
    if label_column is None:
        raise ValueError(f"Could not find label column from {LABEL_COLS}")

    # Drop rows with NaN in text or label
    df.dropna(subset=[text_column, label_column], inplace=True)

    # Perform label mapping
    df['mapped_label'] = -1  # Temporary value for unknown labels
    phishing_keywords = ['phishing', 'spam']
    ham_keywords = ['ham', 'safe', 'legitimate']

    df['mapped_label'] = df[label_column].astype(str).str.lower().apply(lambda x:
        1 if any(k in x for k in phishing_keywords) else
        0 if any(k in x for k in ham_keywords) else
        -1
    )

    # Drop rows with unknown labels (-1)
    df = df[df['mapped_label'] != -1]

    # Validate: Dataset must contain both 0 and 1 after mapping
    if 0 not in df['mapped_label'].unique() or 1 not in df['mapped_label'].unique():
        raise ValueError("Dataset mapping failed: need both phishing and safe labels")

    X = df[text_column].values
    y = df['mapped_label'].values

    # Simple text cleaning: convert to lowercase and remove special characters
    X = np.vectorize(lambda text: re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower()))(X)

    return X, y

# --- Hyperparameters ---
MAX_LEN = 200  # Max sequence length for padding
VOCAB_SIZE = 15000  # Max number of words to keep in vocabulary
EMBEDDING_DIM = 128 # Dimension of the word embeddings
EPOCHS = 10
BATCH_SIZE = 32

# --- Model Builder Functions ---
def build_lstm_model(vocab_size, embedding_dim, max_len):
    """Builds a simple LSTM model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(128),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_bilstm_model(vocab_size, embedding_dim, max_len):
    """Builds a Bidirectional LSTM model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gru_model(vocab_size, embedding_dim, max_len):
    """Builds a simple GRU model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GRU(128),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_cnn_model(vocab_size, embedding_dim, max_len):
    """Builds a 1D CNN model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_hybrid_cnn_dense_model(vocab_size, embedding_dim, max_len):
    """Builds a Hybrid CNN + Dense model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        GlobalMaxPooling1D(), # Aggregate features
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_auto_model(vocab_size, embedding_dim, max_len):
    """
    AI-designed custom model: CNN-GRU with Multi-Head Self-Attention.
    Architecture combines local feature extraction (CNN), sequential context (GRU),
    and adaptive focus on important parts of the sequence (Multi-Head Attention).
    It includes residual connections and Layer Normalization typical of Transformer blocks.
    """
    input_layer = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_len)(input_layer)

    # CNN Block for local feature extraction
    x_cnn = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Dropout(0.3)(x_cnn)

    # GRU Block to capture sequential context
    # return_sequences=True is crucial for attention mechanism to operate on each timestep
    x_gru = GRU(128, return_sequences=True)(x_cnn)
    x_gru = Dropout(0.3)(x_gru)

    # Multi-Head Self-Attention Block
    num_heads_auto = 4
    # key_dim should be a factor of the input feature dimension for MultiHeadAttention
    # Here, input features dimension is 128 (from GRU units)
    attn_key_dim = 128 // num_heads_auto if 128 % num_heads_auto == 0 else 128 # Ensure it divides cleanly
    if attn_key_dim == 0: attn_key_dim = 1 # Avoid zero dimension

    attn_output = MultiHeadAttention(num_heads=num_heads_auto, key_dim=attn_key_dim)(query=x_gru, value=x_gru, key=x_gru)
    attn_output = Dropout(0.3)(attn_output)

    # Add & Norm: Residual connection followed by Layer Normalization
    x_attn = LayerNormalization(epsilon=1e-6)(x_gru + attn_output) # Residual connection for attention

    # Feed-Forward Network (applied to each position independently)
    ff_output = Dense(256, activation="relu")(x_attn)
    ff_output = Dense(128)(ff_output) # Output dimension matches input x_attn
    ff_output = Dropout(0.3)(ff_output)

    # Add & Norm: Residual connection followed by Layer Normalization
    x_final = LayerNormalization(epsilon=1e-6)(x_attn + ff_output) # Residual connection for feed-forward

    # Aggregate sequence features for classification
    x_pooled = GlobalMaxPooling1D()(x_final)

    # Output layer
    output_layer = Dense(1, activation='sigmoid')(x_pooled)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# --- Evaluation Function ---
def evaluate_model(model, X_test, y_test, history, start_time):
    """
    Evaluates a trained Keras model and computes required metrics.
    """
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON serialization

    training_time = time.time() - start_time

    # Extract train/val accuracy history
    train_acc_history = history.history.get('accuracy', [])
    val_acc_history = history.history.get('val_accuracy', [])

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "train_val_accuracy_history": {
            "train": [float(val) for val in train_acc_history],
            "val": [float(val) for val in val_acc_history]
        },
        "training_time_seconds": float(training_time)
    }

def main():
    results = {}

    # 1. Load and Preprocess Data
    X, y = load_and_preprocess_data(DATA_PATH)
    print(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} unique labels.")
    print(f"Label distribution: Class 0: {np.bincount(y)[0]}, Class 1: {np.bincount(y)[1]}")

    # 2. Train/Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train label distribution: Class 0: {np.bincount(y_train)[0]}, Class 1: {np.bincount(y_train)[1]}")
    print(f"Test label distribution: Class 0: {np.bincount(y_test)[0]}, Class 1: {np.bincount(y_test)[1]}")

    # 3. Tokenization and Padding
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(X_train)  # Fit tokenizer on training data only

    # Update VOCAB_SIZE to actual vocabulary size discovered by tokenizer (+1 for 0-index padding)
    current_vocab_size = len(tokenizer.word_index) + 1
    print(f"Actual vocabulary size used by tokenizer: {current_vocab_size}")

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Define common compilation settings and callbacks for Keras models
    optimizer = Adam(learning_rate=0.001)
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    # Early stopping to prevent overfitting and speed up training
    callbacks = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=0)]

    # --- Train and Evaluate Mandatory Deep Learning Models ---
    models_to_train = {
        "lstm": build_lstm_model,
        "bilstm": build_bilstm_model,
        "gru": build_gru_model,
        "cnn": build_cnn_model,
        "hybrid": build_hybrid_cnn_dense_model,
        "auto_model": build_auto_model, # Custom model
    }

    for model_name, model_builder in models_to_train.items():
        print(f"\nTraining {model_name}...", end=' ')
        tf.keras.backend.clear_session() # Clear TF session for a fresh model
        model = model_builder(current_vocab_size, EMBEDDING_DIM, MAX_LEN)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # model.summary() # Uncomment to see model summary for each model

        start_time = time.time()
        history = model.fit(
            X_train_padded, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,  # Use 10% of training data for validation
            callbacks=callbacks,
            verbose=0  # Suppress verbose output during training
        )
        model_results = evaluate_model(model, X_test_padded, y_test, history, start_time)
        results[model_name] = model_results
        if model_name == "auto_model":
            results[model_name]["description"] = "Custom model: CNN-GRU with Multi-Head Self-Attention, residual connections, and Layer Normalization."
        print(f"Completed. Accuracy: {model_results['accuracy']:.4f}")

    # --- Optional Model (DistilBERT) ---
    try:
        from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
        print("\nTraining DistilBERT...", end=' ')

        # DistilBERT specific tokenization
        dbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        # DistilBERT typically uses a shorter max sequence length
        dbert_max_len = 128

        # Tokenize inputs for DistilBERT
        X_train_encodings = dbert_tokenizer(
            list(X_train), truncation=True, padding='max_length', max_length=dbert_max_len, return_tensors='tf'
        )
        X_test_encodings = dbert_tokenizer(
            list(X_test), truncation=True, padding='max_length', max_length=dbert_max_len, return_tensors='tf'
        )

        # Create TensorFlow datasets for DistilBERT fine-tuning
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(X_train_encodings),
            y_train
        )).shuffle(len(X_train)).batch(BATCH_SIZE) # Shuffle with buffer size equal to dataset size

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(X_test_encodings),
            y_test
        )).batch(BATCH_SIZE)

        # Build and compile DistilBERT model for sequence classification
        dbert_model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=2 # Binary classification
        )
        dbert_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) # Common fine-tuning learning rate for transformers
        dbert_model.compile(
            optimizer=dbert_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Model outputs logits
            metrics=['accuracy']
        )

        dbert_callbacks = [EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True, verbose=0)]

        start_time = time.time()
        # Fine-tune DistilBERT model
        dbert_history = dbert_model.fit(
            train_dataset,
            epochs=3, # Typically fewer epochs for fine-tuning pre-trained models
            validation_data=test_dataset, # Use test_dataset for validation during training
            callbacks=dbert_callbacks,
            verbose=0
        )

        # Evaluate DistilBERT on the test set
        dbert_raw_preds = dbert_model.predict(test_dataset).logits
        dbert_y_pred_proba = tf.nn.softmax(dbert_raw_preds, axis=-1).numpy()[:, 1] # Probability of class 1
        dbert_y_pred = (dbert_y_pred_proba > 0.5).astype(int)

        dbert_acc = accuracy_score(y_test, dbert_y_pred)
        dbert_prec = precision_score(y_test, dbert_y_pred, zero_division=0)
        dbert_rec = recall_score(y_test, dbert_y_pred, zero_division=0)
        dbert_f1 = f1_score(y_test, dbert_y_pred, zero_division=0)
        dbert_cm = confusion_matrix(y_test, dbert_y_pred).tolist()

        dbert_training_time = time.time() - start_time

        dbert_train_acc_history = dbert_history.history.get('accuracy', [])
        dbert_val_acc_history = dbert_history.history.get('val_accuracy', [])


        results["distilbert"] = {
            "accuracy": float(dbert_acc),
            "precision": float(dbert_prec),
            "recall": float(dbert_rec),
            "f1": float(dbert_f1),
            "confusion_matrix": dbert_cm,
            "train_val_accuracy_history": {
                "train": [float(val) for val in dbert_train_acc_history],
                "val": [float(val) for val in dbert_val_acc_history]
            },
            "training_time_seconds": float(dbert_training_time)
        }
        print(f"Completed. Accuracy: {dbert_acc:.4f}")

    except ImportError:
        print("\n'transformers' library not installed. Skipping DistilBERT.")
    except Exception as e:
        print(f"\nAn error occurred during DistilBERT training: {e}")
        # Not adding 'distilbert' key if an error occurs to adhere to "only if trained"
        pass

    # Final Output (JSON)
    print(json.dumps(results, indent=4), flush=True)

if __name__ == "__main__":
    main()
