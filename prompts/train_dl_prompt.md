You are an expert deep learning engineer using Keras/TensorFlow.
The dataset has already been preprocessed, tokenized, and padded.
Features (X_train, X_test) are located at:
- Train: '${features_npy_path}'
- Test: '${test_features_path}'
Labels (y_train, y_test) are located at:
- Train: '${labels_npy_path}'
- Test: '${test_labels_path}'
Tokenizer is saved at: '${vectorizer_pkl_path}'
Best hyperparameters are saved as a JSON file at: '${best_params_json_path}'

TASK:
- Write a Python script using **Keras/TensorFlow** to train multiple optimized Deep Learning models (LSTM, BiLSTM, GRU, 1D CNN, Hybrid CNN+Dense, TextCNN, Stacked LSTM, CNN-LSTM, auto_model) using the pre-tuned hyperparameters.
- SUPPRESS ALL TF WARNINGS at the top: `import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'`
- GPU ACCELERATION (MANDATORY FOR RTX 5070): Configure GPU memory growth to avoid OOM:
  ```python
  import tensorflow as tf
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
      for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
  print(f"[INFO] Using device: {'GPU' if gpus else 'CPU'} - {len(gpus)} GPU(s) available")
  ```
- CRITICAL FOR WINDOWS: Always protect script execution with `if __name__ == '__main__':` at the top level. Do NOT use `multiprocessing`. Use sequential training.

- ARCHITECTURE GUIDE (all models use Keras functional or sequential API):
  - LSTM: Embedding -> LSTM(units, dropout=dropout_rate) -> Dense(1, activation='sigmoid')
  - BiLSTM: Embedding -> Bidirectional(LSTM(units, dropout=dropout_rate)) -> Dense(1, activation='sigmoid')
  - GRU: Embedding -> GRU(units, dropout=dropout_rate) -> Dense(1, activation='sigmoid')
  - CNN: Embedding -> Conv1D(filters, kernel_size=5, activation='relu') -> GlobalMaxPooling1D -> Dropout(dropout_rate) -> Dense(1, activation='sigmoid')
  - Hybrid: Embedding -> Conv1D(filters, kernel_size=5, activation='relu') -> GlobalMaxPooling1D -> Dense(units, activation='relu') -> Dropout(dropout_rate) -> Dense(1, activation='sigmoid')
  - TextCNN (Multi-kernel): Use Keras Functional API with 3 parallel branches:
    - Input -> Embedding -> [Conv1D(kernel=3), Conv1D(kernel=4), Conv1D(kernel=5)] -> [GlobalMaxPooling1D x3] -> Concatenate -> Dropout(dropout_rate) -> Dense(1, activation='sigmoid')
  - Stacked LSTM: Embedding -> LSTM(units, return_sequences=True, dropout=dropout_rate) -> LSTM(units, dropout=dropout_rate) -> Dense(1, activation='sigmoid')
  - CNN-LSTM: Embedding -> Conv1D(filters, kernel_size=5, activation='relu') -> MaxPooling1D(pool_size=2) -> LSTM(units, dropout=dropout_rate) -> Dense(1, activation='sigmoid')
  - auto_model: An innovative architecture. CRITICAL DESIGN: Use a clean sequential or functional design such as:
    Input -> Embedding -> Bidirectional(GRU or LSTM) -> MultiHeadAttention(num_heads=4, key_dim=embedding_dim//4) -> GlobalMaxPooling1D -> Dense(64, relu) -> Dropout(dropout_rate) -> Dense(1, sigmoid)
    Ensure all dimensions are consistent (key_dim * num_heads == the sequence feature dim). If using residual connections, add a projection layer to match dimensions. AVOID dimension mismatch errors.

- DO NOT use `recurrent_dropout`. It disables cuDNN kernels on GPU. Use standard `dropout` only.

- LOADING DATA:
  - Load using `np.load(path)`.
  - Flatten labels to 1D using `np.ravel()` and cast to integer.
  - Vocabulary size is 15000, sequence max length is 400.

- LOADING HYPERPARAMETERS:
  - Load from '${best_params_json_path}' using `json` module.
  - Retrieve `embedding_dim`, `units`, `dropout_rate`, `learning_rate` for each model type.
  - For `text_cnn`, reuse cnn params. For `stacked_lstm`, reuse lstm params. For `cnn_lstm`, use cnn/lstm params.
  - If file is missing or values are empty, fallback: embedding_dim=128, units=128, dropout_rate=0.2, learning_rate=0.001.
  - filters = max(32, units // 2) when not provided directly.

- TRAINING SPECIFICATION:
  - Train each model for at most 5 epochs using EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True).
  - Use validation_split=0.1 during training.
  - Compile each model with: `model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])`
  - Fit with: `model.fit(X_train, y_train, epochs=5, batch_size=256, validation_split=0.1, callbacks=[early_stop], verbose=1)`

- HARDWARE RESOURCE MONITORING (MANDATORY):
  - Import `from resource_monitor import ResourceMonitor`
  - For each model: instantiate and start monitor BEFORE training, stop AFTER training.
  - `metrics = monitor.stop()` — do NOT call `get_metrics()`.
  - Include `avg_cpu_percent`, `max_cpu_percent`, `avg_ram_mb`, `max_ram_mb` in model metrics.

- MODELS TO TRAIN & SAVE:
  - Ensure models directory '${models_dir}' exists.
  - Save all models using `model.save(path)` in Keras `.h5` format:
    - lstm         -> '${models_dir}/lstm.h5'
    - bilstm       -> '${models_dir}/bilstm.h5'
    - gru          -> '${models_dir}/gru.h5'
    - cnn          -> '${models_dir}/cnn.h5'
    - hybrid       -> '${models_dir}/hybrid.h5'
    - text_cnn     -> '${models_dir}/text_cnn.h5'
    - stacked_lstm -> '${models_dir}/stacked_lstm.h5'
    - cnn_lstm     -> '${models_dir}/cnn_lstm.h5'
    - auto_model   -> '${models_dir}/auto_model.h5'
  - CRITICAL: Use ONLY `model.save(path)` (Keras format). Do NOT use PyTorch `torch.save`.

- EVALUATION (on test set after training):
  - For each model: `proba = model.predict(X_test, verbose=0).ravel(); preds = (proba >= 0.5).astype(int)`
  - Calculate `accuracy`, `precision`, `recall`, `f1` using sklearn.metrics.
  - Record `time_s` (wall-clock training time in seconds).

- OUTPUT:
  - At successful end, print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only).
  - Include a "model_descriptions" dict with actual architecture + hyperparameter values per model.
  - Dynamic descriptions: for auto_model, include exact layer sequence AND 1-2 sentences explaining WHY the architecture suits phishing detection.
<<<JSON_START>>>
{
"ok": true,
"models_trained": ["lstm", "bilstm", "gru", "cnn", "hybrid", "text_cnn", "stacked_lstm", "cnn_lstm", "auto_model"],
"best_model": "hybrid",
"model_descriptions": {
"lstm": "Embedding(128) -> LSTM(128, dropout=0.2) -> Dense(1, sigmoid)",
"bilstm": "Embedding(128) -> Bidirectional(LSTM(128, dropout=0.2)) -> Dense(1, sigmoid)",
"gru": "Embedding(128) -> GRU(128, dropout=0.2) -> Dense(1, sigmoid)",
"cnn": "Embedding(128) -> Conv1D(64 filters, kernel=5, relu) -> GlobalMaxPool -> Dropout(0.2) -> Dense(1, sigmoid)",
"hybrid": "Embedding(128) -> Conv1D(64 filters, kernel=5, relu) -> GlobalMaxPool -> Dense(128, relu) -> Dropout(0.2) -> Dense(1, sigmoid)",
"text_cnn": "Embedding(128) -> [Conv1D(kernel=3), Conv1D(kernel=4), Conv1D(kernel=5)] -> [GlobalMaxPool x3] -> Concat -> Dropout(0.2) -> Dense(1, sigmoid)",
"stacked_lstm": "Embedding(128) -> LSTM(128, return_seq=True, dropout=0.2) -> LSTM(128, dropout=0.2) -> Dense(1, sigmoid)",
"cnn_lstm": "Embedding(128) -> Conv1D(64 filters, kernel=5, relu) -> MaxPool -> LSTM(128, dropout=0.2) -> Dense(1, sigmoid)",
"auto_model": "<exact layer architecture with values>. <1-2 sentences: WHY this architecture suits phishing email detection>"
},
"metrics": {
"lstm": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94, "time_s": 45.1, "avg_cpu_percent": 15.2, "max_cpu_percent": 35.5, "avg_ram_mb": 220.4, "max_ram_mb": 280.1},
"bilstm": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 52.3, "avg_cpu_percent": 18.1, "max_cpu_percent": 39.2, "avg_ram_mb": 240.5, "max_ram_mb": 310.2},
"gru": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94, "time_s": 42.1, "avg_cpu_percent": 14.5, "max_cpu_percent": 32.1, "avg_ram_mb": 210.3, "max_ram_mb": 260.4},
"cnn": {"accuracy": 0.94, "precision": 0.93, "recall": 0.94, "f1": 0.93, "time_s": 30.5, "avg_cpu_percent": 10.2, "max_cpu_percent": 25.1, "avg_ram_mb": 150.1, "max_ram_mb": 190.2},
"hybrid": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94, "time_s": 65.2, "avg_cpu_percent": 16.4, "max_cpu_percent": 36.8, "avg_ram_mb": 230.2, "max_ram_mb": 290.5},
"text_cnn": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 35.0, "avg_cpu_percent": 12.0, "max_cpu_percent": 28.0, "avg_ram_mb": 180.0, "max_ram_mb": 220.0},
"stacked_lstm": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 90.0, "avg_cpu_percent": 22.0, "max_cpu_percent": 45.0, "avg_ram_mb": 260.0, "max_ram_mb": 320.0},
"cnn_lstm": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 70.0, "avg_cpu_percent": 18.0, "max_cpu_percent": 40.0, "avg_ram_mb": 240.0, "max_ram_mb": 300.0},
"auto_model": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 80.4, "avg_cpu_percent": 19.3, "max_cpu_percent": 41.5, "avg_ram_mb": 260.1, "max_ram_mb": 330.4}
}
}
<<<JSON_END>>>
- DO NOT output any explanation or markdown code fences other than ```python.
