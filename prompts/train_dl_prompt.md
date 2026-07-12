You are an expert deep learning engineer.
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
- Write a Python script using TensorFlow/Keras to train multiple optimized Deep Learning models (LSTM, BiLSTM, GRU, 1D CNN, Hybrid CNN+Dense, and auto_model) using the pre-tuned hyperparameters.
- ARCHITECTURE GUIDE:
  - LSTM: Embedding -> LSTM -> Dense(1, activation='sigmoid')
  - BiLSTM: Embedding -> Bidirectional(LSTM) -> Dense(1, activation='sigmoid')
  - GRU: Embedding -> GRU -> Dense(1, activation='sigmoid')
  - CNN: Embedding -> Conv1D -> GlobalMaxPooling1D -> Dropout -> Dense(1, activation='sigmoid')
  - Hybrid: Embedding -> Conv1D -> GlobalMaxPooling1D -> Dense(dense_units) -> Dropout -> Dense(1, activation='sigmoid')
  - TextCNN (Multi-kernel): Embedding -> [Conv1D(kernel=3), Conv1D(kernel=4), Conv1D(kernel=5)] (parallel) -> [GlobalMaxPooling1D x3] -> Concatenate -> Dropout -> Dense(1, activation='sigmoid')
    Use Keras Functional API for the parallel branches. Key params: embedding_dim, filters (use same for all kernels), dropout_rate.
  - Stacked LSTM: Embedding -> LSTM(return_sequences=True) -> LSTM -> Dense(1, activation='sigmoid')
    Two stacked LSTM layers. Key params: same embedding_dim and units from lstm best_params.
  - CNN-LSTM: Embedding -> Conv1D -> MaxPooling1D -> LSTM -> Dense(1, activation='sigmoid')
    Conv1D extracts local n-gram features; LSTM models sequential dependencies over those features. Key params: embedding_dim, filters, units, dropout_rate.
  - auto_model: An innovative architecture designed by you (e.g. self-attention, multi-branch, or residual blocks).
    **CRITICAL DESIGN GUIDELINE FOR AUTO_MODEL**: To prevent shape broadcasting / dimension mismatch errors (e.g., trying to combine dimension 90 and 64), ensure that all parallel branches or residual connections have EXACTLY matching dimensions before any element-wise operations (like Add, Multiply, Concatenate). Alternatively, design a simple yet powerful sequential model such as:
    Embedding -> Bidirectional(LSTM or GRU) -> MultiHeadAttention -> GlobalMaxPooling1D -> Dense -> Dropout -> Dense(1, sigmoid), ensuring that all attention query/value sizes and dense units align perfectly with the embedding dimension. If using a custom layer like BahdanauAttention, keep the input dimensions consistent throughout.
- Load X_train, X_test, y_train, y_test:
  - Load using `np.load(path)`.
  - CRITICAL: DO NOT use `np.lib.npy.lib.format.NpzFile` or deep internal NumPy format paths to prevent compatibility errors. Instead, check `isinstance(data, np.ndarray)` or `hasattr(data, 'files')` to safely distinguish dense arrays from NPZ archive objects.
  - If loading fails, try fallback to `scipy.sparse.load_npz(path).toarray()`.
  - Flatten labels to 1D using `np.ravel()` and cast to integer.
  - Vocabulary size is 5000, and sequence max length is 200.

- LOADING HYPERPARAMETERS:
  - Load the pre-tuned hyperparameters from '${best_params_json_path}' using python `json` module.
  - Retrieve parameters for `lstm`, `bilstm`, `gru`, `cnn`, `hybrid`, and `auto_model`.
  - For `text_cnn`, reuse cnn params (embedding_dim, dropout_rate) and add filters if available.
  - For `stacked_lstm`, reuse lstm params (embedding_dim, units, dropout_rate).
  - For `cnn_lstm`, reuse cnn/lstm params as appropriate.
  - Parameters include: `embedding_dim`, `units`, `dropout_rate`, and `learning_rate`.
  - If the file is missing or values are empty, fallback to reasonable defaults (e.g. embedding_dim=64, units=64, dropout_rate=0.2, learning_rate=0.001).

- TRAINING SPECIFICATION:
  - Train each model on the FULL training set for at most 5 epochs using EarlyStopping (patience=1).
  - DO NOT use `recurrent_dropout` in LSTM, BiLSTM, or GRU layers. It disables optimized CPU kernels and makes CPU training extremely slow. Use standard `dropout` instead.

- DONANIM KAYNAK İZLEME (RESOURCE MONITORING - MANDATORY):
  - Import the `ResourceMonitor` class from `resource_monitor` (`from resource_monitor import ResourceMonitor`).
  - For each model type:
    - Instantiate and start the monitor *before* training begins (`monitor = ResourceMonitor()`, `monitor.start()`).
    - Stop the monitor *after* the model finishes training.
    - CRITICAL: The `stop()` method returns the monitored resource metrics dictionary directly. Do NOT call `get_metrics()`. Instead, use: `metrics = monitor.stop()`.
    - Include these four metrics under the model's metrics dictionary in the final JSON output.

- MODELS TO TRAIN & SAVE:
  - Ensure the models directory '${models_dir}' exists (create if missing).
  - Save all final models to:
    - lstm         -> '${models_dir}/lstm.h5'
    - bilstm       -> '${models_dir}/bilstm.h5'
    - gru          -> '${models_dir}/gru.h5'
    - cnn          -> '${models_dir}/cnn.h5'
    - hybrid       -> '${models_dir}/hybrid.h5'
    - text_cnn     -> '${models_dir}/text_cnn.h5'
    - stacked_lstm -> '${models_dir}/stacked_lstm.h5'
    - cnn_lstm     -> '${models_dir}/cnn_lstm.h5'
    - auto_model   -> '${models_dir}/auto_model.h5'
  - CRITICAL: In Keras 3/TensorFlow 2.21+, `model.save()` does NOT support the `custom_objects` argument. Save the models simply using `model.save(path)`. DO NOT pass `custom_objects` to `model.save()`.

- EVALUATION:
  - Calculate `accuracy`, `precision`, `recall`, `f1`, and `time_s` (training time in seconds) on the test set.

- OUTPUT:
  - At successful end of execution, print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only). Include a "model_descriptions" dictionary detailing the specific architecture and hyperparameters for each trained model.
  - CRITICAL (Dynamic Descriptions): The "description" for each model must NOT be a static string. It must dynamically reflect the actual architecture and hyperparameter values:
    - For standard models (lstm, bilstm, gru, cnn): layer-by-layer format with actual values (e.g. "Embedding(64) -> LSTM(128, dropout=0.2) -> Dense(1, sigmoid)").
    - For hybrid: MUST list all component layers and their roles (e.g. "Embedding(64) -> Conv1D(52 filters, kernel=5, relu) -> GlobalMaxPool -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)").
    - For text_cnn: MUST show all 3 parallel branches and their kernel sizes.
    - For stacked_lstm: MUST show both LSTM layers with their units.
    - For cnn_lstm: MUST show the Conv1D->MaxPool->LSTM sequence with actual values.
    - For auto_model: MUST include (1) exact layer-by-layer architecture AND (2) 1-2 sentences explaining WHY this architecture was chosen for phishing email detection.
    <<<JSON_START>>>
    {
    "ok": true,
    "models_trained": ["lstm", "bilstm", "gru", "cnn", "hybrid", "text_cnn", "stacked_lstm", "cnn_lstm", "auto_model"],
    "best_model": "bilstm",
    "model_descriptions": {
    "lstm": "Embedding(64) -> LSTM(128, dropout=0.2) -> Dense(1, sigmoid)",
    "bilstm": "Embedding(64) -> Bidirectional(LSTM(128, dropout=0.2)) -> Dense(1, sigmoid)",
    "gru": "Embedding(64) -> GRU(128, dropout=0.2) -> Dense(1, sigmoid)",
    "cnn": "Embedding(64) -> Conv1D(64 filters, kernel=5, relu) -> GlobalMaxPool -> Dropout(0.2) -> Dense(1, sigmoid)",
    "hybrid": "Embedding(64) -> Conv1D(64 filters, kernel=5, relu) -> GlobalMaxPool -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)",
    "text_cnn": "Embedding(64) -> [Conv1D(kernel=3), Conv1D(kernel=4), Conv1D(kernel=5)] -> [GlobalMaxPool x3] -> Concat -> Dropout(0.2) -> Dense(1, sigmoid)",
    "stacked_lstm": "Embedding(64) -> LSTM(128, return_seq=True, dropout=0.2) -> LSTM(128, dropout=0.2) -> Dense(1, sigmoid)",
    "cnn_lstm": "Embedding(64) -> Conv1D(64 filters, kernel=5, relu) -> MaxPool -> LSTM(128, dropout=0.2) -> Dense(1, sigmoid)",
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
