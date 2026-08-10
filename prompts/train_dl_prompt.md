You are an expert deep learning engineer using PyTorch and CUDA GPU acceleration.
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
- Write a Python script using **PyTorch** with **NATIVE CUDA GPU ACCELERATION** to train multiple optimized Deep Learning models (LSTM, BiLSTM, GRU, 1D CNN, Hybrid CNN+Dense, TextCNN, Stacked LSTM, CNN-LSTM, auto_model) using the pre-tuned hyperparameters.

MANDATORY GPU ACCELERATION FOR RTX 5070:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

CRITICAL FOR WINDOWS MULTIPROCESSING:
- Always wrap script execution inside `if __name__ == '__main__':`.
- In PyTorch `DataLoader(..., num_workers=0, pin_memory=True)` ALWAYS set `num_workers=0` to prevent Windows deadlocks.

CRITICAL MODEL SERIALIZATION RULE:
- At the START of your script, write a helper Python module file named `auto_DL_models.py` (in the current directory) containing all PyTorch `nn.Module` class definitions (`LSTMModel`, `BiLSTMModel`, `GRUModel`, `CNNModel`, `HybridModel`, `TextCNNModel`, `StackedLSTMModel`, `CNNLSTMModel`, `AutoModel`).
- Also import `auto_DL_models` in your script: `from auto_DL_models import *`.
- This ensures that `torch.save(model, path)` can be unpickled by ANY script (including evaluation and visualization scripts) without `AttributeError`.

PYTORCH MODEL ARCHITECTURE SPECIFICATIONS (nn.Module):
- Vocabulary size: 15000, max sequence length: 400.
- All models output a single logit/sigmoid value representing phishing probability:
  - LSTM: `nn.Embedding(15000, embed_dim)` -> `nn.LSTM(embed_dim, units, batch_first=True, dropout=dropout_rate)` -> `nn.Linear(units, 1)` -> `nn.Sigmoid()`
  - BiLSTM: `nn.Embedding` -> `nn.LSTM(..., bidirectional=True)` -> `nn.Linear(units*2, 1)` -> `nn.Sigmoid()`
  - GRU: `nn.Embedding` -> `nn.GRU(embed_dim, units, batch_first=True, dropout=dropout_rate)` -> `nn.Linear(units, 1)` -> `nn.Sigmoid()`
  - CNN: `nn.Embedding` -> `nn.Conv1d(embed_dim, filters, kernel_size=5)` -> `nn.AdaptiveMaxPool1d(1)` -> `nn.Dropout` -> `nn.Linear(filters, 1)` -> `nn.Sigmoid()`
  - Hybrid: `nn.Embedding` -> `nn.Conv1d` -> `nn.AdaptiveMaxPool1d(1)` -> `nn.Linear(filters, units)` -> `nn.ReLU()` -> `nn.Dropout` -> `nn.Linear(units, 1)` -> `nn.Sigmoid()`
  - TextCNN (Multi-kernel): 3 parallel branches: `nn.Conv1d` with kernel sizes 3, 4, 5. AdaptiveMaxPool1d each branch, concatenate outputs -> `nn.Dropout` -> `nn.Linear(filters*3, 1)` -> `nn.Sigmoid()`
  - Stacked LSTM: `nn.Embedding` -> `nn.LSTM(embed_dim, units, num_layers=2, batch_first=True, dropout=dropout_rate)` -> `nn.Linear(units, 1)` -> `nn.Sigmoid()`
  - CNN-LSTM: `nn.Embedding` -> `nn.Conv1d` -> `nn.MaxPool1d(2)` -> `nn.LSTM` -> `nn.Linear` -> `nn.Sigmoid()`
  - auto_model: An innovative PyTorch architecture, e.g., `nn.Embedding` -> `nn.GRU(..., bidirectional=True)` -> `nn.MultiheadAttention` -> `nn.AdaptiveMaxPool1d(1)` -> `nn.Linear(units, 64)` -> `nn.ReLU()` -> `nn.Dropout` -> `nn.Linear(64, 1)` -> `nn.Sigmoid()`. Ensure query, key, value dimensions match `embed_dim`.

DATA LOADING & GPU DATASETS:
- Load `X_train`, `X_test`, `y_train`, `y_test` using `np.load(path)`.
- Flatten labels using `np.ravel()`.
- Convert to torch Tensors:
  `train_x = torch.tensor(X_train, dtype=torch.long)`
  `train_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)`
- Create `TensorDataset` and `DataLoader(..., batch_size=256, shuffle=True, num_workers=0)`.

TRAINING LOOP SPECIFICATION:
- Optimizer: `torch.optim.Adam(model.parameters(), lr=learning_rate)`
- Loss function: `nn.BCELoss()` or `nn.BCEWithLogitsLoss()`
- Move model and batch tensors to `device`: `model.to(device)`, `inputs.to(device)`, `targets.to(device)`.
- Train each model on GPU for at most 5 epochs with EarlyStopping (stop if train loss does not improve for 1 epoch).

HARDWARE RESOURCE MONITORING (MANDATORY):
- Import `from resource_monitor import ResourceMonitor`
- For each model: instantiate and start monitor BEFORE training, stop AFTER training.
- `metrics = monitor.stop()` — do NOT call `get_metrics()`.
- Include `avg_cpu_percent`, `max_cpu_percent`, `avg_ram_mb`, `max_ram_mb` in model metrics.

SAVING MODELS:
- Ensure models directory '${models_dir}' exists.
- Save each trained model using `torch.save(model, path)`:
  - lstm         -> '${models_dir}/lstm.h5' (or .pt)
  - bilstm       -> '${models_dir}/bilstm.h5'
  - gru          -> '${models_dir}/gru.h5'
  - cnn          -> '${models_dir}/cnn.h5'
  - hybrid       -> '${models_dir}/hybrid.h5'
  - text_cnn     -> '${models_dir}/text_cnn.h5'
  - stacked_lstm -> '${models_dir}/stacked_lstm.h5'
  - cnn_lstm     -> '${models_dir}/cnn_lstm.h5'
  - auto_model   -> '${models_dir}/auto_model.h5'

EVALUATION (on test set after training):
- Evaluate model on test set using `model.eval()` and `with torch.no_grad():`.
- Compute probability outputs on `device`, move to CPU numpy arrays: `probs = model(test_x).cpu().numpy().ravel()`, `preds = (probs >= 0.5).astype(int)`.
- Calculate `accuracy`, `precision`, `recall`, `f1` using `sklearn.metrics`.
- Record `time_s` (wall-clock training time in seconds).

OUTPUT:
- At successful end, print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only):
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
"lstm": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94, "time_s": 4.1, "avg_cpu_percent": 15.2, "max_cpu_percent": 35.5, "avg_ram_mb": 220.4, "max_ram_mb": 280.1},
"bilstm": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 5.3, "avg_cpu_percent": 18.1, "max_cpu_percent": 39.2, "avg_ram_mb": 240.5, "max_ram_mb": 310.2},
"gru": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94, "time_s": 4.1, "avg_cpu_percent": 14.5, "max_cpu_percent": 32.1, "avg_ram_mb": 210.3, "max_ram_mb": 260.4},
"cnn": {"accuracy": 0.94, "precision": 0.93, "recall": 0.94, "f1": 0.93, "time_s": 2.5, "avg_cpu_percent": 10.2, "max_cpu_percent": 25.1, "avg_ram_mb": 150.1, "max_ram_mb": 190.2},
"hybrid": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94, "time_s": 3.2, "avg_cpu_percent": 16.4, "max_cpu_percent": 36.8, "avg_ram_mb": 230.2, "max_ram_mb": 290.5},
"text_cnn": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 3.0, "avg_cpu_percent": 12.0, "max_cpu_percent": 28.0, "avg_ram_mb": 180.0, "max_ram_mb": 220.0},
"stacked_lstm": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 8.0, "avg_cpu_percent": 22.0, "max_cpu_percent": 45.0, "avg_ram_mb": 260.0, "max_ram_mb": 320.0},
"cnn_lstm": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 6.0, "avg_cpu_percent": 18.0, "max_cpu_percent": 40.0, "avg_ram_mb": 240.0, "max_ram_mb": 300.0},
"auto_model": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.95, "time_s": 7.4, "avg_cpu_percent": 19.3, "max_cpu_percent": 41.5, "avg_ram_mb": 260.1, "max_ram_mb": 330.4}
}
}
<<<JSON_END>>>
- DO NOT output any explanation or markdown code fences other than ```python.
