You are a senior Deep Learning engineer specializing in PyTorch, CUDA GPU acceleration, and Bayesian Hyperparameter Optimization with Optuna.
The dataset has already been preprocessed, tokenized into integer sequence vectors, and padded.
Features (X_train) are located at: '${features_npy_path}'
Labels (y_train) are located at: '${labels_npy_path}'

TASK:
- Write a complete, standalone Python script to optimize hyperparameters for all candidate Deep Learning architectures (TextCNN, BiLSTM, GRU, CNN, Hybrid, Stacked LSTM, CNN-LSTM, auto_model) using Optuna on CUDA GPU.
- Load X_train using np.load and y_train using np.load.
- Flatten label array using `y_train = np.ravel(y_train).astype(int)`.

MANDATORY GPU ACCELERATION FOR RTX 5070:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

CRITICAL FOR WINDOWS MULTIPROCESSING:
- Always wrap script execution inside `if __name__ == '__main__':`.
- In PyTorch `DataLoader(..., num_workers=0, pin_memory=True)` ALWAYS set `num_workers=0` to prevent Windows deadlocks.

SPEED OPTIMIZATION FOR GPU TUNING:
- To achieve high search throughput on the RTX 5070 GPU, perform the Optuna objective evaluation using a stratified 25% subsample of (X_train, y_train) with an internal 80/20 train/val split.
- Run 15 trials per model family with 3 fast epochs per trial.
- Use `optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)` to terminate unpromising parameter trials early based on validation loss.
- Initialize the study sampler with a fixed seed: `sampler = optuna.samplers.TPESampler(seed=42)`.

HYPERPARAMETER SEARCH SPACES:
1. **TextCNN**:
   - `lr`: suggest_float('lr', 1e-4, 1e-2, log=True)
   - `filters`: suggest_categorical('filters', [32, 64, 128])
   - `dropout_rate`: suggest_float('dropout_rate', 0.1, 0.5)
2. **BiLSTM**:
   - `lr`: suggest_float('lr', 1e-4, 1e-2, log=True)
   - `units`: suggest_categorical('units', [64, 128])
   - `dropout_rate`: suggest_float('dropout_rate', 0.1, 0.5)
3. **GRU**:
   - `lr`: suggest_float('lr', 1e-4, 1e-2, log=True)
   - `units`: suggest_categorical('units', [64, 128])
   - `dropout_rate`: suggest_float('dropout_rate', 0.1, 0.5)
4. **CNN**:
   - `lr`: suggest_float('lr', 1e-4, 1e-2, log=True)
   - `filters`: suggest_categorical('filters', [32, 64, 128])
   - `kernel_size`: suggest_categorical('kernel_size', [3, 5])
5. **Hybrid (CNN+Dense)**:
   - `lr`: suggest_float('lr', 1e-4, 1e-2, log=True)
   - `filters`: suggest_categorical('filters', [32, 64])
   - `units`: suggest_categorical('units', [32, 64])
   - `dropout_rate`: suggest_float('dropout_rate', 0.1, 0.4)
6. **Stacked LSTM**:
   - `lr`: suggest_float('lr', 1e-4, 5e-3, log=True)
   - `units`: suggest_categorical('units', [32, 64])
   - `dropout_rate`: suggest_float('dropout_rate', 0.1, 0.3)
7. **CNN-LSTM**:
   - `lr`: suggest_float('lr', 1e-4, 5e-3, log=True)
   - `filters`: suggest_categorical('filters', [32, 64])
   - `units`: suggest_categorical('units', [32, 64])
8. **auto_model (BiGRU + Multi-Head Self-Attention)**:
   - `lr`: suggest_float('lr', 1e-4, 5e-3, log=True)
   - `units`: suggest_categorical('units', [64, 128])
   - `num_heads`: suggest_categorical('num_heads', [2, 4])
   - `dropout_rate`: suggest_float('dropout_rate', 0.1, 0.4)

SAVING PARAMS:
- Save the dictionary of optimal hyperparameters for all DL models to a JSON file at: '${best_params_json_path}'
- Format of saved JSON:
{
  "text_cnn": {"lr": 0.001, "filters": 64, "dropout_rate": 0.2},
  "bilstm": {"lr": 0.001, "units": 128, "dropout_rate": 0.2},
  "gru": {"lr": 0.001, "units": 128, "dropout_rate": 0.2},
  "cnn": {"lr": 0.001, "filters": 64, "kernel_size": 5},
  "hybrid": {"lr": 0.001, "filters": 64, "units": 64, "dropout_rate": 0.2},
  "stacked_lstm": {"lr": 0.001, "units": 64, "dropout_rate": 0.2},
  "cnn_lstm": {"lr": 0.001, "filters": 64, "units": 64},
  "auto_model": {"lr": 0.001, "units": 128, "num_heads": 4, "dropout_rate": 0.2}
}

OUTPUT:
- Print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only) at successful completion:
<<<JSON_START>>>
{
  "ok": true,
  "optuna_dl_statistics": {
    "total_trials": 120,
    "pruned_trials": 28,
    "optimization_time_seconds": 185.4
  }
}
<<<JSON_END>>>
- Replace the values above with your actual calculated metrics and statistics.
- DO NOT output any explanation or markdown code fences other than ```python.
