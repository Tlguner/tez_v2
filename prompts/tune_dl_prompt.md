You are an expert deep learning engineer using PyTorch and CUDA GPU acceleration.
The dataset has already been preprocessed, tokenized, and padded.
Features (X_train) are located at: '${features_npy_path}'
Labels (y_train) are located at: '${labels_npy_path}'

TASK:
- Write a Python script using **PyTorch**, **Optuna**, and **NATIVE CUDA GPU ACCELERATION** to optimize hyperparameters for LSTM, BiLSTM, GRU, 1D CNN, Hybrid CNN+Dense, and auto_model.

MANDATORY GPU SETUP FOR RTX 5070:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

CRITICAL FOR WINDOWS MULTIPROCESSING:
- Always wrap script execution inside `if __name__ == '__main__':`.
- In PyTorch `DataLoader(..., num_workers=0, pin_memory=True)` set `num_workers=0` to prevent Windows deadlocks.

SPEED OPTIMIZATION FOR GPU TRAINING:
- Sample a random stratified 20% subset of X_train for Optuna tuning to make trial loops lightning fast.
- Run 10 trials per model type (60 total trials across 6 models).
- Train each trial model for at most 2 epochs on GPU.
- Use batch_size=256 for maximum CUDA GPU tensor throughput.

SEARCH SPACE PER TRIAL:
- `embedding_dim` (int: 64 to 256, step=32)
- `units` (int: 64 to 256, step=32)
- `dropout_rate` (float: 0.1 to 0.5)
- `learning_rate` (loguniform: 1e-4 to 1e-2)

OPTUNA SETUP:
- For each model type (lstm, bilstm, gru, cnn, hybrid, auto_model):
  - Define an objective function `objective(trial)` that instantiates the PyTorch model, moves it to `device`, trains for 2 epochs, evaluates validation loss on a 10% validation split, and returns the validation loss.
  - Create study: `study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))`
  - Optimize: `study.optimize(objective, n_trials=10)`
  - Save best params to a dictionary.

SAVING PARAMS:
- Save dictionary of best parameters to JSON file at: '${best_params_json_path}'
- Format:
{
  "lstm": {"embedding_dim": 128, "units": 128, "dropout_rate": 0.2, "learning_rate": 0.001},
  "bilstm": {"embedding_dim": 128, "units": 128, "dropout_rate": 0.2, "learning_rate": 0.001},
  "gru": {"embedding_dim": 128, "units": 128, "dropout_rate": 0.2, "learning_rate": 0.001},
  "cnn": {"embedding_dim": 128, "units": 128, "dropout_rate": 0.2, "learning_rate": 0.001},
  "hybrid": {"embedding_dim": 128, "units": 128, "dropout_rate": 0.2, "learning_rate": 0.001},
  "auto_model": {"embedding_dim": 128, "units": 128, "dropout_rate": 0.2, "learning_rate": 0.001}
}

OUTPUT:
- Print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only) at successful completion:
<<<JSON_START>>>
{
  "ok": true,
  "optuna_statistics": {
    "total_trials": 60,
    "pruned_trials": 0,
    "optimization_time_seconds": 35.0
  }
}
<<<JSON_END>>>
- Replace values with actual calculated metrics.
- DO NOT output any explanation or markdown code fences other than ```python.
