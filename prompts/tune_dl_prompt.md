You are an expert deep learning engineer using Keras/TensorFlow.
The dataset has already been preprocessed, tokenized, and padded.
Features (X_train) are located at: '${features_npy_path}'
Labels (y_train) are located at: '${labels_npy_path}'

TASK:
- Write a Python script using **Keras/TensorFlow** and **Optuna** to optimize hyperparameters for LSTM, BiLSTM, GRU, 1D CNN, Hybrid CNN+Dense, and auto_model.
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
- CRITICAL FOR WINDOWS: Always wrap top-level execution with `if __name__ == '__main__':`. Do NOT use `multiprocessing`.
- Load features using `np.load(path)`. Flatten labels to 1D using `np.ravel()` and cast to integer.
- Vocabulary size is 15000, sequence max length is 400.

SPEED OPTIMIZATION FOR GPU TRAINING (CRITICAL):
- During Optuna trials, use a random stratified 20% sample of the training data.
- Run 10 trials per model type (total 60 trials for 6 models).
- Train each trial model for at most 2 epochs.
- Use batch_size=256 for faster GPU utilization.
- Do NOT use `recurrent_dropout`. It disables cuDNN GPU kernels. Use standard `dropout` only.
- Do NOT import or use `optuna.integration` modules (like `TFKerasPruningCallback`) to prevent dependency errors.

HYPERPARAMETER OPTIMIZATION (OPTUNA - MANDATORY):
- Optimize the following models: lstm, bilstm, gru, cnn, hybrid, auto_model.
- Search spaces per trial:
  - `embedding_dim` (int: 64 to 256, step=32)
  - `units` (int: 64 to 256, step=32)
  - `dropout_rate` (float: 0.1 to 0.5)
  - `learning_rate` (loguniform: 1e-4 to 1e-2)
- For each trial, build and compile the model with Keras, train for 2 epochs, return val_loss as the objective.
- Use `optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))`.
- Note: To count pruned trials use `len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])`. Do NOT use `study.pruned_trials`.
- After each model's study completes, extract best params: `study.best_params`.

TRIAL MODEL ARCHITECTURES (Keras):
- lstm: Embedding(vocab=15000, dim=embedding_dim, input_length=400) -> LSTM(units, dropout=dropout_rate) -> Dense(1, sigmoid)
- bilstm: Embedding(...) -> Bidirectional(LSTM(units, dropout=dropout_rate)) -> Dense(1, sigmoid)
- gru: Embedding(...) -> GRU(units, dropout=dropout_rate) -> Dense(1, sigmoid)
- cnn: Embedding(...) -> Conv1D(max(32, units//2), kernel_size=5, relu) -> GlobalMaxPooling1D -> Dropout(dropout_rate) -> Dense(1, sigmoid)
- hybrid: Embedding(...) -> Conv1D(max(32, units//2), kernel_size=5, relu) -> GlobalMaxPooling1D -> Dense(units, relu) -> Dropout(dropout_rate) -> Dense(1, sigmoid)
- auto_model: Embedding(...) -> Bidirectional(GRU(units, dropout=dropout_rate)) -> Dense(64, relu) -> Dropout(dropout_rate) -> Dense(1, sigmoid)

SAVING PARAMS:
- Save the dictionary of best parameters for all 6 models to: '${best_params_json_path}'
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
    "optimization_time_seconds": 320.5
  }
}
<<<JSON_END>>>
- Replace values with your actual calculated metrics.
- DO NOT output any explanation or markdown code fences other than ```python.
