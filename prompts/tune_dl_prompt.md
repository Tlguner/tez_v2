You are an expert deep learning engineer.
The dataset has already been preprocessed, tokenized, and padded.
Features (X_train) are located at: '${features_npy_path}'
Labels (y_train) are located at: '${labels_npy_path}'

TASK:
- Write a Python script using TensorFlow/Keras to optimize hyperparameters for LSTM, BiLSTM, GRU, 1D CNN, Hybrid CNN+Dense, and auto_model using Optuna.
- Load features using `np.load(path)`.
  - Check `isinstance(data, np.ndarray)` or `hasattr(data, 'files')` to distinguish dense arrays from NPZ archive objects. If loading fails, try fallback to `scipy.sparse.load_npz(path).toarray()`.
  - Flatten labels to 1D using `np.ravel()` and cast to integer.
  - Vocabulary size is 5000, and sequence max length is 200.

SPEED OPTIMIZATION FOR CPU TRAINING (CRITICAL):
- Because training runs on CPU, during Optuna trials, use ONLY a random stratified 10% sample of the training data.
- Run ONLY 3 trials per model type.
- Train each trial model for at most 2 epochs.
- DO NOT use `recurrent_dropout` in LSTM, BiLSTM, or GRU layers. It disables optimized CPU kernels and makes CPU training extremely slow. Use standard `dropout` instead.

HYPERPARAMETER OPTIMIZATION (OPTUNA - MANDATORY):
- Optimize the following models: lstm, bilstm, gru, cnn, hybrid, auto_model.
- Search spaces:
  - Embedding Dimension: `embedding_dim` (int: 32 to 128)
  - Hidden Units: `units` (int: 32 to 128)
  - Dropout Rate: `dropout_rate` (float: 0.1 to 0.5)
  - Learning Rate: `learning_rate` (loguniform: 1e-4 to 1e-2)
- DO NOT import or use `optuna.integration` modules (like `TFKerasPruningCallback` or `KerasPruningCallback`) to prevent dependency errors.
- Note for counting trials: In Optuna, do NOT use `study.pruned_trials`. To count pruned trials, use `len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])`.
- Note for reproducibility: Optuna does NOT have a global `optuna.set_random_seed()` function. To set the seed for study replication, pass the seed to the sampler: `optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))`.

SAVING PARAMS:
- Save the dictionary of best parameters for all 6 models to a JSON file at: '${best_params_json_path}'
- Format of saved JSON should match:
{
  "lstm": {"embedding_dim": 64, "units": 64, "dropout_rate": 0.2, "learning_rate": 0.001},
  "bilstm": {"embedding_dim": 64, "units": 64, "dropout_rate": 0.2, "learning_rate": 0.001},
  "gru": {"embedding_dim": 64, "units": 64, "dropout_rate": 0.2, "learning_rate": 0.001},
  "cnn": {"embedding_dim": 64, "units": 64, "dropout_rate": 0.2, "learning_rate": 0.001},
  "hybrid": {"embedding_dim": 64, "units": 64, "dropout_rate": 0.2, "learning_rate": 0.001},
  "auto_model": {"embedding_dim": 64, "units": 64, "dropout_rate": 0.2, "learning_rate": 0.001}
}

OUTPUT:
- Print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only) at successful completion:
<<<JSON_START>>>
{
  "ok": true,
  "optuna_statistics": {
    "total_trials": 18,
    "pruned_trials": 0,
    "optimization_time_seconds": 120.5
  }
}
<<<JSON_END>>>
- Replace the values above with your actual calculated metrics and statistics.
- DO NOT output any explanation or markdown code fences other than ```python.
