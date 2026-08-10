You are a senior machine learning engineer.
The dataset has already been preprocessed and vectorized.
Features (X_train) are located at: '${features_npy_path}'
Labels (y_train) are located at: '${labels_npy_path}'

TASK:
- Write a Python script to optimize hyperparameters for multiple classifiers using Optuna.
- Load X_train using scipy.sparse.load_npz. Keep it as a sparse matrix throughout the entire process. DO NOT convert to dense arrays to prevent OOM/high memory usage.
- Load y_train using np.load.
- Flatten label array using `y_train = np.ravel(y_train).astype(int)`.

WARNINGS & LOGGING:
- Import `warnings` and run `warnings.filterwarnings("ignore")` at the very beginning of the script to suppress scikit-learn warnings (like solver convergence or n_jobs warnings) and keep logs clean.
- Use `optuna.logging.set_verbosity(optuna.logging.WARNING)` to suppress verbose trial logs.

SPEED OPTIMIZATION FOR CPU TUNING (CRITICAL):
- Because training runs on CPU and the dataset is large (65k+ rows with 5000+ features), running 25 trials with 5-fold cross-validation on the full training set will take a very long time.
- To make tuning fast, perform the Optuna search using a stratified 20% random sample of X_train and y_train (use `train_test_split(..., train_size=0.2, stratify=y_train, random_state=42)`).
- This 20% sample is ONLY used inside the Optuna objective function.

HYPERPARAMETER OPTIMIZATION (OPTUNA - MANDATORY):
- Optimize parameters for ALL of the following classifiers using the 20% stratified sample:
  1. **LogisticRegression**: Optimize `C` (suggest_float with log=True: 1e-4 to 10) and `solver` (choice: `liblinear`, `lbfgs`).
  2. **MultinomialNB**: Optimize `alpha` (suggest_float with log=True: 1e-3 to 10).
  3. **GaussianNB**: Optimize `var_smoothing` (suggest_float with log=True: 1e-12 to 1e-6). CRITICAL: GaussianNB does NOT support sparse matrices. Convert the sparse fold matrices to dense using `.toarray()` ONLY for GaussianNB's fit and predict steps inside its objective function.
  4. **RandomForestClassifier**: Optimize `n_estimators` (int: 50 to 200), `max_depth` (int: 5 to 30 or None), and `min_samples_split` (int: 2 to 10).
  5. **LinearSVC**: Optimize `C` (suggest_float with log=True: 1e-4 to 10).
  6. **SGDClassifier**: Optimize `alpha` (suggest_float with log=True: 1e-6 to 1e-1) and `loss` (choice: `hinge`, `modified_huber`).
  7. **auto_model** (HistGradientBoostingClassifier): Optimize `learning_rate` (log=True: 1e-3 to 1e-1), `max_iter` (int: 50 to 150), and `max_depth` (int: 3 to 8).
  
- For each model, run exactly 25 trials using 5-fold cross-validation (`cross_val_score` with `scoring='accuracy'`) inside the Optuna objective function to maximize mean cross-validation accuracy.
- Implement pruning using `optuna.pruners.HyperbandPruner()` to prune unpromising trials early.
- Initialize the study sampler with a seed: `sampler = optuna.samplers.TPESampler(seed=42)`.
- Note: In Optuna, do NOT use `study.pruned_trials`. To count pruned trials, use `len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])`.
- Note: HistGradientBoostingClassifier and LinearSVC do NOT support the `n_jobs` parameter. Never pass `n_jobs` to them. Pass `n_jobs` only to estimators that natively support it (like RandomForestClassifier, LogisticRegression, SGDClassifier) to speed up training.
- Note: Since HistGradientBoostingClassifier does not support sparse matrices natively, you MUST convert the sparse fold matrices to dense using `.toarray()` ONLY for the HistGradientBoostingClassifier's fit and predict steps inside its objective function (e.g., `X_train_fold.toarray()`). This is safe because the 20% sample size is small enough to avoid memory issues.

SAVING PARAMS:
- Save the dictionary of best parameters for all 7 models to a JSON file at: '${best_params_json_path}'
- Format of saved JSON should match:
{
  "LogisticRegression": {"C": 0.5, "solver": "liblinear"},
  "MultinomialNB": {"alpha": 0.1},
  "GaussianNB": {"var_smoothing": 1e-9},
  "RandomForestClassifier": {"n_estimators": 150, "max_depth": 20, "min_samples_split": 2},
  "LinearSVC": {"C": 1.0},
  "SGDClassifier": {"alpha": 0.0001, "loss": "modified_huber"},
  "auto_model": {"learning_rate": 0.05, "max_iter": 100, "max_depth": 5}
}

OUTPUT:
- Print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only) at successful completion:
<<<JSON_START>>>
{
  "ok": true,
  "optuna_statistics": {
    "total_trials": 175,
    "pruned_trials": 12,
    "optimization_time_seconds": 120.5
  }
}
<<<JSON_END>>>
- Replace the values above with your actual calculated metrics and statistics.
- DO NOT output any explanation or markdown code fences other than ```python.
