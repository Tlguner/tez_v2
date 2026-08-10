You are a senior machine learning engineer.
The dataset has already been preprocessed and vectorized.
Features (X_train, X_test) are located at:
- Train: '${features_npy_path}'
- Test: '${features_npy_path.replace("X_train", "X_test")}'
Labels (y_train, y_test) are located at:
- Train: '${labels_npy_path}'
- Test: '${labels_npy_path.replace("y_train", "y_test")}'
Vectorizer (TF-IDF) is saved at: '${vectorizer_pkl_path}'
Best hyperparameters are saved as a JSON file at: '${best_params_json_path}'

TASK:
- Write a Python script to train multiple optimized classifiers and an Ensemble (Hybrid) model using the pre-tuned hyperparameters.
- Load X_train, X_test, y_train, y_test using numpy and scipy.
  - Since features are saved as `.npz`, load them using `scipy.sparse.load_npz`. Keep them as sparse matrices throughout the entire training process. DO NOT convert sparse matrices to dense numpy arrays using `.toarray()` to prevent high memory and CPU usage.
  - Since labels are saved as `.npy`, load them using `np.load`.
- CRITICAL PREPROCESSING:
  - Flatten the label arrays to ensure they are 1D arrays. Use `y_train = np.ravel(y_train)` and `y_test = np.ravel(y_test)`.
  - Ensure labels are integers. Use `y_train = y_train.astype(int)` and `y_test = y_test.astype(int)`.

- WARNINGS & LOGGING:
  - Import `warnings` and run `warnings.filterwarnings("ignore")` at the very beginning of the script to suppress scikit-learn warnings (like solver convergence or n_jobs warnings) and keep logs clean.

- LOADING HYPERPARAMETERS:
  - Load the pre-tuned hyperparameters from '${best_params_json_path}' using python `json` module.
  - Retrieve parameters for `LogisticRegression`, `MultinomialNB`, `GaussianNB`, `RandomForestClassifier`, `LinearSVC`, `SGDClassifier`, and `auto_model`.
  - If the file is missing or values are empty, fallback to reasonable defaults.

- DONANIM KAYNAK İZLEME (RESOURCE MONITORING - MANDATORY):
  - Import the `ResourceMonitor` class from `resource_monitor` (`from resource_monitor import ResourceMonitor`).
  - For each model type:
    - Instantiate and start the monitor *before* training begins (`monitor = ResourceMonitor()`, `monitor.start()`).
    - Stop the monitor *after* the final model finishes training.
    - CRITICAL: The `stop()` method returns the monitored resource metrics dictionary directly. Do NOT call `get_metrics()`. Instead, use: `metrics = monitor.stop()`.
    - Include these four metrics under the model's metrics dictionary in the final JSON output.

- MODELS TO TRAIN & SAVE:
  - Ensure the models directory '${models_dir}' exists (create if missing).
  - Measure training time (in seconds) for each final model fitting step.
  - Train the final baseline / optimized models on the full training set using the loaded hyperparameters:
    1. LogisticRegression (Optimized via Optuna: C, solver) -> Save to '${models_dir}/LogisticRegression.pkl'
    2. MultinomialNB (Optimized via Optuna: alpha) -> Save to '${models_dir}/MultinomialNB.pkl'
    3. GaussianNB (Optimized via Optuna: var_smoothing) -> Save to '${models_dir}/GaussianNB.pkl'
       - CRITICAL: GaussianNB does NOT support sparse matrices. Convert sparse TF-IDF to dense using `.toarray()` ONLY for GaussianNB's fit and predict steps.
    4. RandomForestClassifier (Optimized) -> Save to '${models_dir}/RandomForestClassifier.pkl'
    5. LinearSVC (Optimized) -> Save to '${models_dir}/LinearSVC.pkl'
    6. SGDClassifier (Optimized) -> Save to '${models_dir}/SGDClassifier.pkl'
    7. VotingClassifier (Hybrid Ensemble) -> Save to '${models_dir}/VotingClassifier.pkl'
       - Combine at least 3 of the best performing optimized models above using hard or soft voting.
       - IMPORTANT: SGDClassifier with `hinge` loss does NOT support predict_proba; use `modified_huber` loss or exclude it from soft voting.
    8. auto_model (Optimized HistGradientBoostingClassifier) -> Save to '${models_dir}/auto_model.pkl'
  - CRITICAL WARNING: In Scikit-learn, HistGradientBoostingClassifier, LinearSVC, MultinomialNB, and GaussianNB do NOT support the 'n_jobs' parameter. Passing 'n_jobs' to them will cause a TypeError. Ensure you NEVER pass 'n_jobs' when instantiating these classifiers. Only pass 'n_jobs' to estimators that natively support it (like RandomForestClassifier, LogisticRegression, SGDClassifier, or VotingClassifier).
  - NOTE: Since HistGradientBoostingClassifier does not support sparse matrices natively, you MUST convert the sparse training/testing matrices to dense using `.toarray()` ONLY for the HistGradientBoostingClassifier's fit and predict/predict_proba steps.

- EVALUATION:
  - For each of the 8 final models, obtain 1D class predictions using `.predict(X_test)`.
  - Calculate `accuracy` (using `accuracy_score`), `precision` (using `precision_score`), `recall` (using `recall_score`), and `f1` (using `f1_score`). Use `average='binary'` for precision, recall, and f1.

- OUTPUT:
  - At successful end of execution, print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only). Include a "model_descriptions" dictionary detailing the specific configuration and parameters used for each trained model.
  - CRITICAL (Dynamic Descriptions): The "description" for each model must be a dynamic string that accurately reflects the actual training configuration:
    - For baseline models (LogisticRegression, MultinomialNB, GaussianNB, SGDClassifier, RandomForestClassifier, LinearSVC): state the key hyperparameters used.
    - For VotingClassifier (Hybrid Ensemble): MUST explicitly list ALL component models with their optimized parameters in the format: "Ensemble of ModelA(paramX=valX, paramY=valY) + ModelB(paramX=valX) + ModelC(paramX=valX) using [soft/hard] voting."
    - For auto_model: MUST include (1) the model architecture and optimized parameters AND (2) a 1-2 sentence explanation of WHY this model was chosen for this task (e.g., what makes HistGradientBoosting suitable for high-dimensional TF-IDF features compared to the base models).
<<<JSON_START>>>
{
  "ok": true,
  "models_trained": ["LogisticRegression", "MultinomialNB", "GaussianNB", "RandomForestClassifier", "LinearSVC", "SGDClassifier", "VotingClassifier", "auto_model"],
  "best_model": "VotingClassifier",
  "model_descriptions": {
    "LogisticRegression": "L2 Regularized Logistic Regression with C=7.0, solver=liblinear, optimized via Optuna",
    "MultinomialNB": "Baseline Naive Bayes for text classification with default alpha",
    "GaussianNB": "Gaussian Naive Bayes baseline; assumes Gaussian feature distribution",
    "RandomForestClassifier": "Optimized Random Forest with n_estimators=148, max_depth=None, min_samples_split=6",
    "LinearSVC": "Linear Support Vector Classifier with C=0.86, optimized via Optuna",
    "SGDClassifier": "SGD classifier with modified_huber loss and alpha=0.0001, optimized via Optuna for large-scale linear classification",
    "VotingClassifier": "Ensemble of LogisticRegression(C=7.0, solver=liblinear) + RandomForestClassifier(n_estimators=148, max_depth=None) + LinearSVC(C=0.86) using soft voting",
    "auto_model": "HistGradientBoostingClassifier with learning_rate=0.05, max_iter=80, max_depth=5. Selected because gradient boosting excels at capturing non-linear interactions in high-cardinality TF-IDF features that linear models miss, while remaining faster than RandomForest at inference time."
  },
  "metrics": {
    "LogisticRegression": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.945, "time_s": 4.5, "avg_cpu_percent": 8.2, "max_cpu_percent": 15.5, "avg_ram_mb": 45.3, "max_ram_mb": 50.1},
    "MultinomialNB": {"accuracy": 0.93, "precision": 0.92, "recall": 0.93, "f1": 0.925, "time_s": 0.2, "avg_cpu_percent": 2.1, "max_cpu_percent": 5.0, "avg_ram_mb": 30.1, "max_ram_mb": 32.4},
    "GaussianNB": {"accuracy": 0.91, "precision": 0.90, "recall": 0.91, "f1": 0.905, "time_s": 0.5, "avg_cpu_percent": 3.0, "max_cpu_percent": 6.0, "avg_ram_mb": 200.0, "max_ram_mb": 210.0},
    "RandomForestClassifier": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.955, "time_s": 15.3, "avg_cpu_percent": 24.1, "max_cpu_percent": 85.3, "avg_ram_mb": 75.2, "max_ram_mb": 110.4},
    "LinearSVC": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.945, "time_s": 2.1, "avg_cpu_percent": 9.5, "max_cpu_percent": 20.3, "avg_ram_mb": 48.2, "max_ram_mb": 55.4},
    "SGDClassifier": {"accuracy": 0.94, "precision": 0.93, "recall": 0.94, "f1": 0.935, "time_s": 1.0, "avg_cpu_percent": 7.0, "max_cpu_percent": 14.0, "avg_ram_mb": 40.0, "max_ram_mb": 45.0},
    "VotingClassifier": {"accuracy": 0.97, "precision": 0.96, "recall": 0.97, "f1": 0.965, "time_s": 22.4, "avg_cpu_percent": 15.6, "max_cpu_percent": 45.2, "avg_ram_mb": 85.3, "max_ram_mb": 120.1},
    "auto_model": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.955, "time_s": 45.1, "avg_cpu_percent": 14.1, "max_cpu_percent": 28.5, "avg_ram_mb": 62.4, "max_ram_mb": 70.2}
  }
}
<<<JSON_END>>>
  - Note: Replace the values above with your actual calculated metrics and statistics.
- DO NOT output any explanation or markdown code fences other than ```python.
