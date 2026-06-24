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
  - Retrieve parameters for `LogisticRegression`, `RandomForestClassifier`, `LinearSVC`, and `auto_model`.
  - If the file is missing or values are empty, fallback to reasonable defaults.

- DONANIM KAYNAK İZLEME (RESOURCE MONITORING - MANDATORY):
  - Import the `ResourceMonitor` class from `resource_monitor` (`from resource_monitor import ResourceMonitor`).
  - For each model type (LogisticRegression, MultinomialNB, RandomForestClassifier, LinearSVC, VotingClassifier, auto_model):
    - Instantiate and start the monitor *before* training begins.
    - Stop the monitor *after* the final model finishes training.
    - Retrieve the monitored resource metrics (dictionary with keys: `avg_cpu_percent`, `max_cpu_percent`, `avg_ram_mb`, `max_ram_mb`).
    - Include these four metrics under the model's metrics dictionary in the final JSON output.

- MODELS TO TRAIN & SAVE:
  - Ensure the models directory '${models_dir}' exists (create if missing).
  - Measure training time (in seconds) for each final model fitting step.
  - Train the final baseline / optimized models on the full training set using the loaded hyperparameters:
    1. LogisticRegression (Optimized) -> Save to '${models_dir}/LogisticRegression.pkl'
    2. MultinomialNB (Baseline, no optimization parameters needed) -> Save to '${models_dir}/MultinomialNB.pkl'
    3. RandomForestClassifier (Optimized) -> Save to '${models_dir}/RandomForestClassifier.pkl'
    4. LinearSVC (Optimized) -> Save to '${models_dir}/LinearSVC.pkl'
    5. VotingClassifier (Hybrid Ensemble) -> Save to '${models_dir}/VotingClassifier.pkl'
       - Combine at least 3 of the best performing optimized models above using hard or soft voting.
    6. auto_model (Optimized GradientBoostingClassifier) -> Save to '${models_dir}/auto_model.pkl'
  - CRITICAL WARNING: In Scikit-learn, GradientBoostingClassifier, LinearSVC, and MultinomialNB do NOT support the 'n_jobs' parameter. Passing 'n_jobs' to them will cause a TypeError. Ensure you NEVER pass 'n_jobs' when instantiating these classifiers. Only pass 'n_jobs' to estimators that natively support it (like RandomForestClassifier, LogisticRegression, or VotingClassifier).

- EVALUATION:
  - For each of the 6 final models, obtain 1D class predictions using `.predict(X_test)`.
  - Calculate `accuracy` (using `accuracy_score`), `precision` (using `precision_score`), `recall` (using `recall_score`), and `f1` (using `f1_score`). Use `average='binary'` for precision, recall, and f1.

- OUTPUT:
  - At successful end of execution, print EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>> (stdout only). Include a "model_descriptions" dictionary detailing the specific configuration and parameters used for each trained model.
  - CRITICAL (Dynamic Descriptions): The "description" for each model (especially VotingClassifier and auto_model) must NOT be a static string. It must dynamically print the actual structural layout and the best hyperparameter values found (e.g. for VotingClassifier: "Ensemble of LogisticRegression(C=0.5), RandomForest(n_estimators=100), and LinearSVC(C=0.1)", and for auto_model: "GradientBoostingClassifier with learning_rate=0.05, n_estimators=80, max_depth=5").
<<<JSON_START>>>
{
  "ok": true,
  "models_trained": ["LogisticRegression", "MultinomialNB", "RandomForestClassifier", "LinearSVC", "VotingClassifier", "auto_model"],
  "best_model": "VotingClassifier",
  "model_descriptions": {
    "LogisticRegression": "L2 Regularized Logistic Regression optimized via Optuna",
    "MultinomialNB": "Baseline Naive Bayes for text classification",
    "RandomForestClassifier": "Optimized Random Forest with tuned n_estimators and max_depth",
    "LinearSVC": "Linear Support Vector Classifier",
    "VotingClassifier": "Ensemble of the best performing optimized classifiers (soft/hard voting)",
    "auto_model": "Gradient Boosting Classifier optimized via Optuna"
  },
  "metrics": {
    "LogisticRegression": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.945, "time_s": 4.5, "avg_cpu_percent": 8.2, "max_cpu_percent": 15.5, "avg_ram_mb": 45.3, "max_ram_mb": 50.1},
    "MultinomialNB": {"accuracy": 0.93, "precision": 0.92, "recall": 0.93, "f1": 0.925, "time_s": 0.2, "avg_cpu_percent": 2.1, "max_cpu_percent": 5.0, "avg_ram_mb": 30.1, "max_ram_mb": 32.4},
    "RandomForestClassifier": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.955, "time_s": 15.3, "avg_cpu_percent": 24.1, "max_cpu_percent": 85.3, "avg_ram_mb": 75.2, "max_ram_mb": 110.4},
    "LinearSVC": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.945, "time_s": 2.1, "avg_cpu_percent": 9.5, "max_cpu_percent": 20.3, "avg_ram_mb": 48.2, "max_ram_mb": 55.4},
    "VotingClassifier": {"accuracy": 0.97, "precision": 0.96, "recall": 0.97, "f1": 0.965, "time_s": 22.4, "avg_cpu_percent": 15.6, "max_cpu_percent": 45.2, "avg_ram_mb": 85.3, "max_ram_mb": 120.1},
    "auto_model": {"accuracy": 0.96, "precision": 0.95, "recall": 0.96, "f1": 0.955, "time_s": 45.1, "avg_cpu_percent": 14.1, "max_cpu_percent": 28.5, "avg_ram_mb": 62.4, "max_ram_mb": 70.2}
  }
}
<<<JSON_END>>>
  - Note: Replace the values above with your actual calculated metrics and statistics.
- DO NOT output any explanation or markdown code fences other than ```python.
