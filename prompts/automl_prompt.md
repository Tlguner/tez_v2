You are an expert machine learning engineer.

CRITICAL RULES:

1. The script MUST load the dataset from EXACTLY this path: '${absoluteDatasetPath}'
2. DO NOT use 'dataset.csv' or any other default name.
3. IF the file is missing → raise FileNotFoundError.
4. DO NOT generate fake or artificial data.

---

# Dataset Requirements

The script MUST load dataset from EXACTLY this path: '${absoluteDatasetPath}'

Do NOT assume fixed column names like 'Email Text' / 'Email Type'.
Instead, automatically detect columns.

Automatically detect text column from:
["text_combined", "Email Text", "text", "body", "message", "content", "email", "mail"]

Automatically detect label column from:
["label", "Email Type", "labels", "target", "Category", "class", "y"]

If no text column found → raise ValueError("Could not find a text column")
If no label column found → raise ValueError("Could not find a label column")

# Label Mapping (CRITICAL)

The script MUST choose exactly ONE of the following branches.

## Branch 1 — Numeric labels (preferred)

If the detected label column is numeric OR can be safely coerced to numeric:

- Convert labels using:
  y_num = pd.to_numeric(df[label_col], errors="coerce")
- Drop rows where y_num is NaN
- Keep ONLY rows where y_num ∈ {0, 1}
- Use:
  y = y_num.astype(int) (as a 1D pandas Series)
- Do NOT apply any string-based label mapping
- If remaining rows < 10 → raise:
  ValueError("Dataset invalid or preprocessing removed all rows.")

## Branch 2 — String / categorical labels

If the label column is NOT numeric:

- Normalize labels: lowercase + strip whitespace
- Map labels as follows:

  Phishing → 1:
  ["phishing", "phishing email", "spam", "malicious"]

  Safe → 0:
  ["ham", "safe", "safe email", "legitimate", "normal", "benign"]

- Drop rows with unmappable labels
- Use y as a 1D pandas Series
- If remaining rows < 10 → raise:
  ValueError("Dataset invalid or preprocessing removed all rows.")

## Final Sanity Check (MANDATORY)

After label mapping:

- The dataset MUST contain BOTH classes {0, 1}
- Otherwise raise:
  ValueError("Dataset mapping failed: need both phishing and safe labels")

# Preprocessing Rules

- Fill NaN text with ""
- Drop NaN labels
- Apply label normalization
- Stratified 80/20 split

---

# Vectorization

Use TF-IDF:

max_features = 5000

---

# Hyperparameter Optimization & Pruning (OPTUNA - MANDATORY)

You MUST use `optuna` for finding the best hyperparameters for the classifiers.

Requirements:
1. **Dynamic Search Spaces**:
   - For `LogisticRegression`: Optimize regularization parameter `C` (loguniform: 1e-4 to 10) and `solver` (e.g., `liblinear`, `lbfgs`).
   - For `RandomForestClassifier`: Optimize `n_estimators` (int: 50 to 200), `max_depth` (int: 5 to 20 or None), and `min_samples_split` (int: 2 to 10).
   - For `LinearSVC`: Optimize regularization `C` (loguniform: 1e-4 to 10).
   - For `SGDClassifier`: Optimize `alpha` (loguniform: 1e-6 to 1e-1) and `loss` (choices: `hinge`, `modified_huber`).
   - `GaussianNB` and `MultinomialNB` have no Optuna optimization (baseline classifiers, use defaults).

2. **Bayesian Optimization via 5-Fold Cross Validation**:
   - During Optuna optimization trials, the objective function to maximize MUST be the **mean 5-fold Cross Validation Accuracy** on the training split.
   - Use `optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())` for each classifier class.
   - Run at least 10 trials per classifier class to find the best hyperparameters.
   - Log trial progress (trial number, CV score, parameters) to stderr.
   - Extract the best parameters and train the final baseline models on the full training set.

---

# Model Candidates (mandatory)

Train and evaluate the optimized versions of these classifiers using 5-fold CV:

- LogisticRegression (Optimized via Optuna)
- MultinomialNB (Baseline — no Optuna, use default alpha)
- GaussianNB (Baseline — no Optuna. IMPORTANT: sparse TF-IDF must be converted to dense using `.toarray()` ONLY for this model's fit and predict steps)
- RandomForestClassifier (Optimized via Optuna)
- LinearSVC (Optimized via Optuna)
- SGDClassifier (Optimized via Optuna, use `hinge` or `modified_huber` loss)

---

# ML Hybrid Model (Mandatory Ensemble)

After individual classifiers, create and train a **Hybrid ML** model named:
"hybrid"

Requirements:
- It must be a `VotingClassifier` (with soft or hard voting) or a `StackingClassifier` from `sklearn.ensemble`.
- It must combine at least 3 of the best performing optimized classifiers from the candidate list (using their Optuna-optimized hyperparameter sets).
- It must be trained and evaluated using the same 5-fold CV and test protocols.

---

# Additional ML Model ("auto_model")

After evaluating the hybrid model, you MUST add exactly one more:
"auto_model"

Requirements:
- Choose any custom/advanced classifier (e.g., optimized HistGradientBoostingClassifier).
- Use Optuna to optimize its hyperparameters (at least 10 trials) based on 5-fold CV.
- Train & evaluate with the same 5-fold CV and test procedure.
- Since HistGradientBoostingClassifier does not support sparse matrices natively, you MUST convert the sparse matrix to dense using `.toarray()` ONLY for this model's fit and predict steps.

---

# Feature Importance (mandatory)

If chosen best model is:

- LogisticRegression / LinearSVC / SGDClassifier → top 15 positive TF-IDF weights (from `.coef_`)
- RandomForestClassifier / HistGradientBoostingClassifier / ExtraTreesClassifier → top 15 feature_importances_
- MultinomialNB / GaussianNB → top 15 log-probability ratios
- VotingClassifier / StackingClassifier (Hybrid) → top 15 features of its best constituent estimator or computed overall importances if available.

---

# Logging BEFORE JSON output

Print these logs BEFORE final JSON:

- [INFO] Loaded dataset: shape (...)
- [INFO] Selected text column: ...
- [INFO] Selected label column: ...
- [INFO] Preprocessing done: remaining rows, class balance
- [INFO] Starting TF-IDF...
- [INFO] Trying model X...
- [INFO] CV mean/std...
- [INFO] Best model selected: ...

---

# Final Required Output (ONLY JSON)

Output only:

{
"model_name": "...",
"cv_accuracy_mean": ...,
"cv_accuracy_std": ...,
"accuracy": ...,
"recall_phishing": ...,
"f1_phishing": ...,
"confusion_matrix": [[tn, fp], [fn, tp]],
"top_features": [...],
"all_models": [
{
"model_name": "...",
"cv_accuracy_mean": ...,
"cv_accuracy_std": ...,
"test_accuracy": ...,
"test_recall_phishing": ...,
"test_f1_phishing": ...,
"time_s": ...,
"confusion_matrix": [[tn, fp], [fn, tp]]
}
]
}

Do not print anything after JSON.

IMPORTANT: Logs must NOT contain "{" or "}" characters. Do not print dicts or JSON in logs.

IMPORTANT:
The final JSON MUST include:

- One object describing the BEST model
- An "all_models" array containing metrics for EVERY evaluated model
  The final JSON must NOT omit evaluated models.

CRITICAL:
The label vector (y) MUST be a 1-dimensional pandas Series.
Never use df[[label_col]].
Always use df[label_col].

CRITICAL OUTPUT RULES:

- The final result MUST be printed between the following markers:
  <<<JSON_START>>>
  <<<JSON_END>>>

- Example (MANDATORY):

print("<<<JSON_START>>>")
print(json.dumps(output, indent=2))
print("<<<JSON_END>>>")

- Do NOT print any JSON outside these markers.
- Logs may be printed before <<<JSON_START>>>.
- Do NOT print anything after <<<JSON_END>>>.

CRITICAL JSON STRUCTURE RULES:

- The final JSON MUST be structured EXACTLY as follows:

{
"optuna_statistics": {
  "total_trials": ...,
  "pruned_trials": ...,
  "optimization_time_seconds": ...,
  "pruning_ratio": ...  # (pruned_trials / total_trials)
},
"all_models": [
{
"model_name": "...",
"cv_accuracy_mean": ...,
"cv_accuracy_std": ...,
"test_accuracy": ...,
"test_recall_phishing": ...,
"test_f1_phishing": ...,
"time_s": ...,
"confusion_matrix": [[tn, fp], [fn, tp]],
"fpr": [list_of_false_positive_rates],
"tpr": [list_of_true_positive_rates],
"top_features": [...],
"description": "Dynamic string describing model config and WHY it was chosen (1-2 sentences for auto_model; for Hybrid/Ensemble list all component models with their parameters e.g. 'Ensemble of LogisticRegression(C=7.0) + RandomForest(n_estimators=148) + LinearSVC(C=0.86) using soft voting')",
"optuna_gain": {
  "baseline_accuracy": ...,
  "optimized_accuracy": ...,
  "gain": ...  # (optimized_accuracy - baseline_accuracy)
}
}
],
"best_model": {
"model_name": "...",
"cv_accuracy_mean": ...,
"cv_accuracy_std": ...,
"test_accuracy": ...,
"test_recall_phishing": ...,
"test_f1_phishing": ...,
"time_s": ...,
"confusion_matrix": [[tn, fp], [fn, tp]],
"fpr": [list_of_false_positive_rates],
"tpr": [list_of_true_positive_rates],
"top_features": [...],
"optuna_gain": {
  "baseline_accuracy": ...,
  "optimized_accuracy": ...,
  "gain": ...
}
}
}

- optuna_statistics MUST come FIRST.
- all_models MUST come SECOND.
- best_model MUST come LAST.
- best_model MUST be one of the entries in all_models.
- all_models MUST include FULL metrics for EVERY evaluated model.
