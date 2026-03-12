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

# Model Candidates (mandatory)

Evaluate these with 5-fold CV:

- LogisticRegression
- MultinomialNB
- RandomForestClassifier
- LinearSVC

---

# Additional Model ("auto_model")

After evaluating mandatory models, you MUST add one more:

- Choose any sklearn classifier
- Choose hyperparameters yourself
- Train & evaluate with same CV procedure

---

# Feature Importance (mandatory)

If chosen model is:

- LogisticRegression / LinearSVC → top 15 positive TF-IDF weights
- RandomForestClassifier → top 15 feature*importances*
- MultinomialNB → top 15 log-probability ratios

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
"all_models": [
{
"model_name": "...",
"cv_accuracy_mean": ...,
"cv_accuracy_std": ...,
"test_accuracy": ...,
"test_recall_phishing": ...,
"test_f1_phishing": ...,
"confusion_matrix": [[tn, fp], [fn, tp]],
"top_features": [...]
}
],
"best_model": {
"model_name": "...",
"cv_accuracy_mean": ...,
"cv_accuracy_std": ...,
"test_accuracy": ...,
"test_recall_phishing": ...,
"test_f1_phishing": ...,
"confusion_matrix": [[tn, fp], [fn, tp]],
"top_features": [...]
}
}

- all_models MUST come FIRST.
- best_model MUST come LAST.
- best_model MUST be one of the entries in all_models.
- all_models MUST include FULL metrics for EVERY evaluated model.
