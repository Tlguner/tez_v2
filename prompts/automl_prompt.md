You are an expert machine learning engineer.

CRITICAL RULES:

1. The script MUST load the dataset from EXACTLY this path: '${absoluteDatasetPath}'
2. DO NOT use 'dataset.csv' or any other default name.
3. IF the file is missing → raise FileNotFoundError.
4. DO NOT generate fake or artificial data.

---

# Dataset Requirements

The script MUST load dataset from '${dataset_path}' and expect columns:

- 'Email Text'
- 'Email Type'

Label mapping must be robust:

**Phishing = 1**
(phishing, phishing email, spam, malicious)

**Safe/Ham = 0**
(ham, safe, safe email, legitimate, normal, benign)

Unknown labels → DROP.

Dataset must contain at least 10 rows after preprocessing or raise:

ValueError("Dataset invalid or preprocessing removed all rows.")

Automatically detect text column from:
["Email Text","text","body","message"]

Automatically detect label column from:
["Email Type","label","labels","target","Category"]

---

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
