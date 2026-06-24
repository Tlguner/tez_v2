You are an expert deep learning engineer.

# Dataset Requirements

Load dataset from EXACT path: '${absPath}'

Automatically detect text column:

- If any of these exist, pick the first match:
  ["text_combined","Email Text","email_text","text","body","message","content","email","mail","raw_text","sentence","clean_text"]

If none found:

- Fallback: choose the column with dtype object/string that has the largest average string length.
- If still none, raise ValueError("Could not find a text column")

Automatically detect label column:

- If any of these exist, pick the first match:
  ["label","Email Type","labels","target","Category","class","y","is_phishing","spam","phish"]

If none found:

- Fallback: choose a column with <= 10 unique values (after dropping NaNs) and not equal to the chosen text column.
- If still none, raise ValueError("Could not find a label column")

Perform label mapping:

- Phishing / spam variants → 1
- Ham / safe / legitimate variants → 0

1. If label column dtype is numeric:

   - If unique values ⊆ {0,1}:
     - Use label as-is (0 = safe, 1 = phishing)
   - Else:
     - Raise ValueError("Numeric label column must contain only 0 and 1")

2. If label column dtype is string/object:
   - Map phishing/spam variants → 1
   - Map ham/safe/legitimate variants → 0
   - Drop rows with unmappable labels

Valid phishing labels (case-insensitive):
["phishing", "phishing email", "spam", "malicious"]

Valid safe labels:
["ham", "safe", "safe email", "legitimate", "normal", "benign"]

Drop rows:

- Fill NaN text with empty string ""
- Drop rows with NaN label only

After mapping:

- Dataset MUST contain both 0 and 1
  Else raise:
  ValueError("Dataset mapping failed: need both phishing and safe labels")

---

# Train/Test Split

Use stratified 80/20 split.

---

# Text Processing

- Use Keras Tokenizer
- Convert texts to sequences
- Pad sequences with max_len = 200

---

# Deep Learning Hyperparameter Optimization & Pruning (OPTUNA - MANDATORY)

You MUST use `optuna` to optimize the hyperparameter configuration for the deep learning models.

Requirements:
1. **Search Space for DL Architectures**:
   - Embedding Dimension: `embedding_dim` (int: 32 to 128)
   - Hidden Units: `units` (int: 32 to 128)
   - Dropout Rate: `dropout_rate` (float: 0.1 to 0.5)
   - Learning Rate: `learning_rate` (loguniform: 1e-4 to 1e-2)

2. **Bayesian Optimization & Early Pruning (Keras Integration)**:
   - For each DL architecture class (LSTM, BiLSTM, GRU, CNN, etc.), run an Optuna study with at least 5 trials to select the best hyperparameter set.
   - Use `optuna.integration.TFKerasPruningCallback` (or a custom pruning callback checking epoch loss/accuracy against previous trials) passed to Keras `fit()` callback list.
   - If a trial yields poor validation accuracy (e.g., lower than 60%) within the first 3 epochs, prune the trial immediately to save execution time.

---

# Deep Learning Models to Train (Mandatory)

Train optimized versions of ALL of the following models using their best Optuna configuration. Keep architectures aligned with these definitions:
1. LSTM: Embedding -> LSTM -> Dense(1, activation='sigmoid')
2. BiLSTM: Embedding -> Bidirectional(LSTM) -> Dense(1, activation='sigmoid')
3. GRU: Embedding -> GRU -> Dense(1, activation='sigmoid')
4. 1D CNN: Embedding -> Conv1D -> GlobalMaxPooling1D -> Dropout -> Dense(1, activation='sigmoid')
5. Hybrid CNN + Dense: Embedding -> Conv1D -> GlobalMaxPooling1D -> Dense(dense_units) -> Dropout -> Dense(1, activation='sigmoid')

Each model must be trained and evaluated independently.

---

# Additional Custom Model ("auto_model")

After all mandatory models, create and train **exactly one** additional model named:

"auto_model"

Requirements:

- Architecture must be fully designed by the AI (e.g. customized Attention or multi-branch CNN).
- Hyperparameters must be optimized via Optuna (at least 5 trials).
- Must be evaluated with the same metrics.

---

# Metrics Required for EACH Model

Compute and store:

- accuracy
- precision
- recall
- f1
- confusion matrix
- train accuracy history
- validation accuracy history
- training time (seconds)

---

# Keras Imports Required

The script MUST explicitly import:

from tensorflow.keras.layers import (
LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
GlobalMaxPooling1D, Dense, Dropout, Embedding
)
import optuna
from optuna.integration import TFKerasPruningCallback

---

# HARD RULES

- DO NOT use BERT, DistilBERT, Transformers, or HuggingFace models
- Use ONLY: pandas, numpy, sklearn, tensorflow, optuna
- Do NOT generate fake data
- If dataset file does not exist → raise FileNotFoundError
- Training logs (epochs, loss, accuracy) MAY be printed.
- ALL logs MUST be printed BEFORE <<<JSON_START>>>.
- NOTHING may be printed AFTER <<<JSON_END>>>.

---

# Final Output (ONLY JSON)

At the very end print EXACTLY one JSON object structured as follows:

{
  "optuna_statistics": {
    "total_trials": ...,
    "pruned_trials": ...,
    "optimization_time_seconds": ...,
    "pruning_ratio": ...  # (pruned_trials / total_trials)
  },
  "results": {
    "lstm": {
      "accuracy": ...,
      "precision": ...,
      "recall": ...,
      "f1": ...,
      "confusion_matrix": [[tn, fp], [fn, tp]],
      "train_accuracy_history": [...],
      "validation_accuracy_history": [...],
      "time_s": ...,
      "optuna_gain": {
        "baseline_accuracy": ...,
        "optimized_accuracy": ...,
        "gain": ...
      }
    },
    "bilstm": { ... },
    "gru": { ... },
    "cnn": { ... },
    "hybrid": { ... },
    "auto_model": {
      "description": "short description of architecture",
      "accuracy": ...,
      "precision": ...,
      "recall": ...,
      "f1": ...,
      "confusion_matrix": [[tn, fp], [fn, tp]],
      "train_accuracy_history": [...],
      "validation_accuracy_history": [...],
      "time_s": ...,
      "optuna_gain": {
        "baseline_accuracy": ...,
        "optimized_accuracy": ...,
        "gain": ...
      }
    }
  }
}

Use:
print("<<<JSON_START>>>")
print(json.dumps(results, indent=2))
print("<<<JSON_END>>>")

CRITICAL TRAINING LIMITS:

- Each Optuna trial model training MUST run for at most 5 epochs (to keep search fast).
- Final retrained models (with the best hyperparameters found) MUST train for at most 10 epochs.
- Use EarlyStopping with:
  monitor="val_loss"
  patience=2
  restore_best_weights=True
  - Do NOT use MultiHeadAttention.
- Use ONLY Sequential API.

