You are an expert deep learning engineer.

# Dataset Requirements

Load dataset from EXACT path: '${absPath}'

Automatically detect:

- Text column from: ["Email Text","text","body","message"]
- Label column from: ["Email Type","label","labels","target","Category"]

Perform label mapping:

- Phishing/spam → 1
- Ham/safe/legitimate → 0

Drop rows:

- Unknown labels
- NaN text or label

Dataset must contain both 0 and 1 after mapping, else raise:
ValueError("Dataset mapping failed: need both phishing and safe labels")

---

# Train/Test Split

Use stratified 80/20 split.

---

# Deep Learning Models to Train (Mandatory)

Train ALL of these with Tokenizer + padded sequences (max_len=200):

- LSTM
- BiLSTM
- GRU
- 1D CNN
- Hybrid CNN + Dense

---

# Additional Custom Model ("auto_model")

After all mandatory models, create & train **one extra custom model** named `"auto_model"`.

This architecture must be:

- Fully designed by the AI
- Can use Attention, Transformer encoder, CNN-RNN hybrid, multi-branch CNN, etc.
- MUST be evaluated with the same metrics as other models

---

# Optional Model (DistilBERT)

Include `"distilbert"` key ONLY if training is successful and transformers is installed.

---

# Metrics Required for EACH Model

- accuracy
- precision
- recall
- f1
- confusion matrix
- train/val accuracy history
- training time

---

# Keras Imports Required

from tensorflow.keras.layers import (
LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
GlobalMaxPooling1D, Dense, Dropout, Embedding
)

If your custom model uses Attention / MultiHeadAttention:
from tensorflow.keras.layers import MultiHeadAttention

---

# Final Output (ONLY JSON)

At the end print EXACTLY:

{
"lstm": {...},
"bilstm": {...},
"gru": {...},
"cnn": {...},
"hybrid": {...},
"auto_model": {
"description": "short description",
...
},
"distilbert": {...} // only if trained
}

Do not print anything after the JSON.

Use:
print(json.dumps(results), flush=True)
