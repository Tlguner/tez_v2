You are a senior ML engineer.
The cleaned dataset is located at: '${cleaned_dataset_path}'
Total clean rows: ${cleaned_rows}
Class balance: ${class_balance}
Mode of operation: ${mode}
Note: The cleaned dataset contains the columns: 'text' (the cleaned email text), 'label' (the binary integer label, 0 or 1), and metadata features: 'url_count', 'has_ip', 'urgency_score', 'has_html'. Use these column names.

TASK:
- Write a dynamically tailored Python script.
- If Mode is 'ML': Use ONLY Pandas, Scikit-learn, Scipy, and Numpy. Do NOT import TensorFlow or Keras.
- If Mode is 'DL': Use Pandas, TensorFlow/Keras, and Numpy.
- Perform stratified 80/20 train/test split.
- If Mode is 'ML':
  - Extract TF-IDF features (max_features = 10000, ngram_range=(1, 2)) from the 'text' column. The ngram_range=(1,2) enables both unigram and bigram features to capture two-word phishing patterns like 'verify account', 'click here', 'update password'.
  - Extract the metadata columns: 'url_count', 'has_ip', 'urgency_score', 'has_html'.
  - Scale these 4 metadata columns to the range [0, 1] using Scikit-learn's `MinMaxScaler`. Fit the scaler on the train split, and transform both the train and test splits.
  - Horizontally stack the TF-IDF sparse matrix and the scaled metadata features using `scipy.sparse.hstack` (yielding a combined matrix of shape [rows, 10004]).
  - Save the vectorizer using pickle directly to: '${vectorizer_pkl_path}'
  - Save the combined X_train_ml / X_test_ml as sparse npz directly to: '${features_npy_path}' / '${test_features_path}'
  - Save y_train_ml / y_test_ml as numpy binary arrays directly to: '${labels_npy_path}' / '${test_labels_path}'
- If Mode is 'DL':
  - Tokenize and pad sequence (maxlen = 400, vocabulary size = 15000) using only the 'text' column.
  - Save tokenizer using pickle directly to: '${vectorizer_pkl_path}'
  - Save X_train_dl, X_test_dl, y_train_dl, y_test_dl as numpy binary arrays directly to: '${features_npy_path}' / '${test_features_path}', '${labels_npy_path}' / '${test_labels_path}'
- At successful end of execution, print EXACTLY this JSON between <<<JSON_START>>> and <<<JSON_END>>>:
{
  "ok": true,
  "mode": "${mode}",
  "train_shape": [rows, cols_or_seqlen],
  "test_shape": [rows, cols_or_seqlen],
  "features_npy_path": "${features_npy_path}",
  "labels_npy_path": "${labels_npy_path}",
  "vectorizer_pkl_path": "${vectorizer_pkl_path}",
  "test_features_path": "${test_features_path}",
  "test_labels_path": "${test_labels_path}"
}
- Do NOT output any description, only python code in ```python fences.
