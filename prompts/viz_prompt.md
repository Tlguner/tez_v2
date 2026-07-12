You are an expert ML/DL visualization engineer specializing in academic thesis figures.

# Task Overview

Generate a Python script that produces 5 categories of advanced visualizations for a phishing email detection thesis.
All paths are provided as parameters — never hardcode any paths.

# Input Parameters

- Pipeline Mode: "${mode}"  (ML or DL)
- Models Directory: "${models_dir}"
- Test Features Path: "${test_features_path}"
- Test Labels Path: "${test_labels_path}"
- Cleaned Dataset CSV: "${cleaned_dataset_path}"
- Vectorizer PKL: "${vectorizer_pkl_path}"
- Models Metrics JSON (training results): "${models_metrics_json}"
- Output Directory: "${viz_output_dir}"

# Data Loading Rules

- Mode ML: Load test features with `scipy.sparse.load_npz("${test_features_path}")`.
  Keep sparse for SHAP and calibration. Convert to dense ONLY where explicitly required.
  CRITICAL PREDICTION RULE (ML Mode): Both `GaussianNB` and `auto_model` (HistGradientBoostingClassifier) do NOT support sparse matrices. Whenever obtaining predictions or probability scores from these two models (e.g., in Task 2 for Error Analysis, Task 5 for Calibration, etc.), you MUST convert the sparse test features matrix (X_test) to a dense numpy array using `.toarray()`.
  Use this example pattern:
  ```python
  if model_name in ["GaussianNB", "auto_model"]:
      X_eval = X_test.toarray()
  else:
      X_eval = X_test
  preds = model.predict(X_eval)
  ```
  For all other ML models, pass the sparse matrix directly to preserve system resources.
- Mode DL: Load test features with `np.load("${test_features_path}")`.
- Test labels: `y_test = np.ravel(np.load("${test_labels_path}")).astype(int)` (always 1D integer).
- Models metrics: Load JSON from "${models_metrics_json}" — contains {model_name: {accuracy, precision, recall, f1_score, description, ...}}.

# Model Loading Rules

- Mode ML: Load .pkl files from "${models_dir}" using `joblib.load()`.
  Available: LogisticRegression.pkl, MultinomialNB.pkl, GaussianNB.pkl, RandomForestClassifier.pkl,
             LinearSVC.pkl, SGDClassifier.pkl, VotingClassifier.pkl, auto_model.pkl
- Mode DL: Load .h5 files from "${models_dir}" using `tf.keras.models.load_model()`.
  Available: lstm.h5, bilstm.h5, gru.h5, cnn.h5, hybrid.h5, text_cnn.h5, stacked_lstm.h5, cnn_lstm.h5, auto_model.h5
  IMPORTANT (loading text_cnn): text_cnn.h5 uses the Keras Functional API (multi-input). Load with `tf.keras.models.load_model(path, compile=False)`.
- CRITICAL FILENAME MAPPING RULE: The model name keys loaded from "${models_metrics_json}" might be dynamically formatted display names (e.g. `TextCNN (Multi-kernel)` or `VotingClassifier (LR + RF + SGD)`). When loading a model file, you MUST map these display names to the exact file names on disk. Implement this mapping logically:
  - If name contains "Logistic" or "LogisticRegression" -> "LogisticRegression.pkl"
  - If name contains "Multinomial" -> "MultinomialNB.pkl"
  - If name contains "Gaussian" -> "GaussianNB.pkl"
  - If name contains "RandomForest" or "Random Forest" -> "RandomForestClassifier.pkl"
  - If name contains "LinearSVC" -> "LinearSVC.pkl"
  - If name contains "SGD" -> "SGDClassifier.pkl"
  - If name contains "Voting" -> "VotingClassifier.pkl"
  - If name contains "bilstm" or "BiLSTM" -> "bilstm.h5"
  - If name contains "stacked_lstm" or "Stacked LSTM" -> "stacked_lstm.h5"
  - If name contains "cnn_lstm" or "CNN-LSTM" -> "cnn_lstm.h5"
  - If name contains "lstm" or "LSTM" (and not matched above) -> "lstm.h5"
  - If name contains "gru" or "GRU" -> "gru.h5"
  - If name contains "text_cnn" or "TextCNN" or "Multi-kernel" -> "text_cnn.h5"
  - If name contains "cnn" or "CNN" (and not matched above) -> "cnn.h5"
  - If name contains "hybrid" or "Hybrid" -> "hybrid.h5"
  - If name contains "auto_model" or "AutoModel" -> "auto_model.pkl" (ML) or "auto_model.h5" (DL)
  CRITICAL (loading auto_model): Since a custom `BahdanauAttention` layer and several functional keras/tensor operations (which serialize as `ExpandDims`, `Sum`, and `Multiply` layers) are used in the `auto_model.h5` model, you must define these custom classes in the visualization script and pass them in `custom_objects` when loading:
  ```python
  class BahdanauAttention(tf.keras.layers.Layer):
      def __init__(self, units, **kwargs):
          super(BahdanauAttention, self).__init__(**kwargs)
          self.units = units
          self.W1 = tf.keras.layers.Dense(units)
          self.W2 = tf.keras.layers.Dense(units)
          self.V = tf.keras.layers.Dense(1)
      def call(self, query, values):
          query_with_time_axis = tf.expand_dims(query, 1)
          score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
          attention_weights = tf.nn.softmax(score, axis=1)
          context_vector = attention_weights * values
          context_vector = tf.reduce_sum(context_vector, axis=1)
          return context_vector
      def get_config(self):
          config = super(BahdanauAttention, self).get_config()
          config.update({"units": self.units})
          return config

  class ExpandDims(tf.keras.layers.Layer):
      def __init__(self, axis, **kwargs):
          super().__init__(**kwargs)
          self.axis = axis
      def call(self, inputs, **kwargs):
          return tf.expand_dims(inputs, axis=self.axis)
      def get_config(self):
          config = super().get_config()
          config.update({"axis": self.axis})
          return config

  class Sum(tf.keras.layers.Layer):
      def __init__(self, axis, keepdims=False, **kwargs):
          super().__init__(**kwargs)
          self.axis = axis
          self.keepdims = keepdims
      def call(self, inputs, **kwargs):
          return tf.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)
      def get_config(self):
          config = super().get_config()
          config.update({"axis": self.axis, "keepdims": self.keepdims})
          return config

  class PatchedMultiply(tf.keras.layers.Layer):
      """
      Keras 3-compatible element-wise multiply layer.
      IMPORTANT: Functional model configs deserialized from H5 files might call the layer
      with multiple positional arguments (e.g. layer(t1, t2)) instead of a single list (e.g. layer([t1, t2])).
      To prevent 'TypeError: too many positional arguments' during Keras 3 signature binding
      (inside CallSpec.bind), the call() method signature MUST accept variadic positional arguments (*args).
      """
      def call(self, inputs, *args, **kwargs):
          if args:
              all_inputs = [inputs] + list(args)
          elif isinstance(inputs, (list, tuple)):
              all_inputs = list(inputs)
          else:
              all_inputs = [inputs]
          
          result = all_inputs[0]
          for t in all_inputs[1:]:
              result = result * t
          return result

      def get_config(self):
          return super().get_config()
  ```
  Load `auto_model` using: `tf.keras.models.load_model(path, custom_objects={'BahdanauAttention': BahdanauAttention, 'ExpandDims': ExpandDims, 'Sum': Sum, 'Multiply': PatchedMultiply})`.
- On any model load error, skip that model and print a warning to stderr (do NOT exit).
- Ensure output directory exists: `os.makedirs("${viz_output_dir}", exist_ok=True)`

---

# TASK 1: SHAP Feature Importance

Imports: `import shap`

## ML Mode (shap.TreeExplainer)
- Apply ONLY to tree-based and gradient boosting models: RandomForestClassifier, auto_model (HistGradientBoosting).
- Use `shap.TreeExplainer(model)`.
- Get feature names from vectorizer: `feature_names = vectorizer.get_feature_names_out()`.
- Use at most 500 test samples for explanation (slice X_test[:500] converted to dense array).
- Plot: `shap.summary_plot(shap_values, X_sample_dense, feature_names=feature_names, max_display=20, show=False)`
- Save each model's SHAP plot separately as: `${viz_output_dir}/shap_{clean_model_name}.png` (dpi=150, bbox_inches='tight').
  **CRITICAL FILENAME SANITIZATION RULE**: To prevent Windows MAX_PATH (260-character) path length limits or invalid filename character errors, the model name used in the filename MUST be sanitized. Strip out any parenthetical details, layer specifications, spaces, parentheses, arrows, brackets, and keep ONLY a short alphanumeric identifier (e.g., use `shap_auto_model.png`, `shap_RandomForestClassifier.png`).
- Also generate a combined figure showing both models side by side if possible.

## DL Mode (shap.DeepExplainer with GradientExplainer and KernelExplainer fallbacks)
- Apply ONLY to the best performing DL model (select by highest accuracy from models_metrics_json).
- Try `shap.DeepExplainer(model, background)` first (background = X_test[:50], sample = X_test[:100]).
- If it fails, fallback to `shap.GradientExplainer(model, background)`.
- If GradientExplainer also fails (common in Keras 3/TF 2.16+), fallback to `shap.KernelExplainer(lambda x: model.predict(x, verbose=0), background)` using a smaller sample (e.g. background of 10 and sample of 15, with `nsamples=50` to keep it fast).
- SHAP values for DL are token-level — plot mean absolute importance per token position.
  Create a bar chart: x=token positions (0..199), y=mean |shap_value| across samples.
- Save as: `${viz_output_dir}/shap_dl_best_model.png` (dpi=150).
- On any SHAP error: skip and log to stderr, do NOT exit.

# TASK 2: Error Analysis (Misclassified Examples) - SKIPPED BY USER REQUEST
- This task is skipped. Do NOT implement error analysis tables or save `error_analysis_{model_name}.png` or `error_analysis_summary.csv` to speed up pipeline execution and prevent unneeded visual clutter.

---

# TASK 3: Word Cloud

Imports: `from wordcloud import WordCloud`

- Load cleaned dataset CSV, detect text column and label column automatically.
- Split text into: phishing_texts (label=1) and legit_texts (label=0).
- Generate two word clouds:
  - Phishing word cloud: red/orange colormap, background=black.
  - Legitimate word cloud: green/teal colormap, background=white.
- Remove common English stopwords (`from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS`).
- Figure: 2 subplots side by side, 16x8 inches, title "Phishing vs Legitimate Email - Most Frequent Words".
- Save as: `${viz_output_dir}/wordcloud_comparison.png` (dpi=150, bbox_inches='tight').
- Also save individual files: `${viz_output_dir}/wordcloud_phishing.png` and `${viz_output_dir}/wordcloud_legit.png`.

---

# TASK 4: t-SNE Embedding Visualization

Imports: `from sklearn.manifold import TSNE`, `from sklearn.decomposition import TruncatedSVD`

## ML Mode
- Sample 1000 test examples (balanced: 500 phishing + 500 legit if available).
- Apply TruncatedSVD(n_components=50) on sparse X_test sample to reduce dimensions first.
- Then apply TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000).
- Plot 2D scatter: phishing=red dots, legit=blue dots. Alpha=0.6, s=20.
- Title: "t-SNE Visualization of TF-IDF Email Embeddings".
- Save as: `${viz_output_dir}/tsne_embedding.png` (dpi=150, bbox_inches='tight').

## DL Mode
- Load the best DL model. Build a feature extractor model using the best model's second-to-last layer:
  `extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)`
- Extract embeddings: `embeddings = extractor.predict(X_test[:1000])`.
- If model uses Functional API (multi-input), skip embedding extraction and fall back to ML mode approach.
- Apply TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000) on embeddings.
- Same scatter plot style as ML mode.
- Save as: `${viz_output_dir}/tsne_embedding.png`.

---

# TASK 5: Calibration Curve

Imports: `from sklearn.calibration import calibration_curve`

- For each model that supports probability output:
  - ML: use `.predict_proba(X_test)[:, 1]` if available, skip models without predict_proba
        (LinearSVC, SGDClassifier with hinge loss → try `.decision_function()` + sigmoid transform as fallback).
  - DL: use `model.predict(X_test).ravel()` for probability scores.
- Compute: `fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)`.
- Plot all models on the same figure:
  - Diagonal line (perfectly calibrated) in black dashed.
  - Each model as a separate line with markers.
- Figure size: 10x8, title: "Calibration Curves — Probability Reliability".
- Save as: `${viz_output_dir}/calibration_curve.png` (dpi=150, bbox_inches='tight').

---

# General Rules

- Import `warnings; warnings.filterwarnings("ignore")` at the top.
- Use `matplotlib.use('Agg')` to avoid display errors on headless servers.
- All plot text, titles, labels, axis ticks MUST be in English.
- Use a consistent color palette across plots (e.g. seaborn-v0_8-darkgrid style or similar).
- If a task fails for any reason, catch the exception, print a stderr warning, and continue to the next task.
- Do NOT exit on partial failure — complete as many tasks as possible.

---

# Output (stdout only)

At the end, print EXACTLY this JSON between <<<JSON_START>>> and <<<JSON_END>>>:

<<<JSON_START>>>
{
  "ok": true,
  "viz_output_dir": "${viz_output_dir}",
  "shap_plots": ["list of generated shap png paths"],
  "error_analysis_plots": {"model_name": "path", "...": "..."},
  "error_summary_csv": "${viz_output_dir}/error_analysis_summary.csv",
  "wordcloud_comparison_plot": "${viz_output_dir}/wordcloud_comparison.png",
  "wordcloud_phishing_plot": "${viz_output_dir}/wordcloud_phishing.png",
  "wordcloud_legit_plot": "${viz_output_dir}/wordcloud_legit.png",
  "tsne_plot": "${viz_output_dir}/tsne_embedding.png",
  "calibration_plot": "${viz_output_dir}/calibration_curve.png"
}
<<<JSON_END>>>

- For any plot that failed to generate, set its value to null in the JSON.
- Do NOT print anything after <<<JSON_END>>>.
- DO NOT output any explanation or markdown code fences other than ```python.
