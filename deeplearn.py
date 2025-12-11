# train_all_models.py
#
# This file is for experiments (NOT used by MCP).
# It runs multiple ML & DL models sequentially on Phishing_Email.csv
# and prints metrics for each one.

#%%
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Deep learning imports (TensorFlow / Keras)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding,
        LSTM,
        GRU,
        Conv1D,
        GlobalMaxPooling1D,
        Dense,
        Dropout,
        Bidirectional,
    )
    from tensorflow.keras.optimizers import Adam
except ImportError:
    tf = None  # we will check this before running DL models

# Transformers (DistilBERT)
try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
except ImportError:
    AutoTokenizer = None
    TFAutoModelForSequenceClassification = None


#%% Dataset loading

def load_data(dataset_path: str):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["Email Text"])
    X = df["Email Text"].astype(str)
    y = df["Email Type"].apply(lambda x: 1 if "Phishing" in str(x) else 0)

    num_total = len(df)
    num_phishing = int(y.sum())
    num_legit = num_total - num_phishing
    print(f"\n=== Dataset Loaded ===")
    print(f"Total emails: {num_total}")
    print(f"Phishing: {num_phishing}")
    print(f"Legitimate: {num_legit}")
    return X, y


#%% Helpers for printing metrics

def print_metrics(name: str, y_test, y_pred):
    print(f"\n======================")
    print(f"RESULTS: {name}")
    print("======================")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    return cm.tolist()


#%% Traditional ML: shared TF-IDF split

def vectorize_and_split(X, y, max_features=3000, test_size=0.2, random_state=42):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=test_size, random_state=random_state
    )
    return vectorizer, X_train, X_test, y_train, y_test


def run_logistic(X, y, max_features=3000, test_size=0.2, random_state=42):
    print("\n>>> Training Logistic Regression (TF-IDF)")
    _, X_train, X_test, y_train, y_test = vectorize_and_split(
        X, y, max_features=max_features, test_size=test_size, random_state=random_state
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = print_metrics("Logistic Regression (TF-IDF)", y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "model": "logistic",
        "params": {
            "max_features": max_features,
            "test_size": test_size,
            "random_state": random_state,
        },
        "metrics": {
            "accuracy": report.get("accuracy"),
            "precision_phishing": report.get("1", {}).get("precision"),
            "recall_phishing": report.get("1", {}).get("recall"),
            "f1_phishing": report.get("1", {}).get("f1-score"),
        },
        "confusion_matrix": cm,
    }


def run_naive_bayes(X, y, max_features=3000, test_size=0.2, random_state=42):
    print("\n>>> Training Naive Bayes (TF-IDF)")
    _, X_train, X_test, y_train, y_test = vectorize_and_split(
        X, y, max_features=max_features, test_size=test_size, random_state=random_state
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = print_metrics("Naive Bayes (TF-IDF)", y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "model": "naive_bayes",
        "params": {
            "max_features": max_features,
            "test_size": test_size,
            "random_state": random_state,
        },
        "metrics": {
            "accuracy": report.get("accuracy"),
            "precision_phishing": report.get("1", {}).get("precision"),
            "recall_phishing": report.get("1", {}).get("recall"),
            "f1_phishing": report.get("1", {}).get("f1-score"),
        },
        "confusion_matrix": cm,
    }


from sklearn.svm import LinearSVC

def run_svm(X, y, max_features=3000, test_size=0.2, random_state=42):
    print("\n>>> Training SVM (LinearSVC, TF-IDF)")
    _, X_train, X_test, y_train, y_test = vectorize_and_split(
        X, y, max_features=max_features, test_size=test_size, random_state=random_state
    )

    model = LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = print_metrics("SVM (LinearSVC)", y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": "svm",
        "params": {"max_features": max_features},
        "metrics": {
            "accuracy": report["accuracy"],
            "precision_phishing": report["1"]["precision"],
            "recall_phishing": report["1"]["recall"],
            "f1_phishing": report["1"]["f1-score"]
        },
        "confusion_matrix": cm
    }



from sklearn.ensemble import RandomForestClassifier

def run_random_forest(X, y, max_features=3000, test_size=0.2, random_state=42):
    print("\n>>> Training Random Forest")

    _, X_train, X_test, y_train, y_test = vectorize_and_split(
        X, y, max_features=max_features, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = print_metrics("Random Forest", y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": "random_forest",
        "params": {"n_estimators": 200},
        "metrics": {
            "accuracy": report["accuracy"],
            "precision_phishing": report["1"]["precision"],
            "recall_phishing": report["1"]["recall"],
            "f1_phishing": report["1"]["f1-score"]
        },
        "confusion_matrix": cm
    }


import xgboost as xgb

def run_xgboost(X, y, max_features=3000, test_size=0.2, random_state=42):
    print("\n>>> Training XGBoost")

    _, X_train, X_test, y_train, y_test = vectorize_and_split(
        X, y, max_features=max_features, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = print_metrics("XGBoost", y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": "xgboost",
        "params": {"n_estimators": 300},
        "metrics": {
            "accuracy": report["accuracy"],
            "precision_phishing": report["1"]["precision"],
            "recall_phishing": report["1"]["recall"],
            "f1_phishing": report["1"]["f1-score"]
        },
        "confusion_matrix": cm
    }


from sklearn.neighbors import KNeighborsClassifier

def run_knn(X, y, max_features=3000, test_size=0.2, random_state=42):
    print("\n>>> Training KNN (5 neighbors)")

    _, X_train, X_test, y_train, y_test = vectorize_and_split(
        X, y, max_features=max_features, test_size=test_size, random_state=random_state
    )

    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = print_metrics("KNN", y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": "knn",
        "params": {"n_neighbors": 5},
        "metrics": {
            "accuracy": report["accuracy"],
            "precision_phishing": report["1"]["precision"],
            "recall_phishing": report["1"]["recall"],
            "f1_phishing": report["1"]["f1-score"]
        },
        "confusion_matrix": cm
    }


#%% DL helper: tokenize & split sequences

def prepare_sequences(
    X,
    y,
    max_features=3000,
    max_len=200,
    test_size=0.2,
    random_state=42,
):
    if tf is None:
        raise ImportError("TensorFlow is required for deep learning models.")

    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=test_size, random_state=random_state
    )
    return tokenizer, X_train, X_test, y_train, y_test, max_len


def build_compile_model(model_type, max_features, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))

    if model_type == "lstm":
        model.add(LSTM(64))
    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(64)))
    elif model_type == "gru":
        model.add(GRU(64))
    elif model_type == "cnn":
        # classic TextCNN style: Conv1D + GlobalMaxPool
        model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
        model.add(GlobalMaxPooling1D())
    elif model_type == "hybrid":
        # CNN + LSTM hybrid
        model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation="relu"))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_sequence_model(
    X,
    y,
    model_type: str,
    max_features=3000,
    max_len=200,
    test_size=0.2,
    random_state=42,
    epochs=3,
    batch_size=64,
):
    if tf is None:
        raise ImportError("TensorFlow is required for deep learning models.")

    model_name_map = {
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "gru": "GRU",
        "cnn": "1D CNN",
        "hybrid": "Hybrid CNN + Dense",
    }
    pretty_name = model_name_map.get(model_type, model_type)

    print(f"\n>>> Training {pretty_name} (Embedding + {pretty_name})")

    _, X_train, X_test, y_train, y_test, max_len = prepare_sequences(
        X,
        y,
        max_features=max_features,
        max_len=max_len,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_compile_model(model_type, max_features, max_len)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    y_pred_prob = model.predict(X_test, verbose=1)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    cm = print_metrics(pretty_name, y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    train_acc = float(history.history["accuracy"][-1])
    val_acc = float(history.history["val_accuracy"][-1])

    return {
        "model": model_type,
        "params": {
            "max_features": max_features,
            "max_len": max_len,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "metrics": {
            "accuracy": report.get("accuracy"),
            "precision_phishing": report.get("1", {}).get("precision"),
            "recall_phishing": report.get("1", {}).get("recall"),
            "f1_phishing": report.get("1", {}).get("f1-score"),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        },
        "confusion_matrix": cm,
    }


def run_lstm(X, y, **kwargs):
    return run_sequence_model(X, y, model_type="lstm", **kwargs)


def run_bilstm(X, y, **kwargs):
    return run_sequence_model(X, y, model_type="bilstm", **kwargs)


def run_gru(X, y, **kwargs):
    return run_sequence_model(X, y, model_type="gru", **kwargs)


def run_cnn(X, y, **kwargs):
    return run_sequence_model(X, y, model_type="cnn", **kwargs)


def run_hybrid(X, y, **kwargs):
    return run_sequence_model(X, y, model_type="hybrid", **kwargs)


#%% DistilBERT (Transformer) – experimental / heavy

def run_distilbert(
    X,
    y,
    test_size=0.2,
    random_state=42,
    model_name="distilbert-base-uncased",
    epochs=2,
    batch_size=16,
):
    if AutoTokenizer is None or TFAutoModelForSequenceClassification is None:
        raise ImportError("transformers is required for DistilBERT. Install with: pip install transformers")

    if tf is None:
        raise ImportError("TensorFlow is required for DistilBERT.")

    print("\n>>> Training DistilBERT (transformer)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    def encode_texts(texts, labels):
        encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="tf",
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (dict(encodings), tf.cast(labels.values, tf.int32))
        )
        return dataset

    train_ds = encode_texts(X_train, y_train).shuffle(1000).batch(batch_size)
    test_ds = encode_texts(X_test, y_test).batch(batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=test_ds, epochs=epochs, verbose=1)

    logits = model.predict(test_ds).logits
    y_pred = np.argmax(logits, axis=-1)

    cm = print_metrics("DistilBERT", y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": "distilbert",
        "params": {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "test_size": test_size,
            "random_state": random_state,
        },
        "metrics": {
            "accuracy": report.get("accuracy"),
            "precision_phishing": report.get("1", {}).get("precision"),
            "recall_phishing": report.get("1", {}).get("recall"),
            "f1_phishing": report.get("1", {}).get("f1-score"),
        },
        "confusion_matrix": cm,
    }


#%% Main orchestration

if __name__ == "__main__":
    dataset_path = "Phishing_Email.csv"

    X, y = load_data(dataset_path)

    all_results = {}

    # Traditional ML
    all_results["logistic"] = run_logistic(X, y)
    all_results["naive_bayes"] = run_naive_bayes(X, y)
    all_results["svm"] = run_svm(X, y)
    all_results["random_forest"] = run_random_forest(X, y)
    all_results["knn"] = run_knn(X, y)


    # XGBoost varsa:
    try:
        all_results["xgboost"] = run_xgboost(X, y)
    except:
        print("[WARN] XGBoost not installed")
    
    
    # Deep learning (sequence models)
    if tf is not None:
        all_results["lstm"] = run_lstm(X, y)
        all_results["bilstm"] = run_bilstm(X, y)
        all_results["gru"] = run_gru(X, y)
        all_results["cnn"] = run_cnn(X, y)
        all_results["hybrid"] = run_hybrid(X, y)
    else:
        print("\n[WARN] TensorFlow not available – skipping LSTM/BiLSTM/GRU/CNN/Hybrid.")

    # DistilBERT (can be slow; you can comment this out if needed)
    if AutoTokenizer is not None and tf is not None:
        all_results["distilbert"] = run_distilbert(X, y)
    else:
        print("\n[WARN] transformers or TensorFlow not available – skipping DistilBERT.")

    # Print a compact JSON summary at the end
    print("\n===== SUMMARY (JSON) =====")
    print(json.dumps(all_results, indent=2))
