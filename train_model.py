# train_model.py
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def train_model(dataset_path, model_type="logistic", max_features=3000, test_size=0.2, random_state=42):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["Email Text"])
    X = df["Email Text"].astype(str)
    y = df["Email Type"].apply(lambda x: 1 if "Phishing" in str(x) else 0)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=test_size, random_state=random_state
    )

    if model_type == "naive_bayes":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=500)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "model": model_type,
        "params": {
            "max_features": max_features,
            "test_size": test_size,
            "random_state": random_state
        },
        "metrics": {
            "accuracy": report.get("accuracy"),
            "precision_phishing": report.get("1", {}).get("precision"),
            "recall_phishing": report.get("1", {}).get("recall"),
            "f1_phishing": report.get("1", {}).get("f1-score")
        },
        "confusion_matrix": cm
    }
    return result

if __name__ == "__main__":
    
    # Argümanlar: dataset_path, model_type, max_features, test_size
    # Örn: python train_model.py Phishing_Email.csv logistic 3000 0.2
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Please pass dataset path as first arg"}))
        sys.exit(1)

    dataset_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "logistic"
    max_features = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    test_size = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

    try:
        out = train_model(dataset_path, model_type, max_features, test_size)
        # Sadece JSON bas (stdout). Debug/log gerekiyorsa stderr kullan.
        print(json.dumps(out))
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


