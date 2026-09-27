"""Train a spam/ham classifier used by the FastAPI backend.

Downloads the public SMS Spam Collection dataset (UCI), trains
TF-IDF + Logistic Regression, reports F1 on a held-out test split,
and saves models/spam_vectorizer.joblib + models/spam_model.joblib.

Run once from anywhere:  python fastapi-backend/train_spam_model.py
"""
import io
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    print("Downloading SMS Spam Collection dataset...")
    resp = requests.get(DATASET_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read("SMSSpamCollection")

    df = pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1",
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"Dataset: {len(df)} rows, {int(df.label.sum())} spam")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print("\nTest-set classification report (0=ham, 1=spam):")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "spam_vectorizer.joblib")
    joblib.dump(model, MODELS_DIR / "spam_model.joblib")
    print(f"Saved {MODELS_DIR / 'spam_vectorizer.joblib'} and spam_model.joblib")


if __name__ == "__main__":
    main()
