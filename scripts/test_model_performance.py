"""CI test 3: model performance gate.

Rebuilds test features with the saved vectorizer and fails the pipeline
if test accuracy drops below the threshold (current model: ~0.851).
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
MIN_ACCURACY = 0.80


def test_test_accuracy():
    model = joblib.load(ROOT / "models" / "lgbm_model.joblib")
    vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    df = pd.read_csv(ROOT / "data" / "interim" / "test_processed.csv").dropna()

    features = np.hstack(
        [
            vectorizer.transform(df["comment"]).toarray(),
            df[["word_count", "char_count", "avg_word_length"]].values,
        ]
    )
    accuracy = accuracy_score(df["category"], model.predict(features))
    print(f"\nTest accuracy: {accuracy:.4f}")
    assert accuracy >= MIN_ACCURACY, f"accuracy {accuracy:.4f} below gate {MIN_ACCURACY}"
