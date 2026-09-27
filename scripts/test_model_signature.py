"""CI test 2: model input/output contract (the 'signature').

The model must accept exactly tfidf_width + 3 hand-crafted features and
output only the three sentiment classes 0/1/2.
"""
import joblib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_feature_contract():
    vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    model = joblib.load(ROOT / "models" / "lgbm_model.joblib")

    tfidf_width = len(vectorizer.vocabulary_)
    expected_width = tfidf_width + 3  # word_count, char_count, avg_word_length
    assert model.n_features_in_ == expected_width, (
        f"model expects {model.n_features_in_} features, pipeline produces {expected_width}"
    )


def test_prediction_labels():
    vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    model = joblib.load(ROOT / "models" / "lgbm_model.joblib")
    width = len(vectorizer.vocabulary_) + 3
    dummy = np.zeros((5, width))
    preds = model.predict(dummy)
    assert set(preds).issubset({0, 1, 2}), f"unexpected classes: {set(preds)}"
