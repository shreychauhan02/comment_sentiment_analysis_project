"""CI test 1: model artifacts load correctly."""
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_load_lgbm_model():
    model = joblib.load(ROOT / "models" / "lgbm_model.joblib")
    assert type(model).__name__ == "LGBMClassifier"


def test_load_tfidf_vectorizer():
    vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    assert type(vectorizer).__name__ == "TfidfVectorizer"


def test_load_spam_model():
    spam = ROOT / "models" / "spam_model.joblib"
    assert spam.exists(), "Run python fastapi-backend/train_spam_model.py first"
    assert joblib.load(spam) is not None
