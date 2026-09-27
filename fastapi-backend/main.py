import os
import re
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DAGSHUB_REPO_OWNER = "shreychauhan02"
DAGSHUB_REPO_NAME = "my_final_project"
MODEL_NAME = "sentiment_analysis-lgbm-model"
VECTORIZER_NAME = "sentiment_analysis-vectorizer"

LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

dagshub_token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_USER_TOKEN is not set. Add DAGSHUB_USER_TOKEN=<your token> to "
        f"{PROJECT_ROOT / '.env'} (or the platform env vars) and restart the server."
    )
# The dagshub SDK reads DAGSHUB_USER_TOKEN once, at import time — so the env var
# must be set BEFORE "import dagshub", or it falls back to interactive OAuth
# (impossible on a headless server like Render).
os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

import dagshub
from dagshub.common import config as _dagshub_config

_dagshub_config.token = dagshub_token

youtube_api_key = os.getenv("YOUTUBE_API_KEY")
if not youtube_api_key:
    raise EnvironmentError(
        "YOUTUBE_API_KEY is not set. Add YOUTUBE_API_KEY=<your YouTube Data API v3 key> to "
        f"{PROJECT_ROOT / '.env'} and restart the server."
    )

# Points MLflow tracking at https://dagshub.com/shreychauhan02/my_final_project.mlflow
dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

import nltk

for corpus in ("corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"):
    try:
        nltk.data.find(corpus)
    except LookupError:
        nltk.download(corpus.split("/")[-1])

STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
LEMMATIZER = WordNetLemmatizer()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def load_from_registry(model_name: str):
    client = MlflowClient()
    versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    versions = sorted(versions, key=lambda mv: int(mv.version), reverse=True)
    if not versions:
        raise RuntimeError(
            f"No registered versions found for '{model_name}' in the Dagshub MLflow registry. "
            "Run `dvc repro` (model_registration stage) first."
        )
    uri = f"models:/{model_name}/{versions[0].version}"
    return mlflow.sklearn.load_model(uri)


print(f"Loading '{MODEL_NAME}' and '{VECTORIZER_NAME}' from Dagshub registry...")
model = load_from_registry(MODEL_NAME)
vectorizer = load_from_registry(VECTORIZER_NAME)

SPAM_MODEL_PATH = PROJECT_ROOT / "models" / "spam_model.joblib"
SPAM_VECTORIZER_PATH = PROJECT_ROOT / "models" / "spam_vectorizer.joblib"
if not SPAM_MODEL_PATH.exists() or not SPAM_VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        "Spam model missing. Run 'python fastapi-backend/train_spam_model.py' once, then restart."
    )
spam_model = joblib.load(SPAM_MODEL_PATH)
spam_vectorizer = joblib.load(SPAM_VECTORIZER_PATH)
print("Models loaded.")


def preprocess_comment(comment: str) -> tuple[str, int, int, float]:
    """Mirror of src/data/data_preprocessing.preprocess_and_create_features (training behaviour)."""
    comment = str(comment).strip().lower()
    comment = " ".join(word for word in comment.split() if word not in STOP_WORDS)
    comment = re.sub(r"\n", " ", comment)
    # Training computes word_count as len(comment) (characters), not tokens.
    # Mirrored on purpose so inference matches what the model was fitted on.
    word_count = len(comment)
    char_count = len(comment)
    avg_word_length = char_count / (word_count + 1)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
    comment = " ".join(LEMMATIZER.lemmatize(word) for word in comment.split())
    return comment, word_count, char_count, avg_word_length


def predict_batch(comments: list[str]) -> list[int]:
    processed = [preprocess_comment(c) for c in comments]
    df = pd.DataFrame(
        processed,
        columns=["preprocessed_comment", "word_count", "char_count", "avg_word_length"],
    )
    tfidf = vectorizer.transform(df["preprocessed_comment"].values).toarray()
    features = np.hstack([tfidf, df[["word_count", "char_count", "avg_word_length"]].values])
    return [int(p) for p in model.predict(features)]


def detect_spam(comments: list[str]) -> list[int]:
    """Spam classifier trained by train_spam_model.py (TF-IDF + LogisticRegression, spam F1 ~0.91)."""
    return [int(p) for p in spam_model.predict(spam_vectorizer.transform(comments))]


def extract_themes(comments: list[str], top_n: int = 20) -> list[dict]:
    counts: dict[str, int] = {}
    for c in comments:
        for word in re.sub(r"[^A-Za-z0-9\s]", " ", c.lower()).split():
            if len(word) < 3 or word in STOP_WORDS or word.isdigit():
                continue
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [{"word": w, "count": n} for w, n in ranked[:top_n]]


def compute_trend(comments: list[dict], predictions: list[int]) -> list[dict]:
    """Monthly sentiment percentages, matching the 'Monthly Sentiment Percentage Over Time' chart."""
    df = pd.DataFrame(
        {"timestamp": [c["timestamp"] for c in comments], "label": [LABELS[p] for p in predictions]}
    )
    df["month"] = (
        pd.to_datetime(df["timestamp"], errors="coerce").dt.to_period("M").astype(str)
    )
    df = df[df["month"] != "NaT"]
    counts = df.groupby(["month", "label"]).size().unstack(fill_value=0)
    for label in ("NEGATIVE", "NEUTRAL", "POSITIVE"):
        if label not in counts.columns:
            counts[label] = 0
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0).mul(100).round(1)
    return [
        {
            "month": month,
            "comments": int(totals.loc[month]),
            "NEGATIVE": float(pct.loc[month, "NEGATIVE"]),
            "NEUTRAL": float(pct.loc[month, "NEUTRAL"]),
            "POSITIVE": float(pct.loc[month, "POSITIVE"]),
        }
        for month in pct.index
    ]


def gemini_summary(comments: list[dict], predictions: list[int], percentages: dict, themes: list[dict]) -> str:
    """Ask Gemini for a short natural-language summary of the comment section."""
    by_label: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for c, p in zip(comments, predictions):
        if len(by_label[p]) < 8:
            by_label[p].append(c["text"][:200])
    prompt = (
        "You are analyzing YouTube comments for one video. "
        f"Stats: {len(comments)} comments — {percentages['POSITIVE']}% positive, "
        f"{percentages['NEUTRAL']}% neutral, {percentages['NEGATIVE']}% negative. "
        f"Most frequent words: {', '.join(t['word'] for t in themes[:10])}.\n"
        f"Sample positive comments: {by_label[2]}\n"
        f"Sample neutral comments: {by_label[1]}\n"
        f"Sample negative comments: {by_label[0]}\n"
        "Write a 3-4 sentence summary of what this audience thinks and feels about "
        "the video: overall mood, what they discuss, and any notable complaints or praise. "
        "Be concrete, no bullet points, no intro like 'Here is a summary'."
    )
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 300, "temperature": 0.4},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error {resp.status_code}: {resp.text[:300]}",
        )
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def fetch_comments(video_id: str, max_comments: int = 500) -> list[dict]:
    comments: list[dict] = []
    page_token = ""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    while len(comments) < max_comments:
        resp = requests.get(
            url,
            params={
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "textFormat": "plainText",
                "pageToken": page_token,
                "key": youtube_api_key,
            },
            timeout=30,
        )
        data = resp.json()
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"YouTube API error {resp.status_code}: {data.get('error', {}).get('message', data)}",
            )
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "text": snippet["textOriginal"],
                    "timestamp": snippet["publishedAt"],
                    "author": snippet.get("authorDisplayName", "Unknown"),
                }
            )
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break
    return comments[:max_comments]


app = FastAPI(title="Comment Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeInput(BaseModel):
    videoId: str


@app.get("/")
async def home():
    return {"message": "Comment sentiment API is running", "model": MODEL_NAME}


@app.post("/analyze")
async def analyze(data: AnalyzeInput):
    video_id = data.videoId.strip()
    if not re.fullmatch(r"[\w-]{11}", video_id):
        raise HTTPException(status_code=400, detail=f"Invalid YouTube video id: '{video_id}'")

    comments = fetch_comments(video_id)
    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this video (comments may be disabled).")

    texts = [c["text"] for c in comments]
    predictions = predict_batch(texts)
    spam_flags = detect_spam(texts)

    total = len(comments)
    counts = {label: 0 for label in LABELS.values()}
    for pred in predictions:
        counts[LABELS[pred]] += 1
    percentages = {label: round(count / total * 100, 1) for label, count in counts.items()}

    avg_word_count = round(sum(len(t.split()) for t in texts) / total, 1)
    unique_authors = len({c["author"] for c in comments})
    spam_count = sum(spam_flags)
    avg_score = round(sum(predictions) / total / 2 * 10, 1)

    themes = extract_themes(texts)
    trend = compute_trend(comments, predictions)
    top_theme_names = ", ".join(t["word"] for t in themes[:5])
    summary = (
        f"{total} comments analyzed. {percentages['POSITIVE']}% positive, "
        f"{percentages['NEUTRAL']}% neutral, {percentages['NEGATIVE']}% negative "
        f"(average sentiment {avg_score}/10). Most discussed themes: {top_theme_names}. "
        f"{spam_count} comment(s) flagged as likely spam."
    )
    summary_source = "stats"
    if GEMINI_API_KEY:
        summary = gemini_summary(comments, predictions, percentages, themes)
        summary_source = GEMINI_MODEL

    comments_out = [
        {
            "comment": c["text"],
            "author": c["author"],
            "timestamp": c["timestamp"],
            "sentiment": p,
            "label": LABELS[p],
            "spam": s,
        }
        for c, p, s in zip(comments, predictions, spam_flags)
    ]

    return {
        "videoId": video_id,
        "totalComments": total,
        "counts": counts,
        "percentages": percentages,
        "metrics": {
            "avgCommentLength": avg_word_count,
            "uniqueAuthors": unique_authors,
            "spamCount": spam_count,
            "avgSentimentScore": avg_score,
        },
        "summary": summary,
        "summarySource": summary_source,
        "themes": themes,
        "trend": trend,
        "comments": comments_out,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
