# YouTube Comment Sentiment — How Everything Works (explained.md)

## 1. The big picture flow (simple words)

```
[ Reddit training data ]                [ YouTube comments (live, via API) ]
          |                                        |
   DVC pipeline (dvc repro)                        |
   1. data_ingestion      download raw data        |
   2. data_preprocessing  clean text + features    |
   3. feature_extraction  TF-IDF vectorizer        |
   4. model_building      train LightGBM           |
   5. model_evaluation    test + log to MLflow     |
   6. model_registration  register on Dagshub      |
          |                                        |
   Dagshub MLflow Model Registry                   |
   - sentiment_analysis-lgbm-model                 |
   - sentiment_analysis-vectorizer                 |
          |                                        |
   FastAPI backend (fastapi-backend/main.py)       |
   - loads BOTH registered models at startup  <----+
   - same preprocessing as training                |
   - POST /analyze  ------------------------------>|
          |                                        |
   Chrome extension (extension/)                   |
   - popup reads video ID from the open tab        |
   - calls /analyze, renders results               |
```

One line version: **DVC trains the model → Dagshub stores it → FastAPI loads it and
scores live YouTube comments → the extension shows the results.**

## 2. What happens when you click the extension icon (step by step)

1. `popup.js` reads the active tab URL and extracts the 11-char video ID.
2. It POSTs `{ "videoId": "..." }` to `http://127.0.0.1:8000/analyze`.
3. Backend calls the **YouTube Data API v3** (key from `.env`) and pulls up to 500
   top-level comments.
4. Each comment goes through the **exact same preprocessing** used in training
   (`src/data/data_preprocessing.py`): lowercase → remove stopwords (keeping
   not/but/however/no/yet) → strip symbols → lemmatize.
5. TF-IDF vectorizer (the **registered** one from Dagshub) converts text to numbers,
   3 extra features are appended, LightGBM predicts 0 = NEGATIVE, 1 = NEUTRAL, 2 = POSITIVE.
6. A second, separate model (spam classifier) flags spam comments.
7. Backend also computes: percentages, metrics, key themes, monthly trend, summary text.
8. Popup renders everything and lets you filter / export.

## 2b. Every click in the popup — what actually happens (the full flow map)

| What you click/do | API call? | Where the work happens |
|---|---|---|
| Extension icon on a YouTube video | `POST /analyze` | The ONE endpoint the UI ever calls. Everything below is rendered from its single response |
| Filter buttons (ALL / POSITIVE / NEUTRAL / NEGATIVE / SPAM) | none | Pure JavaScript — filters the `comments` array already in memory (`popup.js` → `filteredComments()`) |
| SHOW MORE | none | JS slices 25 more items from the same in-memory array |
| Hover on trend chart points | none | Plotly tooltip from the `trend` array in the response |
| EXPORT CSV button | none | `exportCsv()` builds a CSV string in the browser and triggers a download — no server involved |

So the whole app is: **1 click = 1 API call = 1 big JSON**, and every other interaction
is instant because it works on data the popup already has.

## 2c. API Reference (fastapi-backend/main.py)

### `GET /` — health check
Returns `{"message": "...", "model": "sentiment_analysis-lgbm-model"}`.
Use it to confirm the server is alive: open `http://127.0.0.1:8000/` in a browser.

### `POST /analyze` — the brain
Request: `{ "videoId": "dQw4w9WgXcQ" }`

What it does, in order:
1. Validates the video ID format (11 chars) → else **400**
2. Fetches up to 500 top-level comments from YouTube Data API → API failure = **502**, zero comments = **404**
3. `predict_batch()` — preprocess + TF-IDF + LightGBM → 0/1/2 per comment
4. `detect_spam()` — spam model → 0/1 per comment
5. `extract_themes()` — word frequencies; `compute_trend()` — monthly % per class; metrics; summary (Gemini if `GEMINI_API_KEY` set, else stats)

Response (one JSON, every field computed from real data):
```json
{
  "videoId": "...", "totalComments": 500,
  "counts": {"NEGATIVE": 50, "NEUTRAL": 334, "POSITIVE": 116},
  "percentages": {"NEGATIVE": 10.0, "NEUTRAL": 66.8, "POSITIVE": 23.2},
  "metrics": {"avgCommentLength": 8.6, "uniqueAuthors": 481,
               "spamCount": 43, "avgSentimentScore": 5.7},
  "summary": "...", "summarySource": "gemini-2.5-flash",
  "themes": [{"word": "rick", "count": 64}, ...],
  "trend": [{"month": "2026-09", "comments": 499,
              "NEGATIVE": 9.6, "NEUTRAL": 67.1, "POSITIVE": 23.2}, ...],
  "comments": [{"comment": "...", "author": "...", "timestamp": "...",
                 "sentiment": 2, "label": "POSITIVE", "spam": 0}, ...]
}
```

Startup (before any request): the server loads the 2 registered models from Dagshub
MLflow + the 2 local spam files. If `.env` keys are missing or nothing is registered,
it **crashes loudly at boot** with the exact reason — no silent fallbacks by design.

## 3. Feature checklist (from the requirements doc)

| Feature | Status | Where it lives | How it works (no hardcoding) |
|---|---|---|---|
| Sentiment classification (pos/neu/neg) | done | `fastapi-backend/main.py` → `predict_batch()` | Registered LightGBM model from Dagshub registry |
| Sentiment distribution visualization | done | `extension/popup.js` → `bar()` | % computed in backend from real predictions, drawn as bars |
| Detailed sentiment insights (drill-down) | done | popup "DETAILED INSIGHTS" filter buttons | ALL / POSITIVE / NEUTRAL / NEGATIVE / SPAM filters over the full 500-comment result, with SHOW MORE paging |
| Average comment length | done | backend `metrics.avgCommentLength` | Actual word count average of the fetched comments (also: unique authors, avg sentiment score /10) |
| Word cloud visualization | done | backend `extract_themes()` + popup `wordCloud()` | Word frequency over real comments (stopwords removed); font size scales with count |
| Highlight key themes | done | same `themes` list | Top 20 most frequent words = the themes |
| Summary of comments | done | backend `summary` + `gemini_summary()` | Two modes: with `GEMINI_API_KEY` in `.env` → real AI summary via Gemini (`gemini-2.5-flash`, override with `GEMINI_MODEL`); without the key → summary computed from real stats. Response field `summarySource` tells you which one ran |
| Trend tracking | done | backend `compute_trend()` + popup Plotly chart | Comments bucketed by month (from their timestamps); per-month % of NEGATIVE/NEUTRAL/POSITIVE drawn as a 3-line Plotly chart (`extension/plotly.min.js`, bundled locally because MV3 blocks CDN scripts) |
| Spam and troll detection | done | `train_spam_model.py` + `detect_spam()` | Real ML model: TF-IDF + Logistic Regression trained on the UCI SMS Spam Collection (5572 rows). **Spam F1 = 0.91, accuracy 98%.** Model files: `models/spam_model.joblib`, `models/spam_vectorizer.joblib` |
| Export data functionality | done | popup `exportCsv()` | Downloads `sentiment_<videoId>.csv` with every comment, author, timestamp, sentiment, label, spam flag |

Not done yet:
- Spam model is not in the DVC pipeline (it's a separate small model with its own
  training script) — could be added as DVC stages later if you want it versioned too.

## 4. File map (only the important ones)

```
comment-sentiment-analysis-r/
├── dvc.yaml                     pipeline definition (6 stages)
├── params.yaml                  training params
├── src/                         pipeline code (ingestion → registration)
├── models/                      joblib artifacts (lgbm, tfidf, spam_model, spam_vectorizer)
├── experiment_info.json         run_id + model URIs written by evaluation stage
├── .env                         DAGSHUB_PAT + YOUTUBE_API_KEY + GEMINI_API_KEY (NEVER commit)
├── .github/workflows/cicd.yaml  GitHub Actions pipeline (see section 7)
├── scripts/                     CI pytest gates: load / signature / performance
├── fastapi-backend/
│   ├── main.py                  the API server (loads registry models + spam model)
│   ├── train_spam_model.py      one-time spam model training (prints F1)
│   └── requirements.txt
└── extension/                   Chrome MV3 extension (Load unpacked / Web Store zip)
    ├── manifest.json
    ├── popup.html
    ├── popup.css                neo-brutalism styling
    ├── popup.js                 fetch + render + filters + trend chart + CSV export
    └── plotly.min.js            bundled charting library (~4.5 MB, needed for MV3 CSP)
```

## 5. How to run it

```powershell
# one-time: train the spam model (already done on this machine)
python fastapi-backend\train_spam_model.py

# every session: start the backend (keep the window open)
python fastapi-backend\main.py

# Chrome: chrome://extensions -> Developer mode -> Load unpacked -> pick extension/
# then open a YouTube video and click the icon
```

Prerequisites in `.env` (gitignored):
- `DAGSHUB_PAT` — to download the registered models from Dagshub MLflow
- `YOUTUBE_API_KEY` — YouTube Data API v3 key for fetching comments
- `GEMINI_API_KEY` — optional; enables the AI-written summary (free key at aistudio.google.com)

## 6. Things to know (honest limitations)

- The spam model was trained on **text messages (SMS)**, not YouTube comments, so it
  over-flags things like "sub to my channel". It's a real model with a real F1 score,
  but retraining on a YouTube-comment dataset would make it sharper.
- `127.0.0.1:8000` means the backend must run on the **same computer** as Chrome.
  For other people to use the store version, the backend must be deployed publicly
  (e.g. Render) and `API_URL` in `popup.js` line 1 changed to that URL.
- YouTube API free quota: 10,000 units/day; one analysis (500 comments) ≈ 5 units.
- Comments must be enabled on the video, and only **top-level** comments are analyzed
  (replies are not fetched).

## 7. CI/CD (GitHub Actions)

File: `.github/workflows/cicd.yaml`. Every `git push` to main/master triggers:

1. Fresh Ubuntu runner installs `requirements.txt` (pinned versions)
2. DVC remote credentials configured from the `DAGSHUB_PAT` repo secret (no AWS anywhere — Dagshub replaces the S3 bucket from the course version)
3. `dvc repro` — runs the full pipeline; unchanged stages are skipped, MLflow logs to Dagshub
4. `dvc push` — new data/model cache goes to Dagshub storage (teammates get it with `dvc pull`)
5. Spam model retrained (it's outside DVC)
6. Three pytest gates: `scripts/test_load_model.py` (artifacts load), `scripts/test_model_signature.py` (feature width + class contract), `scripts/test_model_performance.py` (test accuracy >= 0.80, current ~0.851)
7. `dvc.lock` + `experiment_info.json` committed back to GitHub **and** Dagshub

One-time setup in the GitHub repo:
- Settings → Secrets and variables → Actions → new secret `DAGSHUB_PAT`
- Settings → Actions → General → Workflow permissions → **Read and write**

Note: there is no "promote to production" step — our `model_registration` DVC stage
already registers the model in the Dagshub MLflow registry, and the backend always
loads the latest registered version. That IS the promotion mechanism.

### Why our YAML differs from the course/AWS version

| Course (AWS) step | Our version | Why |
|---|---|---|
| `AWS_ACCESS_KEY_ID/SECRET` env everywhere | single `DAGSHUB_PAT` secret | Dagshub replaces S3 + the registry host |
| `dvc push` to S3 bucket | `dvc push` to `dagshub.com/.../my_final_project.dvc` | same command, different remote (already set in `.dvc/config`) |
| `promote_model.py` (stage=production) | deleted | registration stage + "backend loads latest version" already does promotion |
| Docker build → push to ECR account `390844758498` | deleted | that's the instructor's AWS account — unusable; API hosting decision is Render (pending) |
| zip + S3 + CodeDeploy to EC2 | deleted | needs paid AWS resources that don't exist for us |
| python 3.10 | 3.13 | matches the machine the pipeline was verified on |
| `scripts/test_*.py` (from course repo) | written by us, all 6 tests pass locally | performance gate = accuracy >= 0.80 (current 0.851) |

### Still pending (not done, decision needed)
- **API deployment (the real "CD" for the backend)**: Render free tier + Dockerfile,
  then flip `API_URL` in `popup.js` line 1 to the public URL. Until then the extension
  only works on the computer running the backend.
