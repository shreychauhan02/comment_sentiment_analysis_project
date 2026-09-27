# YouTube Comment Sentiment Analysis

A Chrome Extension that analyzes the sentiment of YouTube comments in real-time — built as an end-to-end MLOps project, not just a model in a notebook.

---

### Why I Built This

Most ML projects stop at "here's my model's accuracy." I wanted to go further — to actually **operate** a model, not just train one: a real data pipeline, experiment tracking, versioning, a live API, and an interface someone can actually click.

So I picked a genuinely messy, real-world problem — YouTube comment sentiment — and built the entire pipeline around it, end to end: raw data → trained model → FastAPI backend → a working Chrome extension.

Along the way I used it to explore things YouTube Studio doesn't surface out of the box:
- A clear Positive / Neutral / Negative breakdown
- Key themes & a word cloud
- Sentiment trend over time
- Spam detection
- Exportable, labeled comment data

**Current state:** it runs locally (backend + extension on the same machine). Public deployment is the next step — see [explained.md](explained.md) for the honest limitations.

---

### Features

- Real-time sentiment analysis (Positive / Neutral / Negative)
- Spam detection
- Word cloud + key themes
- Sentiment trend over time
- Gemini-powered audience summary
- Detailed comment-level insights
- One-click CSV export

---

### Screenshots

#### Chrome Extension UI
![Extension UI](screenshots/p.png)

#### MLflow Experiments Overview
![MLflow Experiments](screenshots/p2.png)

#### Handling Class Imbalance
![Imbalance Handling](screenshots/p3.png)

#### Bag of Words vs TF-IDF
![BOW vs TF-IDF](screenshots/p4.png)

#### TF-IDF Trigram Experiments
![TF-IDF Trigram](screenshots/p5.png)

#### Multiple Algorithms Comparison
![ML Algorithms](screenshots/p6.png)

#### LightGBM Hyperparameter Tuning (Optuna)
![LightGBM Tuning](screenshots/p7.png)

---

### Tech Stack & Experiments

**Models Tested**
- LightGBM
- XGBoost
- Random Forest
- Logistic Regression
- Multinomial Naive Bayes
- SVM

**Feature Engineering**
- Bag of Words vs TF-IDF (unigram to trigram)

**Class Imbalance Techniques**
- SMOTE + ENN
- ADASYN
- Oversampling
- Undersampling
- Class Weights

**Hyperparameter Tuning**
- Optuna (50+ LightGBM trials)

**MLOps**
- DVC pipeline (6 stages: ingestion → preprocessing → feature extraction → training → evaluation → registration)
- MLflow experiment tracking
- Model registration on Dagshub
- CI/CD via GitHub Actions (automated retraining, testing, and registration on every push)
- FastAPI backend
- Chrome Extension (Manifest V3)

**Separate Spam Classifier**
- TF-IDF + Logistic Regression, trained on the UCI SMS Spam Collection
- F1 = 0.91, accuracy 98%
- Known limitation: trained on SMS text, not YouTube comments, so it over-flags things like "sub to my channel" — noted honestly rather than hidden

---

### How to Run It

```bash
# one-time: train the spam model
python fastapi-backend/train_spam_model.py

# every session: start the backend (keep the window open)
python fastapi-backend/main.py

# Chrome: chrome://extensions -> Developer mode -> Load unpacked -> select extension/
# then open a YouTube video and click the icon
```

Requires a `.env` file (gitignored) with:
- `DAGSHUB_PAT` — to pull the registered models from Dagshub MLflow
- `YOUTUBE_API_KEY` — YouTube Data API v3 key
- `GEMINI_API_KEY` — optional, enables the AI-written summary


---

### Project Structure

```bash
├── extension/              # Chrome Extension
├── fastapi-backend/        # FastAPI server
├── src/                    # DVC pipeline code
├── models/                 # Trained models
├── screenshots/            # Project screenshots
├── dvc.yaml
├── params.yaml
└── explained.md            # Detailed technical explanation
```
