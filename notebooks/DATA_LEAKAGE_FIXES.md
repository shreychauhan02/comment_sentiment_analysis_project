# Data Leakage Audit & Fixes (notebooks/)

**Date:** 2026-09-25

## What is data leakage?

Leakage happens when information from the test set influences training. The model then looks artificially good in evaluation but will underperform on real, unseen data. In this project there were three recurring patterns:

1. **Vectorizer fitted before the split** — `vectorizer.fit_transform(df[...])` on the full dataset lets the test text contribute words and IDF weights to the training vocabulary. The model has already "seen" the test set's language.
2. **Resampling (SMOTE / ADASYN / SMOTEENN) before the split** — synthetic samples are generated from the whole dataset, so test-row neighbors leak into the training set.
3. **Hyperparameter tuning on the test set** — Optuna picked the "best" trial by test accuracy. That quietly turns the test set into a tuning set, so the final reported score is no longer an honest estimate.

**Correct order:** split raw text → fit vectorizer on **train only** → transform val/test → resample **train only** → tune on **val** → evaluate on **test once**.

## Fixes per notebook

| Old name | New name | Leaks found & fixed |
|---|---|---|
| `exp1_baseline.ipynb` | *(unchanged)* | `CountVectorizer` was fit on the full dataset before `train_test_split`. Now the split happens on raw text first, and the vectorizer is fit on train only. |
| `improving_baseline.ipynb` | `exp2_bow_vs_tfidf.ipynb` | Same vectorizer-before-split leak (BoW and TF-IDF). Fixed. Also fixed a broken data path (`pre.csv` → `../dataset/pre.csv`). |
| `improving(exp3)_baseline.ipynb` | `exp3_tfidf_max_features.ipynb` | Same vectorizer leak inside `run_experiment_tfidf_max_features`. Fixed. Broken data path fixed. |
| `exp4-handling-imbalance-data.ipynb` | `exp4_handling_imbalance.ipynb` | **(a)** Vectorizer fit on full data; **(b)** SMOTE/ADASYN/undersampling/SMOTEENN applied to full data *before* the split; **(c)** extra bug: the run name used `'class_weights'` but the code checked `'class_weight'`, so for that method **no train/test split was ever created** and the model was evaluated on its own training data. All fixed: split first → fit vectorizer on train → resample train only → `class_weights` now correctly maps to `class_weight='balanced'`. |
| `exp4_ml_algo_hyp_tuning.ipynb` | `exp5_ml_algo_hyp_tuning.ipynb` | **No leakage — already correct** (train/val/test split, vectorizer fit on train only, ADASYN on train only, Optuna tuned on val). Renamed only; kept as the reference pattern for the others. |
| `exp5_xgboost_with_hpt.ipynb` | `exp6_xgboost_hpt.ipynb` | **(a)** Vectorizer fit on full data; **(b)** SMOTE on full data before split; **(c)** Optuna objectives for XGBoost *and* LightGBM scored trials on `y_test`. Fixed: proper train/val/test split, vectorizer fit on train only, SMOTE on train only, Optuna now tunes on the **validation** set; test is used once in `log_mlflow`. Broken data path fixed. |
| `Untitled42.ipynb` | `exp7_lightgbm_hpt.ipynb` | Same three leaks as above (this is the LightGBM tuning notebook — `Untitled42` gave no clue). Fixed with the same pattern. Each Optuna trial no longer trains and logs a full model on test; trials train on train and score on val, and only the best model is evaluated on test. |

## Naming scheme

All notebooks now follow `exp<N>_<short-description>.ipynb` with underscores, ordered by experiment flow:

1. `exp1_baseline` — BoW + RandomForest baseline
2. `exp2_bow_vs_tfidf` — BoW vs TF-IDF n-gram comparison
3. `exp3_tfidf_max_features` — TF-IDF `max_features` sweep
4. `exp4_handling_imbalance` — resampling / class-weight comparison
5. `exp5_ml_algo_hyp_tuning` — algorithm comparison with Optuna
6. `exp6_xgboost_hpt` — XGBoost (+LightGBM) hyperparameter tuning
7. `exp7_lightgbm_hpt` — LightGBM tuning v2 (GPU)

## Adapted from a friend's notebook: `exp8_lightgbm_final.ipynb`

`exp_5_lightGBM_final.ipynb` (written for a YouTube dataset) was adapted to this project and renamed to `exp8_lightgbm_final.ipynb`:

- DagsHub switched to `shreychauhan02/my_final_project`.
- His data assumptions fixed: `comment` → `clean_comment`, `/content/preprocessed_data.csv` → `../dataset/pre.csv`, his `max_features=9410` (tuned on his dataset) → your `1000`.
- **Imbalance handling changed to your approach**: he relied on `class_weight='balanced'` inside LightGBM; this now uses **SMOTE on the training split only** (val/test untouched), matching exp6/exp7.
- Kept his good ideas: TF-IDF + word-count/char-count/avg-word-length extra features (computed from text only, so no label leakage) and multi-objective Optuna (accuracy + macro-F1) — all scored on **validation**; the test set is evaluated exactly once in the final run.
- His broken prediction demo (referenced undefined variables and forgot the extra features) was rewritten to work with the fitted vectorizer + final model, and now outputs readable `neutral / positive / negative` labels.
- Added the missing final MLflow run: params, test metrics, confusion matrix, and `log_model`.
- `DEVICE = 'cuda'` at the top — set it to `'cpu'` when not on a Colab GPU runtime.

## Not changed

- `eda_data.ipynb`, `analysis_for_special_character.ipynb`, `data_preprocessing.ipynb`, `mlflow/hello.ipynb`, `test.py` — no modeling leakage (EDA / preprocessing / logging smoke tests only).
- `improving_lightgbm.ipynb` — this file is **empty (0 bytes)** and cannot be opened as a notebook; it was likely meant to hold the LightGBM work that actually lives in `Untitled42.ipynb` (now `exp7_lightgbm_hpt.ipynb`). Safe to delete if you agree.

## Important caveat

All metrics previously logged to MLflow from the leaked notebooks (exp1–exp4, exp6, exp7) are **optimistically biased**. Re-run the fixed notebooks to get honest numbers — expect test accuracy to drop slightly. That drop is the leakage you were previously measuring, not a regression.
