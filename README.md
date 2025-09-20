# Atelier 2 — Modular Churn Pipeline (Short README)

This homework builds a modular ML pipeline to predict **near-term player activity (churn)** using a Steam-like games dataset (`gaming_100mb.csv`) that contains per-title attributes (genre, platform, price, ratings, tags) and engagement aggregates (playtime_forever, playtime_2weeks, reviews, players, recommendations, etc.). The target is defined as: `churn = 1 if playtime_2weeks > 0 else 0`.

## Files
- **main.py** — CLI with subcommands: `prepare`, `train`, `predict`, `evaluate`, `cv`.
- **model_pipeline.py** — Preprocessing + model:
  - Ordinal (label) encoding for categoricals/booleans.
  - Median imputation + scaling for numerics.
  - **Training is always undersampled** to 50/50 (balanced).
  - Optional `SelectKBest` feature selection.

## How it works (as required)
- **Preparation ≠ Training**
  - `prepare`: builds target, **drops leakage** (`playtime_2weeks`, `median_playtime_2weeks`, IDs), infers types, **builds preprocessor**, makes split.
  - `train`: reuses that preprocessor via a `DataBundle`, undersamples train, fits `preprocessor → [KBest]* → RandomForest`.
- Test set stays **natural (imbalanced)** for realistic evaluation.

## Quick commands
```bash
python main.py prepare --csv gaming_100mb.csv
python main.py train --csv gaming_100mb.csv --out churn_model.joblib
python main.py predict --csv new_games_predict.csv --model churn_model.joblib --out preds.csv
python main.py evaluate --csv gaming_100mb.csv --model churn_model.joblib
python main.py cv --csv gaming_100mb.csv --folds 5
