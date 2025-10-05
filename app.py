# app.py
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from model_pipeline import load_model, predict_dataframe  # tes helpers

# Config (aligné avec le Makefile)
MODEL_PATH = os.getenv("MODEL", "models/churn_model.joblib")
TARGET = os.getenv("TARGET", "churn")

# ----- Schemas -----
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[int]                  # on précise int
    probabilities: Optional[List[float]] = None

# ----- App & model load -----
app = FastAPI(title="Predict API", version="1.0")

try:
    MODEL = load_model(MODEL_PATH)   # Pipeline sklearn
    print(f"[app] Loaded model from {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    MODEL = None
    print(f"[app] Warning: could not load model at startup: {e}", file=sys.stderr)

@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/healthz"}

@app.get("/healthz")
def healthz():
    return {"model_loaded": MODEL is not None, "model_path": MODEL_PATH}

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    threshold: Optional[float] = Query(default=None, description="If set, threshold class=1 at this probability"),
    return_proba: bool = Query(default=False, description="If true and model supports proba, include probabilities"),
    target: str = Query(default=TARGET, description="Target name used at training"),
):
    try:
        if MODEL is None:
            raise RuntimeError("Model not loaded")
        df = pd.DataFrame(req.records)
        preds, proba = predict_dataframe(
            MODEL, df, target=target, threshold=threshold, return_proba=return_proba
        )
        # *** Cast vers types Python natifs pour éviter les 500 ***
        preds_py = [int(x) for x in (preds.tolist() if hasattr(preds, "tolist") else list(preds))]
        proba_py = None
        if proba is not None:
            proba_py = [float(x) for x in (proba.tolist() if hasattr(proba, "tolist") else list(proba))]
        return {"predictions": preds_py, "probabilities": proba_py}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
