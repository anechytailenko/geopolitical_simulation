"""FastAPI inference server (plans/01 §Python ML Service, plans/03 §5). Loads the trained
model + preprocess bundle + calibrator once at startup and serves predictions + explanations
for directed Country pairs. The Go API proxies to this service.

Run:  uvicorn app:app --host 127.0.0.1 --port 8000
Env:  GEO_DATA_DIR (parquet dir), GEO_ARTIFACTS_DIR (best.pt, preprocess.pkl, calibrator.pkl)
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.config import Config
from ml.infer import Predictor

app = FastAPI(title="geopolitic-ml", version="1.0")
_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(Config.from_env())
    return _predictor


@app.on_event("startup")
def _startup() -> None:
    try:
        get_predictor()
    except Exception as e:  # don't crash the server if artifacts aren't present yet
        print(f"[app] model not loaded at startup: {e}")


class PredictRequest(BaseModel):
    source_id: str
    target_id: str
    time_step: int
    explain: bool = False


class BatchRequest(BaseModel):
    pairs: list[tuple[str, str]]
    time_step: int


@app.get("/health")
def health() -> dict:
    loaded = _predictor is not None
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    pred = _safe(lambda: get_predictor().predict(req.source_id, req.target_id, req.time_step))
    out = {
        "source_id": pred.source_id, "target_id": pred.target_id, "time_step": pred.time_step,
        "probabilities": pred.probabilities, "predicted_class": pred.predicted_class,
        "confidence": pred.confidence,
    }
    if req.explain:
        out["explanation"] = _safe(
            lambda: get_predictor().explain(req.source_id, req.target_id, req.time_step))
    return out


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    preds = _safe(lambda: get_predictor().predict_batch(req.pairs, req.time_step))
    return {"predictions": [
        {"source_id": p.source_id, "target_id": p.target_id,
         "probabilities": p.probabilities, "predicted_class": p.predicted_class} for p in preds]}


def _safe(fn):
    try:
        return fn()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"model artifacts not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
