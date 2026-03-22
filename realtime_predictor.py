"""
realtime_predictor.py
Loads both trained models, runs weighted-vote prediction,
and returns  "Normal" or "Abnormal" with a confidence score.

Weights:
  Isolation Forest  → 0.6  (better generalisation, faster)
  One-Class SVM     → 0.4  (complements IF on dense clusters)
"""

import json
import numpy as np
import joblib
from pathlib import Path
from realtime_preprocessor import preprocess, load_artefacts

MODEL_DIR = "models"

# Model weights for voting (must sum to 1.0)
MODEL_WEIGHTS = {
    "IsolationForest": 0.6,
    "OneClassSVM":     0.4,
}

# Threshold derived from training data score percentile (top 5% = anomaly)
# Computed dynamically in load_threshold(); fallback hardcoded below
CONTAMINATION = 0.05


def load_models() -> dict:
    models = {}
    for name in MODEL_WEIGHTS:
        path = Path(MODEL_DIR) / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        models[name] = joblib.load(path)
        print(f"[LOAD] {path.name}")
    return models


def compute_threshold(models: dict, scaler, encoders) -> float:
    """
    With binary weighted voting the natural threshold is 0.5
    (i.e. weighted votes must exceed 50% to be Abnormal).
    We keep this function for transparency / future tuning.
    """
    threshold = 0.5
    print(f"[THRESHOLD] Using weighted-vote threshold: {threshold}")
    return threshold


def _anomaly_probability(model, X) -> tuple:
    """
    Returns (binary_flag, raw_score).
    binary_flag: 1 if model says anomaly, 0 if normal.
    raw_score: negated score_samples (higher = more anomalous).
    """
    raw_score    = float(-model.score_samples(X)[0])
    binary_flag  = 1 if model.predict(X)[0] == -1 else 0
    return binary_flag, raw_score


def predict(record: dict, models: dict, scaler, encoders,
            threshold: float = 0.5) -> dict:
    """
    Weighted voting:
      - Each model casts a binary vote (0=normal, 1=anomaly)
      - Votes are weighted by MODEL_WEIGHTS
      - weighted_vote >= threshold → Abnormal
      - raw scores reported for transparency
    """
    X = preprocess(record, scaler, encoders)

    model_scores  = {}
    model_votes   = {}
    weighted_vote = 0.0

    for name, model in models.items():
        flag, score       = _anomaly_probability(model, X)
        model_votes[name] = flag
        model_scores[name]= round(score, 4)
        weighted_vote    += MODEL_WEIGHTS[name] * flag

    weighted_vote = round(weighted_vote, 4)
    result        = "Abnormal" if weighted_vote >= threshold else "Normal"
    confidence    = weighted_vote if result == "Abnormal" \
                    else round(1 - weighted_vote, 4)

    return {
        "result":         result,
        "confidence":     confidence,
        "weighted_vote":  weighted_vote,
        "model_votes":    model_votes,
        "model_scores":   model_scores,
    }


# ── CLI / demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scaler, encoders = load_artefacts()
    models           = load_models()
    threshold        = compute_threshold(models, scaler, encoders)

    test_cases = [
        {
            "label": "Normal traffic",
            "data": {
                "source_ip": "127.0.0.1",
                "requests_per_minute": 8,
                "failed_logins": 0,
                "request_method": "GET",
                "endpoint": "/users/stats",
                "status_code": 200,
                "session_duration_ms": 12,
                "data_transfer_bytes": 420,
                "request_interval_ms": 12,
                "user_agent": "Mozilla/5.0...",
                "cpu_usage_percent": 4.5,
                "uptime": "00:12:34",
                "memory_usage": "62%",
                "disk_usage": "45%",
                "avg_response_time_ms": 15,
                "total_requests_last_hour": 42,
                "total_failed_logins": 3,
                "status_code_breakdown": {"200": 38, "401": 3, "404": 1},
                "http_method_breakdown": {"GET": 35, "POST": 7},
                "geographic_location": "Chennai, TN, IN",
            }
        },
        {
            "label": "Suspicious – brute force + high rate",
            "data": {
                "source_ip": "192.168.1.99",
                "requests_per_minute": 620,
                "failed_logins": 95,
                "request_method": "POST",
                "endpoint": "/login",
                "status_code": 401,
                "session_duration_ms": 3,
                "data_transfer_bytes": 8_500_000,
                "request_interval_ms": 0.08,
                "cpu_usage_percent": 97.0,
                "total_failed_logins": 95,
                "geographic_location": "Unknown",
            }
        },
    ]

    print("\n=== REAL-TIME ANOMALY DETECTION ===\n")
    for tc in test_cases:
        out = predict(tc["data"], models, scaler, encoders, threshold)
        print(f"  Input  : {tc['label']}")
        print(f"  Result : {out['result']}")
        print(f"  Confidence    : {out['confidence']}")
        print(f"  Weighted Vote : {out['weighted_vote']}")
        print(f"  IF vote  : {out['model_votes']['IsolationForest']}  score={out['model_scores']['IsolationForest']}")
        print(f"  SVM vote : {out['model_votes']['OneClassSVM']}  score={out['model_scores']['OneClassSVM']}")
        print()
