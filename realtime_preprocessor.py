"""
realtime_preprocessor.py
Transforms raw server telemetry (as received from the server)
into the feature vector expected by the trained models.
"""

import json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = "models"

# Mapping: server field → model feature name
FIELD_MAP = {
    "requests_per_minute":    "request_rate",
    "failed_logins":          "failed_login_attempts",
    "status_code":            "status_code",
    "request_interval_ms":    "request_interval",
    "data_transfer_bytes":    "network_traffic_volume",
    "session_duration_ms":    "session_duration",
    "data_transfer_bytes":    "data_transfer_size",
    "cpu_usage_percent":      "cpu_usage",
    "request_method":         "request_method",
    "geographic_location":    "geographic_location",
}

# Features the model expects (must match training order)
FEATURE_COLS = [
    "request_rate",
    "failed_login_attempts",
    "status_code",
    "request_interval",
    "network_traffic_volume",
    "session_duration",
    "data_transfer_size",
    "cpu_usage",
    "request_method",
    "geographic_location",
]


def _extract_features(record: dict) -> dict:
    """
    Pull and rename fields from raw server payload.
    Derives extra signals from nested breakdowns.
    """
    feat = {}

    feat["request_rate"]           = float(record.get("requests_per_minute", 0))
    feat["failed_login_attempts"]  = float(record.get("failed_logins", 0)
                                           + record.get("total_failed_logins", 0))
    feat["status_code"]            = float(record.get("status_code", 200))
    feat["request_interval"]       = float(record.get("request_interval_ms", 0))
    feat["network_traffic_volume"] = float(record.get("data_transfer_bytes", 0))
    feat["session_duration"]       = float(record.get("session_duration_ms", 0))
    feat["data_transfer_size"]     = float(record.get("data_transfer_bytes", 0))
    feat["cpu_usage"]              = float(str(record.get("cpu_usage_percent", 0))
                                           .replace("%", ""))
    feat["request_method"]         = str(record.get("request_method", "GET")).upper()

    # Normalise geographic_location → keep only country portion for consistency
    geo = str(record.get("geographic_location", "Unknown"))
    feat["geographic_location"]    = geo.split(",")[-1].strip() if "," in geo else geo

    return feat


def preprocess(record: dict, scaler, encoders: dict) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. Extract & rename fields
      2. Encode categoricals with saved LabelEncoders
      3. Scale with saved StandardScaler
    Returns a (1, n_features) numpy array ready for model.predict()
    """
    feat = _extract_features(record)

    # Encode categoricals
    for col in ["request_method", "geographic_location"]:
        le  = encoders.get(col)
        val = feat[col]
        if le is not None:
            val = val if val in le.classes_ else le.classes_[0]
            feat[col] = float(le.transform([val])[0])
        else:
            feat[col] = 0.0

    # Build ordered row
    row = pd.DataFrame([[feat[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

    # Scale — keep as DataFrame to preserve feature names for sklearn
    scaled = scaler.transform(row)
    return pd.DataFrame(scaled, columns=FEATURE_COLS)


def load_artefacts():
    scaler   = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
    return scaler, encoders


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "source_ip": "127.0.0.1",
        "requests_per_minute": 8,
        "failed_logins": 0,
        "request_method": "GET",
        "endpoint": "/users/stats",
        "status_code": 200,
        "session_duration_ms": 12,
        "data_transfer_bytes": 420,
        "request_interval_ms": 12,
        "cpu_usage_percent": 4.5,
        "total_failed_logins": 3,
        "geographic_location": "Chennai, TN, IN",
    }
    scaler, encoders = load_artefacts()
    X = preprocess(sample, scaler, encoders)
    print("Preprocessed vector shape:", X.shape)
    print("Values:", X)
