"""
Anomaly Detection – Isolation Forest + One-Class SVM
Dataset: final_log.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH     = "final_log.csv"
MODEL_DIR     = "models"
CONTAMINATION = 0.05
RANDOM_STATE  = 42

NUMERIC_FEATURES = [
    "request_rate", "failed_login_attempts", "status_code",
    "request_interval", "network_traffic_volume",
    "session_duration", "data_transfer_size", "cpu_usage",
]
CATEGORICAL_FEATURES = ["request_method", "geographic_location"]

Path(MODEL_DIR).mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 – PREPROCESS
# ─────────────────────────────────────────────

def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    print(f"[DATA] {len(df):,} rows loaded")

    df.drop(columns=["source_ip"], errors="ignore", inplace=True)
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    df.drop(columns=norm_cols, errors="ignore", inplace=True)

    # Fill missing
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col].fillna("Unknown", inplace=True)

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    print(f"[DATA] Feature matrix: {X_scaled.shape}")
    return X_scaled, scaler, encoders, feature_cols

# ─────────────────────────────────────────────
# STEP 2 – TRAIN MODELS
# ─────────────────────────────────────────────

def train_isolation_forest(X):
    print("\n[MODEL] Training Isolation Forest ...")
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)
    scores = -model.score_samples(X)          # higher = more anomalous
    preds  = np.where(model.predict(X) == -1, 1, 0)
    n_anom = preds.sum()
    print(f"    Anomalies detected: {n_anom} ({n_anom/len(preds)*100:.2f}%)")
    return model, preds, scores


def train_ocsvm(X):
    print("\n[MODEL] Training One-Class SVM ...")
    # Sub-sample for speed (OC-SVM scales O(n²))
    n   = min(5000, len(X))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X), n, replace=False)
    model = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale")
    model.fit(X.iloc[idx])
    scores = -model.score_samples(X)
    preds  = np.where(model.predict(X) == -1, 1, 0)
    n_anom = preds.sum()
    print(f"    Anomalies detected: {n_anom} ({n_anom/len(preds)*100:.2f}%)")
    return model, preds, scores

# ─────────────────────────────────────────────
# STEP 3 – EVALUATE & COMPARE
# ─────────────────────────────────────────────

def evaluate_and_compare(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    summary = {}

    for ax, (name, (preds, scores)) in zip(axes, results.items()):
        n_anom = int(preds.sum())
        pct    = n_anom / len(preds) * 100
        thresh = np.percentile(scores, 100 * (1 - CONTAMINATION))
        summary[name] = {"anomalies": n_anom, "pct": round(pct, 2),
                         "score_mean": round(scores.mean(), 4),
                         "score_std":  round(scores.std(), 4)}

        ax.hist(scores, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(thresh, color="red", linestyle="--", linewidth=1.5,
                   label=f"threshold ({thresh:.3f})")
        ax.set_title(f"{name}\n{n_anom} anomalies ({pct:.1f}%)")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig("anomaly_score_distributions.png", dpi=120)
    plt.close()
    print("\n[EVAL] Score distributions saved → anomaly_score_distributions.png")

    print("\n[EVAL] Model Comparison:")
    print(f"  {'Model':<22} {'Anomalies':>10} {'Pct':>7} {'Score Mean':>12} {'Score Std':>10}")
    print("  " + "-" * 65)
    for name, s in summary.items():
        print(f"  {name:<22} {s['anomalies']:>10} {s['pct']:>6.2f}%"
              f" {s['score_mean']:>12.4f} {s['score_std']:>10.4f}")
    return summary


def select_best(summary: dict) -> str:
    target = CONTAMINATION * 100
    best   = min(summary, key=lambda k: abs(summary[k]["pct"] - target))
    print(f"\n[SELECT] Best model: {best}  "
          f"(anomaly% = {summary[best]['pct']}%, target ≈ {target}%)")
    return best

# ─────────────────────────────────────────────
# STEP 4 – SAVE ARTEFACTS
# ─────────────────────────────────────────────

def save_artefacts(scaler, encoders, feature_cols, best_model, best_name,
                   all_models: dict):
    joblib.dump(scaler,   f"{MODEL_DIR}/scaler.pkl")
    print(f"[SAVE] {MODEL_DIR}/scaler.pkl")

    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
    print(f"[SAVE] {MODEL_DIR}/encoders.pkl")

    with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"[SAVE] {MODEL_DIR}/feature_cols.json")

    # Save each model under its own name
    for name, (model, _, _) in all_models.items():
        path = f"{MODEL_DIR}/{name}.pkl"
        joblib.dump(model, path)
        print(f"[SAVE] {path}")

    # Also save best model explicitly
    best_path = f"{MODEL_DIR}/best_model_{best_name}.pkl"
    joblib.dump(best_model, best_path)
    print(f"[SAVE] {best_path}  ← best model")

# ─────────────────────────────────────────────
# STEP 5 – REAL-TIME INFERENCE
# ─────────────────────────────────────────────

class AnomalyDetector:
    """
    Real-time anomaly detector backed by the best saved model.

    Usage
    -----
    detector = AnomalyDetector.load()
    result   = detector.predict({
        "request_rate": 2, "failed_login_attempts": 0, "status_code": 200,
        "request_interval": 30.0, "network_traffic_volume": 5000,
        "session_duration": 120.0, "data_transfer_size": 5000,
        "cpu_usage": 20.0, "request_method": "GET",
        "geographic_location": "Unknown",
    })
    # {"anomaly": 0, "anomaly_score": 0.312, "probability": 0.577, "label": "NORMAL"}
    """

    def __init__(self, model, scaler, encoders, feature_cols):
        self.model        = model
        self.scaler       = scaler
        self.encoders     = encoders
        self.feature_cols = feature_cols

    @classmethod
    def load(cls, model_dir: str = MODEL_DIR):
        scaler       = joblib.load(f"{model_dir}/scaler.pkl")
        encoders     = joblib.load(f"{model_dir}/encoders.pkl")
        feature_cols = json.load(open(f"{model_dir}/feature_cols.json"))
        # Load best_model_<name>.pkl
        pkls = list(Path(model_dir).glob("best_model_*.pkl"))
        if not pkls:
            raise FileNotFoundError("No saved model found. Run training first.")
        model = joblib.load(pkls[0])
        print(f"[LOAD] Model loaded: {pkls[0].name}")
        return cls(model, scaler, encoders, feature_cols)

    def _preprocess(self, record: dict) -> np.ndarray:
        row = {}
        for col in self.feature_cols:
            val = record.get(col, 0)
            if col in self.encoders:
                le  = self.encoders[col]
                val = str(val)
                val = le.transform([val])[0] if val in le.classes_ \
                      else le.transform([le.classes_[0]])[0]
            row[col] = val
        return self.scaler.transform(pd.DataFrame([row], columns=self.feature_cols))

    def predict(self, record: dict) -> dict:
        X     = self._preprocess(record)
        score = float(-self.model.score_samples(X)[0])
        raw   = self.model.predict(X)[0]
        anom  = 1 if raw == -1 else 0
        prob  = round(float(1 / (1 + np.exp(-score))), 4)
        return {
            "anomaly":       anom,
            "anomaly_score": round(score, 6),
            "probability":   prob,
            "label":         "ANOMALY" if anom else "NORMAL",
        }

    def predict_batch(self, records: list) -> pd.DataFrame:
        return pd.DataFrame([self.predict(r) for r in records])

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n=== ANOMALY DETECTION: Isolation Forest + One-Class SVM ===\n")

    X, scaler, encoders, feature_cols = load_and_preprocess(DATA_PATH)

    # Train both models
    m_if,  p_if,  s_if  = train_isolation_forest(X)
    m_svm, p_svm, s_svm = train_ocsvm(X)

    all_models = {
        "IsolationForest": (m_if,  p_if,  s_if),
        "OneClassSVM":     (m_svm, p_svm, s_svm),
    }

    # Evaluate & compare
    summary   = evaluate_and_compare({k: (v[1], v[2]) for k, v in all_models.items()})
    best_name = select_best(summary)
    best_model = all_models[best_name][0]

    # Save
    save_artefacts(scaler, encoders, feature_cols, best_model, best_name, all_models)

    # Write predictions to CSV
    df_out = pd.read_csv(DATA_PATH)
    df_out["IF_anomaly"]   = p_if
    df_out["SVM_anomaly"]  = p_svm
    df_out["IF_score"]     = s_if
    df_out["SVM_score"]    = s_svm
    df_out["best_anomaly"] = all_models[best_name][1]
    df_out.to_csv("final_log_with_predictions.csv", index=False)
    print("\n[OUT] Predictions saved → final_log_with_predictions.csv")

    # Real-time demo
    print("\n=== REAL-TIME INFERENCE DEMO ===")
    detector = AnomalyDetector(best_model, scaler, encoders, feature_cols)

    demos = [
        ("Normal",    {"request_rate": 2,   "failed_login_attempts": 0,  "status_code": 200,
                       "request_interval": 30.0, "network_traffic_volume": 5000,
                       "session_duration": 120.0, "data_transfer_size": 5000,
                       "cpu_usage": 20.0, "request_method": "GET", "geographic_location": "Unknown"}),
        ("Suspicious",{"request_rate": 500, "failed_login_attempts": 80, "status_code": 403,
                       "request_interval": 0.05, "network_traffic_volume": 9_000_000,
                       "session_duration": 5.0,  "data_transfer_size": 9_000_000,
                       "cpu_usage": 95.0, "request_method": "POST", "geographic_location": "Unknown"}),
    ]
    for tag, rec in demos:
        res = detector.predict(rec)
        print(f"  [{tag:10s}] → {res}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
