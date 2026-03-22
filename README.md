# Web Server Defense Agent

An AI-powered anomaly detection and automated response system for web servers.

## Project Structure

```
web_defense_agent/
├── defense_agent.py              # Main LangGraph agent (entry point)
├── anomaly_detection_pipeline.py # Log parsing & feature engineering
├── anomaly_detection_model.py    # ML model training (IF + OC-SVM)
├── gnn_user_model.py             # Graph Neural Network (per-user nodes)
├── realtime_preprocessor.py      # Preprocessing pipeline for live data
├── realtime_predictor.py         # Weighted-vote prediction engine
├── models/                       # Pre-trained model files
│   ├── IsolationForest.pkl
│   ├── OneClassSVM.pkl
│   ├── best_model_IsolationForest.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_cols.json
│   ├── gnn_model.pt
│   └── gnn_stats.npz
├── data/
│   ├── final_log.csv                  # Processed feature dataset
│   ├── final_log_with_predictions.csv # Dataset with model predictions
│   └── anomaly_score_distributions.png
├── .env.example                  # Copy to .env and add your API key
└── requirements.txt
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create your `.env` file:
```bash
cp .env.example .env
```
Then edit `.env` and add your Mistral API key:
```
MISTRAL_API_KEY=your_key_here
```
Get a key at: https://console.mistral.ai/api-keys

## Run the Agent

```bash
python defense_agent.py
```

## Agent Tools

| # | Tool | Purpose |
|---|------|---------|
| 1 | log_getter | Fetch server telemetry |
| 2 | ml_model | Isolation Forest + One-Class SVM detection |
| 3 | llm_model | Mistral LLM for unknown anomalies |
| 4 | captcha_giver | Issue CAPTCHA to suspicious IP |
| 5 | login_remover | Revoke session for unauthorised access |
| 6 | temp_block | Block user login for 3 hours |
| 7 | ip_blocker | Permanent firewall-level IP block |
| 8 | status_check | Re-check if threat persists after mitigation |

## Agent Decision Flow

```
log_getter → ml_model
    ├── Normal → allow
    └── Abnormal → classify
            ├── brute_force  → captcha → status_check → temp_block → ip_blocker
            ├── unauth_access → login_remover
            ├── dos           → temp_block → status_check → ip_blocker
            └── unknown       → llm_model (Mistral) → action
```

## Retrain Models (optional)

If you have new log data, retrain from scratch:
```bash
# 1. Rebuild feature dataset
python anomaly_detection_pipeline.py

# 2. Retrain ML models
python anomaly_detection_model.py

# 3. Retrain GNN
python gnn_user_model.py
```
