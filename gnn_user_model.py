"""
gnn_user_model.py
Graph Neural Network where each NODE = one user/IP.

Node features per user:
  0  request_rate          – requests per minute
  1  failed_login_attempts – total failed logins
  2  data_transfer_size    – bytes transferred
  3  session_duration      – session length (ms)
  4  cpu_usage             – server cpu at time of request
  5  status_code_norm      – normalised HTTP status
  6  is_accessing_resource – 1 if actively hitting protected endpoints
  7  lat                   – latitude  (from geo)
  8  lon                   – longitude (from geo)

Edges:
  Two users are connected if they share the same /24 subnet
  OR their geographic distance < GEO_EDGE_KM km.
  Edge weight = 1 / (distance + 1)

Model: 3-layer GraphSAGE → binary classifier (normal=0 / anomalous=1)
Training: uses pseudo-labels from Isolation Forest scores on node features.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import joblib
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GEO_EDGE_KM   = 500      # connect users within this geographic distance
SUBNET_EDGE   = True     # also connect users on same /24 subnet
HIDDEN_DIM    = 32
EPOCHS        = 80
LR            = 0.01
MODEL_PATH    = "models/gnn_model.pt"
SCALER_PATH   = "models/gnn_scaler.pkl"

# Protected endpoint prefixes
PROTECTED = ["/admin", "/login", "/api/user", "/dashboard", "/config"]

# ── Geo helpers ───────────────────────────────────────────────────────────────
# Approximate lat/lon per country code (fallback when GeoIP unavailable)
COUNTRY_COORDS = {
    "IN": (20.5937, 78.9629), "US": (37.0902, -95.7129),
    "CN": (35.8617, 104.1954), "RU": (61.5240, 105.3188),
    "DE": (51.1657, 10.4515), "GB": (55.3781, -3.4360),
    "BR": (14.2350, -51.9253), "Unknown": (0.0, 0.0),
}

def _geo_to_latlon(geo: str):
    """Extract lat/lon from 'City, State, CC' or country code."""
    parts = [p.strip() for p in geo.split(",")]
    cc    = parts[-1] if parts else "Unknown"
    return COUNTRY_COORDS.get(cc, (0.0, 0.0))

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def _subnet24(ip: str) -> str:
    parts = ip.split(".")
    return ".".join(parts[:3]) if len(parts) == 4 else ip

# ── Build graph from dataframe ────────────────────────────────────────────────

def build_graph(df: pd.DataFrame) -> Data:
    """
    df columns expected (subset of final_log.csv + source_ip + geographic_location):
      source_ip, request_rate, failed_login_attempts, data_transfer_size,
      session_duration, cpu_usage, status_code, endpoint, geographic_location
    """
    # Aggregate per IP (one node per unique IP)
    agg = df.groupby("source_ip").agg(
        request_rate          =("request_rate",           "mean"),
        failed_login_attempts =("failed_login_attempts",  "sum"),
        data_transfer_size    =("data_transfer_size",     "sum"),
        session_duration      =("session_duration",       "mean"),
        cpu_usage             =("cpu_usage",              "mean"),
        status_code           =("status_code",            "mean"),
        geographic_location   =("geographic_location",    "first"),
        endpoint              =("endpoint",               "last"),
    ).reset_index()

    # is_accessing_resource
    agg["is_accessing_resource"] = agg["endpoint"].apply(
        lambda e: 1.0 if any(str(e).startswith(p) for p in PROTECTED) else 0.0
    )

    # lat / lon
    agg[["lat","lon"]] = agg["geographic_location"].apply(
        lambda g: pd.Series(_geo_to_latlon(g))
    )

    # subnet
    agg["subnet24"] = agg["source_ip"].apply(_subnet24)

    # ── Node feature matrix ──────────────────────────────────────────────
    feat_cols = ["request_rate","failed_login_attempts","data_transfer_size",
                 "session_duration","cpu_usage","status_code",
                 "is_accessing_resource","lat","lon"]

    X = agg[feat_cols].values.astype(np.float32)

    # Min-max normalise
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom   = np.where(col_max - col_min == 0, 1, col_max - col_min)
    X_norm  = (X - col_min) / denom

    # ── Build edges (vectorised) ─────────────────────────────────────────
    n      = len(agg)
    lats   = agg["lat"].values
    lons   = agg["lon"].values
    subnets= agg["subnet24"].values

    src_list, dst_list, w_list = [], [], []

    # Subnet edges: group by subnet, connect all pairs within group
    from itertools import combinations
    subnet_groups = {}
    for idx, sn in enumerate(subnets):
        subnet_groups.setdefault(sn, []).append(idx)

    for sn, members in subnet_groups.items():
        if len(members) < 2:
            continue
        for i, j in combinations(members, 2):
            src_list += [i, j]; dst_list += [j, i]; w_list += [1.0, 1.0]

    # Geo edges: only if lat/lon are non-zero (i.e. known location)
    known_mask = (lats != 0.0) | (lons != 0.0)
    known_idx  = np.where(known_mask)[0]

    if len(known_idx) > 1:
        # Vectorised haversine between all known-location pairs
        lat_r = np.radians(lats[known_idx])
        lon_r = np.radians(lons[known_idx])
        for ii in range(len(known_idx)):
            dlat = lat_r[ii+1:] - lat_r[ii]
            dlon = lon_r[ii+1:] - lon_r[ii]
            a    = np.sin(dlat/2)**2 + \
                   np.cos(lat_r[ii]) * np.cos(lat_r[ii+1:]) * np.sin(dlon/2)**2
            dist = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            close = np.where(dist < GEO_EDGE_KM)[0]
            for jj in close:
                i = int(known_idx[ii])
                j = int(known_idx[ii + 1 + jj])
                w = float(1.0 / (dist[jj] + 1))
                src_list += [i, j]; dst_list += [j, i]; w_list += [w, w]

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(w_list, dtype=torch.float).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float)

    x = torch.tensor(X_norm, dtype=torch.float)

    # ── Pseudo-labels from Isolation Forest ─────────────────────────────
    try:
        if_model = joblib.load("models/IsolationForest.pkl")
        scaler   = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")

        feat_for_if = agg[["request_rate","failed_login_attempts","status_code",
                            "session_duration","data_transfer_size","cpu_usage"]].copy()

        # encode categoricals with defaults
        for col in ["request_method","geographic_location"]:
            le = encoders.get(col)
            if le:
                feat_for_if[col] = le.transform([le.classes_[0]] * len(feat_for_if))

        # align columns to scaler
        from realtime_preprocessor import FEATURE_COLS
        for c in FEATURE_COLS:
            if c not in feat_for_if.columns:
                feat_for_if[c] = 0.0
        feat_for_if = feat_for_if[FEATURE_COLS]

        X_scaled = scaler.transform(feat_for_if)
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
        preds    = if_model.predict(X_scaled_df)
        labels   = torch.tensor(
            np.where(preds == -1, 1, 0), dtype=torch.long
        )
    except Exception as e:
        print(f"[GNN] Pseudo-label fallback (all normal): {e}")
        labels = torch.zeros(n, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    data.ip_list = agg["source_ip"].tolist()
    data.feat_min = col_min
    data.feat_max = col_max
    return data


# ── GNN Model ─────────────────────────────────────────────────────────────────

class UserGraphSAGE(nn.Module):
    def __init__(self, in_dim: int = 9, hidden: int = HIDDEN_DIM, out_dim: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim,  hidden)
        self.conv2 = SAGEConv(hidden,  hidden)
        self.conv3 = SAGEConv(hidden,  out_dim)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.drop(x)
        x = self.conv3(x, edge_index)
        return x   # raw logits per node


# ── Training ──────────────────────────────────────────────────────────────────

def train_gnn(data: Data) -> UserGraphSAGE:
    model   = UserGraphSAGE(in_dim=data.x.shape[1])
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            preds = out.argmax(dim=1)
            acc   = (preds == data.y).float().mean().item()
            print(f"  [GNN] epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[GNN] Model saved → {MODEL_PATH}")
    return model


# ── Inference on a single new user record ─────────────────────────────────────

def predict_user_node(user_record: dict, model: UserGraphSAGE,
                      feat_min: np.ndarray, feat_max: np.ndarray) -> dict:
    """
    Score a single incoming user without rebuilding the full graph.
    Uses node features only (no edges) — fast path for real-time.
    """
    feat_order = ["request_rate","failed_login_attempts","data_transfer_size",
                  "session_duration","cpu_usage","status_code",
                  "is_accessing_resource","lat","lon"]

    geo  = user_record.get("geographic_location", "Unknown")
    lat, lon = _geo_to_latlon(geo)
    endpoint = user_record.get("endpoint", "")

    raw = np.array([[
        float(user_record.get("requests_per_minute",    0)),
        float(user_record.get("failed_logins", 0) +
              user_record.get("total_failed_logins", 0)),
        float(user_record.get("data_transfer_bytes",    0)),
        float(user_record.get("session_duration_ms",    0)),
        float(str(user_record.get("cpu_usage_percent",  0)).replace("%","")),
        float(user_record.get("status_code",          200)),
        1.0 if any(endpoint.startswith(p) for p in PROTECTED) else 0.0,
        lat, lon,
    ]], dtype=np.float32)

    denom = np.where(feat_max - feat_min == 0, 1, feat_max - feat_min)
    x_norm = torch.tensor((raw - feat_min) / denom, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        # Single node, no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        logits = model(x_norm, edge_index)
        probs  = F.softmax(logits, dim=1).squeeze()
        pred   = int(logits.argmax(dim=1).item())

    return {
        "gnn_anomaly":     pred,                          # 0=normal, 1=anomaly
        "gnn_confidence":  round(float(probs[pred]), 4),
        "gnn_label":       "Abnormal" if pred == 1 else "Normal",
    }


# ── Load saved GNN ────────────────────────────────────────────────────────────

def load_gnn(feat_min: np.ndarray = None,
             feat_max: np.ndarray = None) -> tuple:
    model = UserGraphSAGE()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    # Load normalisation stats saved alongside model
    stats = np.load("models/gnn_stats.npz")
    return model, stats["feat_min"], stats["feat_max"]


# ── Main: train from final_log.csv ────────────────────────────────────────────

if __name__ == "__main__":
    print("[GNN] Building graph from final_log.csv ...")
    df = pd.read_csv("final_log.csv")

    # Ensure required columns exist
    for col in ["endpoint", "geographic_location"]:
        if col not in df.columns:
            df[col] = "Unknown"

    data  = build_graph(df)
    n_nodes = data.x.shape[0]
    n_edges = data.edge_index.shape[1] // 2
    n_anom  = int(data.y.sum().item())
    print(f"[GNN] Nodes={n_nodes}  Edges={n_edges}  Anomalous={n_anom}")

    print("[GNN] Training GraphSAGE ...")
    model = train_gnn(data)

    # Save normalisation stats
    np.savez("models/gnn_stats.npz",
             feat_min=data.feat_min, feat_max=data.feat_max)
    print("[GNN] Stats saved → models/gnn_stats.npz")

    # Quick inference test
    sample = {
        "requests_per_minute": 8, "failed_logins": 0,
        "data_transfer_bytes": 420, "session_duration_ms": 12,
        "cpu_usage_percent": 4.5, "status_code": 200,
        "endpoint": "/users/stats", "geographic_location": "Chennai, TN, IN",
    }
    result = predict_user_node(sample, model, data.feat_min, data.feat_max)
    print(f"[GNN] Sample prediction: {result}")
