"""
Anomaly Detection Feature Engineering Pipeline
Supports: web-server-access-logs_10k.log, Apache.log, nasa_aug95_c.csv,
          ssh_login_attempts.csv, SSH.log, CIDDS-001-external-week1.csv
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TIME_WINDOW = "5min"          # grouping window for rate/volume features
SESSION_WINDOW = "30min"      # session boundary
MERGE_TOLERANCE = "5min"      # max gap for SSH ↔ web merge

# ─────────────────────────────────────────────
# STEP 1 – PARSE WEB LOGS
# ─────────────────────────────────────────────

# Combined Log Format regex – handles both:
#   standard:  "GET /path HTTP/1.1"
#   CSV-quoted: ""GET /path HTTP/1.1""
CLF_RE = re.compile(
    r'(?P<source_ip>\d+\.\d+\.\d+\.\d+)'
    r'.+?\[(?P<timestamp>[^\]]+)\]'
    r'\s+"{1,2}(?P<method>[A-Z]+)\s+(?P<endpoint>\S+)\s+HTTP/[\d.]+"{1,2}'
    r'\s+(?P<status_code>\d{3})'
    r'\s+(?P<response_size>\d+|-)'
)

MONTH_MAP = {
    "Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06",
    "Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"
}

def parse_clf_timestamp(ts: str) -> pd.Timestamp:
    """Parse '22/Jan/2019:03:56:14 +0330' → UTC-normalised Timestamp."""
    try:
        return pd.to_datetime(ts, format="%d/%b/%Y:%H:%M:%S %z", utc=True)
    except Exception:
        return pd.NaT


def parse_log_file(path: str) -> pd.DataFrame:
    """Parse any CLF-style log file (handles leading index column)."""
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = CLF_RE.search(line)
            if not m:
                continue
            records.append({
                "source_ip":     m.group("source_ip"),
                "timestamp":     parse_clf_timestamp(m.group("timestamp")),
                "request_method": m.group("method"),
                "endpoint":      m.group("endpoint"),
                "status_code":   int(m.group("status_code")),
                "response_size": 0 if m.group("response_size") == "-"
                                   else int(m.group("response_size")),
            })
    df = pd.DataFrame(records)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["source_ip","timestamp","request_method",
                                      "endpoint","status_code","response_size"])
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_nasa_csv(path: str) -> pd.DataFrame:
    """NASA log stored as CSV – each row is a raw log line."""
    rows = []
    try:
        raw = pd.read_csv(path, header=None, names=["raw"], dtype=str,
                          on_bad_lines="skip")
        for line in raw["raw"].dropna():
            m = CLF_RE.search(line)
            if m:
                rows.append({
                    "source_ip":     m.group("source_ip"),
                    "timestamp":     parse_clf_timestamp(m.group("timestamp")),
                    "request_method": m.group("method"),
                    "endpoint":      m.group("endpoint"),
                    "status_code":   int(m.group("status_code")),
                    "response_size": 0 if m.group("response_size") == "-"
                                       else int(m.group("response_size")),
                })
    except Exception as e:
        print(f"[WARN] nasa csv: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.dropna(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def load_web_logs() -> pd.DataFrame:
    """Load and concatenate all available web log sources."""
    frames = []
    sources = {
        "access_log":  "web-server-access-logs_10k.log",
        "apache":      "Apache.log",
        "nasa":        "nasa_aug95_c.csv",
    }
    for label, fname in sources.items():
        p = Path(fname)
        if not p.exists():
            print(f"[INFO] {fname} not found – skipping")
            continue
        if fname.endswith(".csv"):
            df = parse_nasa_csv(fname)
        else:
            df = parse_log_file(fname)
        if not df.empty:
            df["log_source"] = label
            frames.append(df)
            print(f"[OK]  {fname}: {len(df):,} records")

    if not frames:
        raise FileNotFoundError("No web log files could be parsed.")
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ─────────────────────────────────────────────
# STEP 2 – WEB LOG FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_web_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-IP, per-time-window features:
      request_rate, request_interval, session_duration,
      network_traffic_volume, data_transfer_size
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values(["source_ip", "timestamp"], inplace=True)

    # ── request_interval: seconds between consecutive requests per IP ──
    df["request_interval"] = (
        df.groupby("source_ip")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
    )

    # ── session_duration: within SESSION_WINDOW, last - first per IP ──
    df.set_index("timestamp", inplace=True)
    session_dur = (
        df.groupby("source_ip")
          .resample(SESSION_WINDOW, level="timestamp")["response_size"]
          .agg(lambda x: (x.index[-1] - x.index[0]).total_seconds()
               if len(x) > 1 else 0)
          .rename("session_duration")
          .reset_index()
    )

    # ── time-window aggregates ──
    def safe_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan

    grp = df.groupby(["source_ip", pd.Grouper(level="timestamp", freq=TIME_WINDOW)])
    window_agg = grp.agg(
        request_rate          =("response_size", "count"),
        network_traffic_volume=("response_size", "sum"),
        data_transfer_size    =("response_size", "sum"),
        status_code           =("status_code",   safe_mode),
        request_method        =("request_method", safe_mode),
        endpoint              =("endpoint",       "last"),
    ).reset_index()
    window_agg.rename(columns={"timestamp": "time_window"}, inplace=True)

    # ── merge session_duration into window_agg ──
    session_dur.rename(columns={"timestamp": "time_window"}, inplace=True)
    window_agg = pd.merge_asof(
        window_agg.sort_values("time_window"),
        session_dur.sort_values("time_window"),
        on="time_window",
        by="source_ip",
        direction="nearest",
        tolerance=pd.Timedelta(SESSION_WINDOW),
    )

    # ── avg request_interval per IP per window ──
    df.reset_index(inplace=True)
    interval_agg = (
        df.groupby(["source_ip",
                    pd.Grouper(key="timestamp", freq=TIME_WINDOW)])
          ["request_interval"]
          .mean()
          .reset_index()
          .rename(columns={"timestamp": "time_window",
                            "request_interval": "request_interval"})
    )
    window_agg = pd.merge_asof(
        window_agg.sort_values("time_window"),
        interval_agg.sort_values("time_window"),
        on="time_window",
        by="source_ip",
        direction="nearest",
        tolerance=pd.Timedelta(TIME_WINDOW),
    )

    window_agg.fillna({"session_duration": 0, "request_interval": 0},
                      inplace=True)
    return window_agg


# ─────────────────────────────────────────────
# STEP 3 – SSH LOG PROCESSING
# ─────────────────────────────────────────────

# Patterns for raw SSH.log (syslog style)
SSH_FAIL_RE  = re.compile(
    r'(?P<month>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d+:\d+:\d+)'
    r'.*?(?:Failed|Invalid|error).*?'
    r'(?:from\s+)?(?P<source_ip>\d+\.\d+\.\d+\.\d+)',
    re.IGNORECASE
)
SSH_OK_RE = re.compile(
    r'(?P<month>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d+:\d+:\d+)'
    r'.*?Accepted.*?from\s+(?P<source_ip>\d+\.\d+\.\d+\.\d+)',
    re.IGNORECASE
)

def _ssh_ts(month: str, day: str, time_str: str, year: int = 2024) -> pd.Timestamp:
    m = MONTH_MAP.get(month, "01")
    try:
        return pd.Timestamp(f"{year}-{m}-{int(day):02d} {time_str}", tz="UTC")
    except Exception:
        return pd.NaT


def parse_ssh_csv(path: str) -> pd.DataFrame:
    """Parse ssh_login_attempts.csv  (Month, DayOftheMonth, Time, Username, IPAddress, Port)."""
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    df["timestamp"] = df.apply(
        lambda r: _ssh_ts(r["Month"], r["DayOftheMonth"], r["Time"]), axis=1
    )
    # CSV has no explicit success/failure column → treat all as failed attempts
    # (brute-force datasets typically log only attempts)
    df["login_status"] = "failure"
    df.rename(columns={"IPAddress": "source_ip"}, inplace=True)
    return df[["source_ip", "timestamp", "login_status"]].dropna()


def parse_ssh_log(path: str) -> pd.DataFrame:
    """Parse raw SSH.log syslog file."""
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for pattern, status in [(SSH_FAIL_RE, "failure"),
                                     (SSH_OK_RE,   "success")]:
                m = pattern.search(line)
                if m:
                    ts = _ssh_ts(m.group("month"), m.group("day"),
                                 m.group("time"))
                    records.append({
                        "source_ip":    m.group("source_ip"),
                        "timestamp":    ts,
                        "login_status": status,
                    })
                    break
    return pd.DataFrame(records).dropna(subset=["timestamp"])


def parse_cidds(path: str) -> pd.DataFrame:
    """
    CIDDS-001 external week CSV.
    Use Src IP + timestamp; map class to login_status proxy.
    """
    try:
        df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        df["timestamp"] = pd.to_datetime(
            df["Date first seen"], errors="coerce", utc=True
        )
        df.rename(columns={"Src IP Addr": "source_ip"}, inplace=True)
        # Only keep rows with real IPs (skip OPENSTACK_NET / EXT_SERVER tokens)
        df = df[df["source_ip"].str.match(r"\d+\.\d+\.\d+\.\d+", na=False)]
        df["login_status"] = df["class"].apply(
            lambda c: "failure" if str(c).strip().lower() != "normal" else "success"
        )
        return df[["source_ip", "timestamp", "login_status"]].dropna()
    except Exception as e:
        print(f"[WARN] CIDDS: {e}")
        return pd.DataFrame(columns=["source_ip", "timestamp", "login_status"])


def load_ssh_logs() -> pd.DataFrame:
    """Load and concatenate all SSH / network-flow sources."""
    frames = []
    loaders = {
        "ssh_csv":  ("ssh_login_attempts.csv", parse_ssh_csv),
        "ssh_log":  ("SSH.log",                parse_ssh_log),
        "cidds":    ("CIDDS-001-external-week1.csv", parse_cidds),
    }
    for label, (fname, fn) in loaders.items():
        p = Path(fname)
        if not p.exists():
            print(f"[INFO] {fname} not found – skipping")
            continue
        df = fn(fname)
        if not df.empty:
            df["ssh_source"] = label
            frames.append(df)
            print(f"[OK]  {fname}: {len(df):,} records")

    if not frames:
        print("[WARN] No SSH logs found – failed_login_attempts will be 0")
        return pd.DataFrame(columns=["source_ip", "timestamp", "login_status"])
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def engineer_ssh_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate failed_login_attempts per IP per TIME_WINDOW."""
    if df.empty:
        return pd.DataFrame(columns=["source_ip", "time_window",
                                      "failed_login_attempts"])
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)

    agg = (
        df[df["login_status"] == "failure"]
          .groupby("source_ip")
          .resample(TIME_WINDOW)["login_status"]
          .count()
          .rename("failed_login_attempts")
          .reset_index()
          .rename(columns={"timestamp": "time_window"})
    )
    return agg


# ─────────────────────────────────────────────
# STEP 4 – MERGE WEB + SSH
# ─────────────────────────────────────────────

def merge_datasets(web_feat: pd.DataFrame,
                   ssh_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Merge on source_ip + nearest time_window within MERGE_TOLERANCE.
    Web features are the left (anchor) side.
    Falls back to IP-only merge when timestamps are incompatible.
    """
    if ssh_feat.empty:
        web_feat["failed_login_attempts"] = 0
        return web_feat

    web = web_feat.copy()
    ssh = ssh_feat.copy()

    # Align SSH timestamps to the same year range as web logs
    web_year = web["time_window"].dt.year.mode()[0]
    ssh_year = ssh["time_window"].dt.year.mode()[0]
    if web_year != ssh_year:
        ssh["time_window"] = ssh["time_window"] + pd.DateOffset(years=int(web_year - ssh_year))

    web = web.sort_values(["source_ip", "time_window"]).reset_index(drop=True)
    ssh = ssh.sort_values(["source_ip", "time_window"]).reset_index(drop=True)

    # Check for overlapping IPs; if none, do a plain IP-level join
    common_ips = set(web["source_ip"]) & set(ssh["source_ip"])
    if not common_ips:
        # Aggregate SSH failures per IP (ignore time window) and left-join on IP
        ssh_ip = ssh.groupby("source_ip")["failed_login_attempts"].sum().reset_index()
        merged = web.merge(ssh_ip, on="source_ip", how="left")
    else:
        merged = pd.merge_asof(
            web,
            ssh,
            on="time_window",
            by="source_ip",
            direction="nearest",
            tolerance=pd.Timedelta(MERGE_TOLERANCE),
        )

    merged["failed_login_attempts"] = (
        merged["failed_login_attempts"].fillna(0).astype(int)
    )
    return merged


# ─────────────────────────────────────────────
# STEP 5 – EXTERNAL FEATURES
# ─────────────────────────────────────────────

def add_geo_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    GeoIP lookup via geoip2 (MaxMind GeoLite2-City.mmdb).
    Falls back to 'Unknown' if library / DB not available.
    Install: pip install geoip2
    DB:      https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
    """
    try:
        import geoip2.database  # type: ignore
        import geoip2.errors    # type: ignore

        db_path = "GeoLite2-City.mmdb"
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found")

        reader = geoip2.database.Reader(db_path)
        cache: dict = {}

        def lookup(ip: str) -> str:
            if ip in cache:
                return cache[ip]
            try:
                r = reader.city(ip)
                loc = f"{r.country.name or 'Unknown'}/{r.city.name or 'Unknown'}"
            except Exception:
                loc = "Unknown"
            cache[ip] = loc
            return loc

        df["geographic_location"] = df["source_ip"].apply(lookup)
        reader.close()
        print("[OK]  GeoIP lookup complete")
    except ImportError:
        print("[WARN] geoip2 not installed – geographic_location = 'Unknown'")
        df["geographic_location"] = "Unknown"
    except FileNotFoundError as e:
        print(f"[WARN] {e} – geographic_location = 'Unknown'")
        df["geographic_location"] = "Unknown"
    return df


def add_cpu_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate cpu_usage as a function of request_rate (proxy for server load).
    Replace with real telemetry if available.
    """
    rng = np.random.default_rng(seed=42)
    base = np.clip(df["request_rate"] / df["request_rate"].max() * 80, 5, 95)
    noise = rng.normal(0, 5, size=len(df))
    df["cpu_usage"] = np.clip(base + noise, 0, 100).round(2)
    return df


# ─────────────────────────────────────────────
# STEP 6 – FINAL DATASET ASSEMBLY
# ─────────────────────────────────────────────

FINAL_COLUMNS = [
    "source_ip",
    "request_rate",
    "failed_login_attempts",
    "status_code",
    "request_interval",
    "network_traffic_volume",
    "session_duration",
    "data_transfer_size",
    "cpu_usage",
    "geographic_location",
    "request_method",
]

def assemble_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select, clean, and normalise the final feature set."""
    # Ensure all required columns exist
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FINAL_COLUMNS].copy()

    # ── Missing value strategy ──────────────────────────────────────────
    numeric_cols = [
        "request_rate", "failed_login_attempts", "request_interval",
        "network_traffic_volume", "session_duration",
        "data_transfer_size", "cpu_usage",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

    df["status_code"].fillna(0, inplace=True)
    df["request_method"].fillna("UNKNOWN", inplace=True)
    df["geographic_location"].fillna("Unknown", inplace=True)

    # ── Min-Max normalisation for numeric features ──────────────────────
    norm_cols = [
        "request_rate", "request_interval", "network_traffic_volume",
        "session_duration", "data_transfer_size", "cpu_usage",
        "failed_login_attempts",
    ]
    for col in norm_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# STEP 7 – MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(output_csv: str = "final_log.csv") -> pd.DataFrame:
    print("\n=== ANOMALY DETECTION PIPELINE ===\n")

    # 1. Load & parse web logs
    print("── Step 1: Loading web logs ──")
    web_raw = load_web_logs()
    print(f"    Total web records: {len(web_raw):,}\n")

    # 2. Engineer web features
    print("── Step 2: Engineering web features ──")
    web_feat = engineer_web_features(web_raw)
    print(f"    Web feature rows: {len(web_feat):,}\n")

    # 3. Load & process SSH logs
    print("── Step 3: Processing SSH logs ──")
    ssh_raw  = load_ssh_logs()
    ssh_feat = engineer_ssh_features(ssh_raw)
    print(f"    SSH feature rows: {len(ssh_feat):,}\n")

    # 4. Merge
    print("── Step 4: Merging datasets ──")
    merged = merge_datasets(web_feat, ssh_feat)
    print(f"    Merged rows: {len(merged):,}\n")

    # 5. External features
    print("── Step 5: Adding external features ──")
    merged = add_geo_location(merged)
    merged = add_cpu_usage(merged)

    # 6. Final assembly
    print("\n── Step 6: Assembling final dataset ──")
    final = assemble_final_dataset(merged)
    print(f"    Final rows:    {len(final):,}")
    print(f"    Final columns: {list(final.columns)}\n")

    # 7. Save
    final.to_csv(output_csv, index=False)
    print(f"── Step 7: Saved → {output_csv}\n")
    print(final.head(5).to_string())
    print("\n=== PIPELINE COMPLETE ===")
    return final


if __name__ == "__main__":
    df = run_pipeline()
