"""
defense_agent.py  –  LangGraph Web Server Defense Agent
LLM: Mistral AI (mistral-large-latest)

8 Tools:
  1. log_getter          – fetch server telemetry
  2. ml_model            – IF + OC-SVM anomaly detection
  3. llm_model           – Mistral reasoning for unknown anomalies
  4. captcha_giver       – issue CAPTCHA to suspicious IP
  5. login_remover       – revoke session / access for unauthorised resource access
  6. temp_block          – block user login for 3 hours
  7. ip_blocker          – permanent firewall-level IP block
  8. status_check        – re-poll server to see if threat persists

Agent cycle:
  log_getter
      ↓
  ml_model
      ├─ Normal ──────────────────────────────────────────→ END (allow)
      └─ Abnormal
           ↓
       classify_attack
           ├─ brute_force  → captcha_giver → status_check
           │                     ↓ still attacking
           │                 temp_block → status_check
           │                     ↓ still attacking
           │                 ip_blocker → END
           │
           ├─ unauth_access → login_remover → END
           │
           ├─ dos           → temp_block → status_check
           │                     ↓ still attacking
           │                 ip_blocker → END
           │
           └─ unknown       → llm_model → (captcha / temp_block / ip_blocker / allow)
"""

import os, json, time
from typing import TypedDict
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()   # reads .env file
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
from mistralai.client.sdk import Mistral

from realtime_preprocessor import preprocess, load_artefacts
from realtime_predictor     import load_models, predict as ml_predict, compute_threshold

# ── Optional GNN ──────────────────────────────────────────────────────────────
try:
    from gnn_user_model import load_gnn, predict_user_node
    _gnn_model, _gnn_feat_min, _gnn_feat_max = load_gnn()
    GNN_AVAILABLE = True
    print("[AGENT] GNN loaded")
except Exception as e:
    GNN_AVAILABLE = False
    print(f"[AGENT] GNN skipped: {e}")

# ── Load ML artefacts once at startup ─────────────────────────────────────────
print("[AGENT] Loading ML models ...")
_scaler, _encoders = load_artefacts()
_models             = load_models()
_threshold          = compute_threshold(_models, _scaler, _encoders)

# ── Mistral client ────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
_mistral        = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL   = "mistral-large-latest"

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    raw_input:       dict    # server telemetry
    ml_result:       dict    # ml_model output
    gnn_result:      dict    # gnn output
    attack_type:     str     # brute_force | dos | unauth_access | unknown | normal
    llm_action:      str     # action recommended by LLM
    llm_reasoning:   str     # LLM explanation
    captcha_issued:  bool
    login_removed:   bool
    temp_blocked:    bool
    ip_blocked:      bool
    status_normal:   bool    # True = threat resolved
    escalation_step: int     # tracks how many escalation steps taken
    final_verdict:   str     # Normal | Abnormal
    final_action:    str
    explanation:     str

# ─────────────────────────────────────────────
# TOOL 1 – LOG GETTER
# ─────────────────────────────────────────────
def log_getter(state: AgentState) -> AgentState:
    """
    Fetches latest server telemetry.
    STUB: replace _fetch() with your real HTTP/API call.
    """
    def _fetch(ip: str) -> dict:
        # e.g. return requests.get(f"http://server/api/logs?ip={ip}").json()
        return state["raw_input"]

    ip   = state["raw_input"].get("source_ip", "unknown")
    data = _fetch(ip)
    state["raw_input"]   = data
    state["escalation_step"] = 0
    print(f"\n[TOOL 1 – log_getter]  IP={ip}")
    return state

# ─────────────────────────────────────────────
# TOOL 2 – ML MODEL
# ─────────────────────────────────────────────
def ml_model(state: AgentState) -> AgentState:
    """Runs Isolation Forest + One-Class SVM weighted vote."""
    try:
        result = ml_predict(
            state["raw_input"], _models, _scaler, _encoders, _threshold
        )
        state["ml_result"] = result

        # GNN as secondary signal
        if GNN_AVAILABLE:
            gnn = predict_user_node(
                state["raw_input"], _gnn_model, _gnn_feat_min, _gnn_feat_max
            )
            state["gnn_result"] = gnn
        else:
            state["gnn_result"] = {"gnn_label": "Unknown", "gnn_anomaly": -1}

        print(f"[TOOL 2 – ml_model]   ML={result['result']}  "
              f"vote={result['weighted_vote']}  "
              f"GNN={state['gnn_result'].get('gnn_label','—')}")
    except Exception as e:
        print(f"[TOOL 2 – ml_model]   ERROR: {e}")
        state["ml_result"]  = {"result": "Unknown", "weighted_vote": 0.0,
                                "model_votes": {}, "model_scores": {}}
        state["gnn_result"] = {"gnn_label": "Unknown", "gnn_anomaly": -1}
    return state

# ─────────────────────────────────────────────
# CLASSIFY (internal routing helper)
# ─────────────────────────────────────────────
def classify_attack(state: AgentState) -> AgentState:
    rec   = state["raw_input"]
    rpm   = float(rec.get("requests_per_minute", 0))
    fails = float(rec.get("failed_logins", 0)) + float(rec.get("total_failed_logins", 0))
    cpu   = float(str(rec.get("cpu_usage_percent", 0)).replace("%", ""))
    code  = int(rec.get("status_code", 200))
    ep    = str(rec.get("endpoint", ""))

    ml_abn  = state["ml_result"].get("result") == "Abnormal"
    gnn_abn = state["gnn_result"].get("gnn_anomaly", 0) == 1

    PROTECTED = ["/admin", "/config", "/api/user", "/dashboard", "/internal"]

    if not ml_abn and not gnn_abn:
        state["attack_type"] = "normal"
    elif fails > 20 or (code in (401, 403) and rpm > 30):
        state["attack_type"] = "brute_force"
    elif any(ep.startswith(p) for p in PROTECTED) and code in (401, 403):
        state["attack_type"] = "unauth_access"
    elif rpm > 300 or cpu > 85:
        state["attack_type"] = "dos"
    else:
        state["attack_type"] = "unknown"

    print(f"[CLASSIFY]            attack_type={state['attack_type']}")
    return state

# ─────────────────────────────────────────────
# TOOL 3 – LLM MODEL (Mistral)
# Called when ML cannot classify the anomaly
# ─────────────────────────────────────────────
def llm_model(state: AgentState) -> AgentState:
    """
    Uses Mistral to reason about unknown anomalies and decide action.
    Returns one of: captcha | login_remover | temp_block | ip_blocker | allow
    """
    rec = state["raw_input"]
    prompt = f"""You are a cybersecurity AI agent protecting a web server.

Analyse the server telemetry and ML results below.
Decide the best defensive action.

Server telemetry:
{json.dumps(rec, indent=2)}

ML model result: {json.dumps(state['ml_result'])}
GNN model result: {json.dumps(state['gnn_result'])}

Reply ONLY in this JSON format (no markdown):
{{
  "threat": true or false,
  "attack_type": "brute_force | dos | unauth_access | data_exfiltration | other | none",
  "action": "captcha | login_remover | temp_block | ip_blocker | allow",
  "reasoning": "one sentence explanation"
}}"""

    try:
        response = _mistral.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)

        state["llm_action"]    = parsed.get("action", "temp_block")
        state["llm_reasoning"] = parsed.get("reasoning", "")
        if parsed.get("threat"):
            state["attack_type"] = parsed.get("attack_type", "unknown")

        print(f"[TOOL 3 – llm_model]  action={state['llm_action']}  "
              f"type={state['attack_type']}")
        print(f"                      reason: {state['llm_reasoning']}")

    except Exception as e:
        state["llm_action"]    = "temp_block"
        state["llm_reasoning"] = f"LLM error – defaulting to temp_block: {e}"
        print(f"[TOOL 3 – llm_model]  fallback=temp_block ({e})")

    return state

# ─────────────────────────────────────────────
# TOOL 4 – CAPTCHA GIVER
# ─────────────────────────────────────────────
def captcha_giver(state: AgentState) -> AgentState:
    ip = state["raw_input"].get("source_ip", "unknown")
    # STUB: inject CAPTCHA via WAF / reverse-proxy API
    # e.g. requests.post("http://waf/captcha", json={"ip": ip})
    print(f"[TOOL 4 – captcha_giver]     CAPTCHA issued → {ip}")
    state["captcha_issued"] = True
    return state

# ─────────────────────────────────────────────
# TOOL 5 – LOGIN REMOVER
# Revokes session / access for unauthorised resource access
# ─────────────────────────────────────────────
def login_remover(state: AgentState) -> AgentState:
    ip   = state["raw_input"].get("source_ip", "unknown")
    user = state["raw_input"].get("username", "unknown")
    # STUB: invalidate session token / revoke access
    # e.g. requests.post("http://auth-service/revoke", json={"ip": ip, "user": user})
    print(f"[TOOL 5 – login_remover]     Session/access revoked → {ip} (user={user})")
    state["login_removed"] = True
    state["final_verdict"] = "Abnormal"
    state["final_action"]  = "login_removed"
    state["explanation"]   = (
        f"Unauthorised resource access detected from {ip}. "
        f"Session revoked."
    )
    return state

# ─────────────────────────────────────────────
# TOOL 6 – TEMPORARY BLOCK (3 hours)
# ─────────────────────────────────────────────
def temp_block(state: AgentState) -> AgentState:
    ip = state["raw_input"].get("source_ip", "unknown")
    # STUB: add IP to temporary deny list with 3h TTL
    # e.g. redis.setex(f"block:{ip}", 10800, "1")
    print(f"[TOOL 6 – temp_block]        {ip} blocked for 3 hours")
    state["temp_blocked"]    = True
    state["escalation_step"] = state.get("escalation_step", 0) + 1
    return state

# ─────────────────────────────────────────────
# TOOL 7 – IP BLOCKER (permanent)
# ─────────────────────────────────────────────
def ip_blocker(state: AgentState) -> AgentState:
    ip = state["raw_input"].get("source_ip", "unknown")
    # STUB: add permanent firewall rule
    # e.g. os.system(f"iptables -A INPUT -s {ip} -j DROP")
    print(f"[TOOL 7 – ip_blocker]        {ip} PERMANENTLY BLOCKED")
    state["ip_blocked"]    = True
    state["final_verdict"] = "Abnormal"
    state["final_action"]  = "ip_blocked"
    state["explanation"]   = (
        f"IP {ip} permanently blocked after repeated attacks "
        f"(type={state['attack_type']}) persisted through all mitigations."
    )
    return state

# ─────────────────────────────────────────────
# TOOL 8 – STATUS CHECK
# Re-polls server to see if threat is resolved
# ─────────────────────────────────────────────
def status_check(state: AgentState) -> AgentState:
    ip  = state["raw_input"].get("source_ip", "unknown")
    # STUB: re-fetch latest metrics for this IP and re-run quick check
    # In production: call log API + re-run ml_predict
    time.sleep(0.1)

    rec   = state["raw_input"]
    fails = float(rec.get("failed_logins", 0)) + float(rec.get("total_failed_logins", 0))
    rpm   = float(rec.get("requests_per_minute", 0))
    cpu   = float(str(rec.get("cpu_usage_percent", 0)).replace("%", ""))

    still_attacking = (fails > 20 or rpm > 300 or cpu > 85)
    state["status_normal"] = not still_attacking

    status = "RESOLVED ✓" if state["status_normal"] else "STILL ATTACKING ✗"
    print(f"[TOOL 8 – status_check]      {ip}  →  {status}")
    return state

# ─────────────────────────────────────────────
# FINAL RESPOND
# ─────────────────────────────────────────────
def respond(state: AgentState) -> AgentState:
    if not state.get("final_verdict"):
        if state.get("attack_type") in ("normal", ""):
            state["final_verdict"] = "Normal"
            state["final_action"]  = "allow"
            state["explanation"]   = "No anomaly detected. Traffic is normal."
        else:
            state["final_verdict"] = "Abnormal"
            state["final_action"]  = state.get("llm_action", "monitor")
            state["explanation"]   = (
                state.get("llm_reasoning") or
                f"Threat mitigated. Attack type: {state.get('attack_type')}. "
                f"CAPTCHA={state.get('captcha_issued', False)}, "
                f"TempBlock={state.get('temp_blocked', False)}."
            )

    print("\n" + "═" * 58)
    print(f"  VERDICT    : {state['final_verdict']}")
    print(f"  ACTION     : {state['final_action']}")
    print(f"  ATTACK     : {state.get('attack_type', '—')}")
    print(f"  CAPTCHA    : {state.get('captcha_issued', False)}")
    print(f"  LOGIN REM  : {state.get('login_removed', False)}")
    print(f"  TEMP BLOCK : {state.get('temp_blocked', False)}")
    print(f"  IP BLOCKED : {state.get('ip_blocked', False)}")
    print(f"  REASON     : {state['explanation']}")
    print("═" * 58 + "\n")
    return state

# ─────────────────────────────────────────────
# ROUTING FUNCTIONS
# ─────────────────────────────────────────────
def route_after_ml(state: AgentState) -> str:
    return "respond" if state["ml_result"].get("result") == "Normal" \
           else "classify_attack"

def route_after_classify(state: AgentState) -> str:
    t = state["attack_type"]
    if t == "normal":        return "respond"
    if t == "brute_force":   return "captcha_giver"
    if t == "unauth_access": return "login_remover"
    if t == "dos":           return "temp_block"
    return "llm_model"   # unknown

def route_after_captcha(state: AgentState) -> str:
    return "status_check"

def route_after_status(state: AgentState) -> str:
    """Escalation ladder: captcha → temp_block → ip_blocker"""
    if state["status_normal"]:
        return "respond"
    step = state.get("escalation_step", 0)
    if not state.get("captcha_issued"):
        return "captcha_giver"
    if not state.get("temp_blocked"):
        return "temp_block"
    return "ip_blocker"

def route_after_temp_block(state: AgentState) -> str:
    return "status_check"

def route_after_llm(state: AgentState) -> str:
    action = state.get("llm_action", "temp_block")
    routes = {
        "captcha":      "captcha_giver",
        "login_remover":"login_remover",
        "temp_block":   "temp_block",
        "ip_blocker":   "ip_blocker",
        "allow":        "respond",
    }
    return routes.get(action, "temp_block")

# ─────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)

    # Register all nodes
    for name, fn in [
        ("log_getter",      log_getter),
        ("ml_model",        ml_model),
        ("classify_attack", classify_attack),
        ("llm_model",       llm_model),
        ("captcha_giver",   captcha_giver),
        ("login_remover",   login_remover),
        ("temp_block",      temp_block),
        ("ip_blocker",      ip_blocker),
        ("status_check",    status_check),
        ("respond",         respond),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("log_getter")
    g.add_edge("log_getter", "ml_model")

    g.add_conditional_edges("ml_model", route_after_ml,
        {"respond": "respond", "classify_attack": "classify_attack"})

    g.add_conditional_edges("classify_attack", route_after_classify, {
        "respond":       "respond",
        "captcha_giver": "captcha_giver",
        "login_remover": "login_remover",
        "temp_block":    "temp_block",
        "llm_model":     "llm_model",
    })

    # LLM routes
    g.add_conditional_edges("llm_model", route_after_llm, {
        "captcha_giver": "captcha_giver",
        "login_remover": "login_remover",
        "temp_block":    "temp_block",
        "ip_blocker":    "ip_blocker",
        "respond":       "respond",
    })

    # Captcha → status check → escalate if needed
    g.add_edge("captcha_giver", "status_check")
    g.add_conditional_edges("status_check", route_after_status, {
        "respond":       "respond",
        "captcha_giver": "captcha_giver",
        "temp_block":    "temp_block",
        "ip_blocker":    "ip_blocker",
    })

    # Temp block → status check → escalate if needed
    g.add_conditional_edges("temp_block", route_after_temp_block,
        {"status_check": "status_check"})

    g.add_edge("login_remover", "respond")
    g.add_edge("ip_blocker",    "respond")
    g.add_edge("respond",       END)

    return g.compile()


agent = build_agent()
print("[AGENT] Graph compiled.\n")

# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def run_agent(server_json: dict) -> dict:
    initial: AgentState = {
        "raw_input":       server_json,
        "ml_result":       {},
        "gnn_result":      {},
        "attack_type":     "",
        "llm_action":      "",
        "llm_reasoning":   "",
        "captcha_issued":  False,
        "login_removed":   False,
        "temp_blocked":    False,
        "ip_blocked":      False,
        "status_normal":   False,
        "escalation_step": 0,
        "final_verdict":   "",
        "final_action":    "",
        "explanation":     "",
    }
    final = agent.invoke(initial)
    return {
        "verdict":     final["final_verdict"],
        "action":      final["final_action"],
        "attack_type": final["attack_type"],
        "explanation": final["explanation"],
        "ip_blocked":  final["ip_blocked"],
        "temp_blocked":final["temp_blocked"],
    }


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        {
            "label": "Normal user",
            "data": {
                "source_ip": "127.0.0.1", "requests_per_minute": 5,
                "failed_logins": 0, "request_method": "GET",
                "endpoint": "/dashboard", "status_code": 200,
                "session_duration_ms": 120, "data_transfer_bytes": 420,
                "request_interval_ms": 30, "cpu_usage_percent": 4.5,
                "total_failed_logins": 0, "geographic_location": "Chennai, TN, IN",
            }
        },
        {
            "label": "Brute force attack",
            "data": {
                "source_ip": "45.33.32.156", "requests_per_minute": 620,
                "failed_logins": 95, "request_method": "POST",
                "endpoint": "/login", "status_code": 401,
                "session_duration_ms": 3, "data_transfer_bytes": 8_500_000,
                "request_interval_ms": 0.08, "cpu_usage_percent": 97.0,
                "total_failed_logins": 95, "geographic_location": "Unknown",
            }
        },
        {
            "label": "Unauthorised resource access",
            "data": {
                "source_ip": "10.0.0.55", "requests_per_minute": 12,
                "failed_logins": 3, "request_method": "GET",
                "endpoint": "/admin/users", "status_code": 403,
                "session_duration_ms": 60, "data_transfer_bytes": 1200,
                "request_interval_ms": 5, "cpu_usage_percent": 20.0,
                "total_failed_logins": 3, "geographic_location": "Mumbai, MH, IN",
            }
        },
        {
            "label": "Unknown anomaly (LLM decides)",
            "data": {
                "source_ip": "203.0.113.42", "requests_per_minute": 45,
                "failed_logins": 2, "request_method": "GET",
                "endpoint": "/api/export", "status_code": 200,
                "session_duration_ms": 9000, "data_transfer_bytes": 4_200_000,
                "request_interval_ms": 1.3, "cpu_usage_percent": 55.0,
                "total_failed_logins": 2, "geographic_location": "Unknown",
            }
        },
    ]

    for c in cases:
        print(f"\n{'─'*58}")
        print(f">>> TEST: {c['label']}")
        run_agent(c["data"])
