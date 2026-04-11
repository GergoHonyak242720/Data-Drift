import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# src/ is one level up from dashboard/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.drift_monitor import run_drift_monitor
from src.retrain_trigger import evaluate_trigger

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection · Drift Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0e1117; color: #e0e0e0; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  .verdict-card {
    border-radius: 10px; padding: 20px 28px;
    margin-bottom: 18px; border-left: 6px solid;
  }
  .verdict-stable   { background:#0d2818; border-color:#2ea043; }
  .verdict-warning  { background:#2d1f00; border-color:#d29922; }
  .verdict-critical { background:#2d0c0c; border-color:#f85149; }

  .verdict-title  { font-size:1.5rem; font-weight:700; margin:0 0 6px 0; }
  .verdict-reason { font-size:0.92rem; color:#aaa; margin:0; }

  .section-header {
    font-size:0.75rem; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:#8b949e;
    border-bottom:1px solid #21262d;
    padding-bottom:6px; margin:28px 0 14px 0;
  }
  div[data-testid="metric-container"] {
    background:#161b22; border:1px solid #21262d;
    border-radius:8px; padding:14px 18px;
  }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
ART_DIR      = os.path.join(PROJECT_ROOT, "notebooks")

# ── Matplotlib dark style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#21262d", "axes.labelcolor": "#c9d1d9",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#c9d1d9", "grid.color":      "#21262d",
    "grid.linewidth":   0.6,       "legend.facecolor":"#161b22",
    "legend.edgecolor": "#21262d", "font.size":       10,
})

ACCENT  = "#58a6ff"
DANGER  = "#f85149"
WARNING = "#d29922"
SUCCESS = "#2ea043"
NEUTRAL = "#8b949e"

BATCH_LABELS = {
    "drift_1": "Batch 1 — Stable Fraud Rate",
    "drift_2": "Batch 2 — Low Fraud",
    "drift_3": "Batch 3 — Rising Fraud",
    "drift_4": "Batch 4 — Covariate Shift",
    "drift_5": "Batch 5 — Concept Drift",
}

DRIFT_TYPES = {
    "drift_1": "Localised covariate drift (V1 spike)",
    "drift_2": "Feature distribution shift (Amount spike)",
    "drift_3": "Label drift + threshold miscalibration",
    "drift_4": "Silent distributed covariate drift",
    "drift_5": "Label drift only — patterns stable",
}


# ── Load all artifacts ─────────────────────────────────────────────────────────
@st.cache_data
def load_artifacts():
    def jload(name):
        return json.load(open(os.path.join(ART_DIR, name)))

    training    = jload("training_artifacts.json")
    drift       = jload("drift_analysis.json")
    production  = jload("production_results.json")

    X_train = np.load(os.path.join(ART_DIR, "X_train.npy"))
    train_df = pd.DataFrame(X_train, columns=training["feature_names"])

    return training, drift, production, train_df


@st.cache_data
def load_batch_df(batch_key: str, feature_names: list) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{batch_key}.csv")
    df   = pd.read_csv(path)
    if "day" in df.columns:
        df = df.drop(columns=["day"])
    df["Amount"] = np.log(df["Amount"] + 0.001)
    return df


# ── Retraining decision (pure logic, no computation) ──────────────────────────



# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading artifacts…"):
    training, drift, production, train_df = load_artifacts()

feature_names    = training["feature_names"]
baseline_metrics = training["baseline_metrics"]
baseline_fraud   = training["baseline_fraud_rate"] * 100
v_features       = [f for f in feature_names if f.startswith("V")]

# Strip descriptive labels from drift keys to match production keys
# drift_analysis uses "drift_1 (stable)" — map to "drift_1"
drift_key_map = {k.split(" ")[0]: k for k in drift.keys()}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Drift Monitor")
    st.markdown("---")
    batch_key = st.selectbox(
        "Production Batch",
        list(BATCH_LABELS.keys()),
        format_func=lambda k: BATCH_LABELS[k]
    )
    st.markdown("---")
    st.markdown("**Monitoring Thresholds**")
    st.markdown("PSI Warning : `≥ 0.10`")
    st.markdown("PSI Critical : `≥ 0.20`")
    st.markdown("KS p-value : `< 0.05`")
    st.markdown("AUC-ROC Critical : `< 0.50`")
    st.markdown("AUC-ROC Warning : `drop > 0.10`")
    st.markdown("PR-AUC Critical : `drop > 0.20`")
    st.caption("F1 is shown for reference only. Decisions are driven by AUC-ROC, PR-AUC, and KS+PSI.")
    st.markdown("---")
    st.markdown(f"**Baseline fraud rate** `{baseline_fraud:.3f}%`")
    st.markdown(f"**Baseline F1** `{baseline_metrics['F1']:.4f}`")
    st.markdown(f"**Baseline AUC** `{baseline_metrics['AUC-ROC']:.4f}`")

# ── Resolve keys ───────────────────────────────────────────────────────────────
drift_label = drift_key_map[batch_key]      # e.g. "drift_1 (stable)"
perf        = production[batch_key]
drift_batch = drift[drift_label]

# Run drift monitor and trigger using src/ modules
@st.cache_data
def get_verdict(batch_key, batch_path, artifact_dir):
    monitor_report = run_drift_monitor(
        batch_path=batch_path,
        artifact_dir=artifact_dir,
    )
    verdict = evaluate_trigger(
        monitor_report=monitor_report,
        batch_key=batch_key,
        artifact_dir=artifact_dir,
    )
    return monitor_report, verdict

batch_path   = os.path.join(DATA_DIR, f"{batch_key}.csv")
monitor_report, decision = get_verdict(batch_key, batch_path, ART_DIR)

prod_fraud = perf["fraud_rate"]
f1_drop    = perf["F1"] - baseline_metrics["F1"]
batch_keys = list(BATCH_LABELS.keys())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Header + Verdict
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Fraud Detection · Drift Monitor")
st.markdown(
    f"**Evaluating:** {BATCH_LABELS[batch_key]} &nbsp;|&nbsp; "
    f"*Drift type: {DRIFT_TYPES[batch_key]}*"
)

feature_note = (
    f"<br><small>Actionable features: "
    f"<code>{', '.join(decision['features'])}</code></small>"
    if decision["features"] else ""
)

st.markdown(f"""
<div class="verdict-card verdict-{decision['status']}">
  <p class="verdict-title">{decision['title']}</p>
  <p class="verdict-reason">{decision['reason']}{feature_note}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Snapshot metrics
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Snapshot</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Fraud Rate",  f"{prod_fraud:.3f}%",
          f"{prod_fraud - baseline_fraud:+.3f}%", delta_color="inverse")
c2.metric("AUC-ROC",     f"{perf['AUC-ROC']:.4f}",
          f"{perf['AUC-ROC'] - baseline_metrics['AUC-ROC']:+.4f}")
c3.metric("PR-AUC",      f"{perf['PR-AUC']:.4f}",
          f"{perf['PR-AUC'] - baseline_metrics['PR-AUC']:+.4f}")
c4.metric("Recall",      f"{perf['Recall']:.4f}",
          f"{perf['Recall'] - baseline_metrics['Recall']:+.4f}")
c5.metric("Precision",   f"{perf['Precision']:.4f}",
          f"{perf['Precision'] - baseline_metrics['Precision']:+.4f}")
c6.metric("F1 Score",    f"{perf['F1']:.4f}",
          f"{perf['F1'] - baseline_metrics['F1']:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Drift Evidence
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Drift Evidence</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.4, 1])

with col_left:
    psi_series = pd.Series(drift_batch["psi"]).sort_values(ascending=False).head(15)
    bar_colors = [DANGER if v >= 0.20 else WARNING if v >= 0.10 else SUCCESS
                  for v in psi_series.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(psi_series.index[::-1], psi_series.values[::-1],
            color=bar_colors[::-1], edgecolor="#0e1117", linewidth=0.5)
    ax.axvline(0.10, color=WARNING, linestyle="--", linewidth=1.2, label="Warning (0.10)")
    ax.axvline(0.20, color=DANGER,  linestyle="--", linewidth=1.2, label="Critical (0.20)")
    ax.set_xlabel("PSI Value")
    ax.set_title("Top 15 Features by PSI", fontweight="bold", color="#e0e0e0")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close()

with col_right:
    ev         = drift_batch.get("evidently", {})
    n_crit_psi = sum(1 for v in drift_batch["psi"].values() if v >= 0.20)
    n_warn_psi = sum(1 for v in drift_batch["psi"].values() if 0.10 <= v < 0.20)
    n_ks       = sum(1 for v in drift_batch["ks_drifted"].values() if v)
    n_action   = len(decision["features"])

    st.markdown("**Your Analysis**")
    st.markdown(f"""
| Test | Count |
|---|---|
| PSI Critical (≥0.20) | `{n_crit_psi}` |
| PSI Warning (≥0.10) | `{n_warn_psi}` |
| KS Flagged (p<0.05) | `{n_ks}` |
| Actionable (both) | `{n_action}` |
""")

    if ev:
        st.markdown("**Evidently AI (independent)**")
        st.markdown(f"""
| | |
|---|---|
| Drifted columns | `{ev.get('n_drifted','—')}/{ev.get('n_columns','—')}` |
| Share drifted | `{ev.get('share_drifted','—')}` |
| Dataset drift | `{'Yes' if ev.get('dataset_drift') else 'No'}` |
""")
        st.caption("Evidently uses Jensen-Shannon divergence independently of PSI methodology.")

    worst_feat = max(drift_batch["psi"], key=drift_batch["psi"].get)
    worst_psi  = drift_batch["psi"][worst_feat]
    worst_ks   = drift_batch["ks_stat"][worst_feat]
    st.markdown("**Most Drifted Feature**")
    st.markdown(f"""
| | |
|---|---|
| Feature | `{worst_feat}` |
| PSI | `{worst_psi:.4f}` |
| KS Stat | `{worst_ks:.4f}` |
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Performance Impact
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Performance Impact</p>', unsafe_allow_html=True)

col_perf, col_cm = st.columns([1.2, 1])

with col_perf:
    all_f1     = [production[b]["F1"]      for b in batch_keys]
    all_auc    = [production[b]["AUC-ROC"] for b in batch_keys]
    all_prauc  = [production[b]["PR-AUC"]  for b in batch_keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(baseline_metrics["AUC-ROC"], color=NEUTRAL, linestyle="--",
               linewidth=1.4, label=f"Baseline AUC ({baseline_metrics['AUC-ROC']:.3f})")
    ax.axhline(0.5, color=DANGER, linestyle=":", linewidth=1,
               alpha=0.6, label="Random classifier (AUC=0.5)")
    ax.plot(batch_keys, all_auc,   marker="o", color=ACCENT,
            linewidth=2.2, markersize=7, label="AUC-ROC")
    ax.plot(batch_keys, all_prauc, marker="s", color=WARNING,
            linewidth=2.2, markersize=7, label="PR-AUC", linestyle="--")
    ax.plot(batch_keys, all_f1,    marker="^", color=NEUTRAL,
            linewidth=1.5, markersize=6, label="F1 (informational)",
            linestyle=":", alpha=0.7)

    idx = batch_keys.index(batch_key)
    ax.axvline(idx, color="#ffffff", linewidth=0.8, alpha=0.25)
    ax.scatter([idx], [all_auc[idx]],   color=ACCENT,  s=120, zorder=5)
    ax.scatter([idx], [all_prauc[idx]], color=WARNING, s=120, zorder=5)

    ax.fill_between(range(len(batch_keys)), all_auc, baseline_metrics["AUC-ROC"],
                    where=[v < baseline_metrics["AUC-ROC"] for v in all_auc],
                    alpha=0.10, color=DANGER)

    ax.set_xticks(range(len(batch_keys)))
    ax.set_xticklabels(batch_keys, rotation=15)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("AUC-ROC & PR-AUC Across All Production Batches",
                 fontweight="bold", color="#e0e0e0")
    ax.legend(fontsize=9)
    st.pyplot(fig)
    plt.close()

with col_cm:
    cm_prod = np.array(perf["confusion_matrix"])
    cm_base = np.array(production.get("baseline", {}).get(
        "confusion_matrix",
        [[56858, 4], [9, 89]]   # fallback — replace with actual if saved
    ))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for cm, ax, title, cmap in [
        (cm_base, axes[0], "Baseline", "Blues"),
        (cm_prod, axes[1], BATCH_LABELS[batch_key].split("—")[0].strip(), "Reds")
    ]:
        ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Legit", "Fraud"]).plot(
            ax=ax, cmap=cmap, colorbar=False)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(False)

    plt.suptitle("Confusion Matrix Comparison", fontsize=10,
                 fontweight="bold", color="#e0e0e0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Feature Inspector
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Feature Inspector</p>', unsafe_allow_html=True)

col_sel, col_kde = st.columns([1, 2.5])

with col_sel:
    sorted_feats  = sorted(drift_batch["psi"], key=drift_batch["psi"].get, reverse=True)
    selected_feat = st.selectbox("Select feature to inspect", sorted_feats)

    psi_val = drift_batch["psi"][selected_feat]
    ks_val  = drift_batch["ks_stat"][selected_feat]
    ks_p    = drift_batch["ks_pval"][selected_feat]
    drifted = drift_batch["ks_drifted"][selected_feat]
    level   = "critical" if psi_val >= 0.20 else "warning" if psi_val >= 0.10 else "stable"

    st.markdown(f"""
| Metric | Value |
|---|---|
| PSI | `{psi_val:.4f}` |
| PSI Level | `{level}` |
| KS Statistic | `{ks_val:.4f}` |
| KS p-value | `{ks_p:.6f}` |
| KS Drifted | `{'Yes' if drifted else 'No'}` |
""")

with col_kde:
    prod_df       = load_batch_df(batch_key, feature_names)
    fig, ax       = plt.subplots(figsize=(9, 4))
    sns.kdeplot(train_df[selected_feat], ax=ax,
                label="Training (reference)", color=ACCENT,
                fill=True, alpha=0.25, linewidth=2)
    sns.kdeplot(prod_df[selected_feat], ax=ax,
                label=BATCH_LABELS[batch_key], color=DANGER,
                fill=True, alpha=0.2, linewidth=2, linestyle="--")
    ax.set_title(
        f"Distribution Shift — {selected_feat}   "
        f"PSI={psi_val:.4f} | KS={ks_val:.4f}",
        fontweight="bold", color="#e0e0e0"
    )
    ax.legend(fontsize=9)
    ax.set_xlabel(selected_feat)
    st.pyplot(fig)
    plt.close()