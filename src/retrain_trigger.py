"""
retrain_trigger.py
------------------
Evaluates the output of drift_monitor.py and applies the retraining
decision logic to produce a structured verdict.

Decision logic (consistent with dashboard and project methodology):

  PRIMARY TRIGGERS (automated, no labels needed):
    1. AUC-ROC < 0.5        → model has lost discriminative ability, not safe to deploy
    2. Feature fails both    → two-key KS + PSI evidence of distribution
       KS and PSI critical     collapse beyond safe operating bounds

  PERFORMANCE CONTEXT (requires labels, used for explanation only):
    - AUC-ROC, PR-AUC, Precision, Recall, F1
    - F1 is shown informally only — it conflates model quality with
      threshold miscalibration under changing fraud rates

  IMPORTANT LIMITATION:
    Silent distributed drift (e.g. drift_4 pattern) may pass both KS
    and PSI thresholds individually while causing catastrophic model
    failure. AUC-ROC monitoring is essential alongside distribution tests.

Usage (standalone):
    python src/retrain_trigger.py --batch data/drift_1.csv

Usage (imported):
    from src.drift_monitor import run_drift_monitor
    from src.retrain_trigger import evaluate_trigger

    monitor_report = run_drift_monitor(batch_path, artifact_dir)
    verdict = evaluate_trigger(monitor_report, batch_key, artifact_dir)
"""

import argparse
import json
import os
from src.drift_monitor import run_drift_monitor


# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "auc_random":       0.50,   # below this = model inverting predictions
    "auc_drop_warning": 0.10,   # AUC degraded but not inverted
}


# ── Decision logic ─────────────────────────────────────────────────────────────
def evaluate_trigger(monitor_report: dict,
                     batch_key: str,
                     artifact_dir: str = "artifacts/") -> dict:
    """
    Apply retraining decision logic to a drift monitor report.

    Parameters
    ----------
    monitor_report : output of run_drift_monitor()
    batch_key      : e.g. "drift_1" — used to look up production metrics
    artifact_dir   : directory containing production_results.json
                     and training_artifacts.json

    Returns
    -------
    dict with keys:
      'status'    : "critical" | "warning" | "stable"
      'title'     : human-readable verdict
      'reason'    : explanation of the decision
      'action'    : recommended next step
      'features'  : list of actionable features
      'metrics'   : dict of relevant performance metrics for context
    """
    # Load performance metrics if available
    prod_results_path = os.path.join(artifact_dir, "production_results.json")
    training_path     = os.path.join(artifact_dir, "training_artifacts.json")

    prod_metrics  = None
    baseline_auc  = None

    if os.path.exists(prod_results_path) and os.path.exists(training_path):
        with open(prod_results_path) as f:
            production = json.load(f)
        with open(training_path) as f:
            training = json.load(f)

        prod_metrics = production.get(batch_key)
        baseline_auc = training["baseline_metrics"]["AUC-ROC"]

    summary    = monitor_report["summary"]
    actionable = summary["actionable_features"]
    n_action   = summary["n_actionable"]

    # Performance context
    auc      = prod_metrics["AUC-ROC"] if prod_metrics else None
    pr_auc   = prod_metrics["PR-AUC"]  if prod_metrics else None
    f1       = prod_metrics["F1"]      if prod_metrics else None
    auc_drop = (auc - baseline_auc)    if (auc and baseline_auc) else None

    metrics_context = {
        "AUC-ROC":      auc,
        "PR-AUC":       pr_auc,
        "F1":           f1,
        "AUC-ROC drop": round(auc_drop, 4) if auc_drop else None,
        "fraud_rate":   monitor_report["fraud_rate"],
    }

    # ── Decision tree ──────────────────────────────────────────────────────────

    # Tier 1 — AUC below random: model is actively harmful
    if auc is not None and auc < THRESHOLDS["auc_random"]:
        return {
            "status":   "critical",
            "title":    "🚨 RETRAIN REQUIRED",
            "reason":   (
                f"AUC-ROC={auc:.4f} is near random. "
                f"The model has lost discriminative ability — its scores "
                f"no longer meaningfully distinguish fraud from legitimate transactions. "
                f"Actionable features: {actionable if actionable else 'none flagged by PSI/KS'}."
            ),
            "action":   (
                "Immediate full retrain on fresh labeled data. "
                "Investigate which features caused the inversion before retraining."
            ),
            "features": actionable,
            "metrics":  metrics_context,
        }

    # Tier 2 — Two-key distribution evidence
    if n_action >= 1:
        return {
            "status":   "critical",
            "title":    "🚨 RETRAIN REQUIRED",
            "reason":   (
                f"{n_action} feature(s) failed both KS (p<0.05) and PSI≥0.20: "
                f"{actionable}. Input distribution has shifted beyond safe "
                f"operating bounds. "
                + (f"AUC-ROC={auc:.4f}." if auc else "")
            ),
            "action":   (
                "Full retrain required. "
                "Investigate root cause of feature shift before retraining — "
                "a data pipeline issue may be causing the distribution change."
            ),
            "features": actionable,
            "metrics":  metrics_context,
        }

    # Tier 3 — AUC degraded but not inverted (warning, labels required)
    if auc is not None and auc_drop < -THRESHOLDS["auc_drop_warning"]:
        return {
            "status":   "warning",
            "title":    "⚠️  MONITOR CLOSELY",
            "reason":   (
                f"AUC-ROC dropped by {abs(auc_drop):.4f} to {auc:.4f}. "
                f"Model ranking ability is degrading. "
                f"No features failed the two-key drift test — drift may be "
                f"distributed across many features at moderate levels."
            ),
            "action":   (
                "Investigate feature distributions. "
                "If fraud rate has changed, recalibrate classification threshold "
                "before considering full retrain."
            ),
            "features": actionable,
            "metrics":  metrics_context,
        }

    # Tier 4 — Moderate distribution signals, model still functioning
    if summary["n_warning"] >= 3 or summary["n_ks_flagged"] > 20:
        return {
            "status":   "warning",
            "title":    "⚠️  MONITOR CLOSELY",
            "reason":   (
                f"{summary['n_warning']} features at PSI warning level. "
                f"{summary['n_ks_flagged']} features flagged by KS test. "
                f"No critical PSI threshold breached via two-key test. "
                + (f"AUC-ROC={auc:.4f} — model still functioning." if auc else "")
            ),
            "action":   (
                "Schedule drift review. Monitor AUC trend over next batches. "
                "No immediate action required."
            ),
            "features": actionable,
            "metrics":  metrics_context,
        }

    # Stable
    return {
        "status":   "stable",
        "title":    "✅  MODEL HEALTHY",
        "reason":   (
            "No features failed the two-key drift test. "
            + (f"AUC-ROC={auc:.4f} within acceptable bounds." if auc else
               "Distribution tests show no critical drift.")
        ),
        "action":   "No action required. Continue scheduled monitoring.",
        "features": [],
        "metrics":  metrics_context,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate retraining trigger for a production batch."
    )
    parser.add_argument("--batch",     required=True,
                        help="Path to production batch CSV")
    parser.add_argument("--key",       required=True,
                        help="Batch key for performance lookup e.g. drift_1")
    parser.add_argument("--artifacts", default="artifacts/",
                        help="Path to artifact directory")
    args = parser.parse_args()

    monitor_report = run_drift_monitor(
        batch_path=args.batch,
        artifact_dir=args.artifacts,
    )

    verdict = evaluate_trigger(
        monitor_report=monitor_report,
        batch_key=args.key,
        artifact_dir=args.artifacts,
    )

    print(f"\n=== RETRAINING VERDICT ===")
    print(f"{verdict['title']}")
    print(f"\nReason : {verdict['reason']}")
    print(f"Action : {verdict['action']}")
    if verdict["features"]:
        print(f"Features: {verdict['features']}")
    print(f"\nPerformance Context:")
    for k, v in verdict["metrics"].items():
        if v is not None:
            print(f"  {k:15s}: {v}")