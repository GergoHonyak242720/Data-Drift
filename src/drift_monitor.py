"""
drift_monitor.py
----------------
Computes statistical drift between a reference dataset and a new
production batch. Uses KS test and PSI with combined-range binning.

Methodology consistent with notebook 03_drift_detection.ipynb:
  - Reference: X_train.npy (80% training split, not full dataset)
  - PSI bins: combined range of reference + production
  - Epsilon: conditional, applied only to empty bins
  - Time excluded: not meaningful across deployment windows

Usage (standalone):
    python src/drift_monitor.py --batch data/drift_1.csv

Usage (imported):
    from src.drift_monitor import run_drift_monitor
    report = run_drift_monitor(batch_path="data/drift_1.csv",
                               artifact_dir="artifacts/")
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ── Constants ──────────────────────────────────────────────────────────────────
PSI_WARNING  = 0.10
PSI_CRITICAL = 0.20
KS_ALPHA     = 0.05


# ── Core statistical functions ─────────────────────────────────────────────────
def compute_psi(reference: np.ndarray,
                production: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Population Stability Index with combined-range bin edges.

    Bins are defined over the combined min/max of both distributions
    so that out-of-range production values are captured rather than
    clipped into edge bins.

    Epsilon is applied only to empty bins to avoid log(0) while
    minimising distortion of non-empty bins.

    Interpretation:
      < 0.10  → stable
      0.10 – 0.20 → moderate, investigate
      > 0.20  → critical, consider retraining
    """
    breakpoints = np.linspace(
        min(reference.min(), production.min()),
        max(reference.max(), production.max()),
        n_bins + 1
    )
    ref_counts  = np.histogram(reference,  bins=breakpoints)[0]
    prod_counts = np.histogram(production, bins=breakpoints)[0]

    eps      = 1e-4
    ref_pct  = np.where(ref_counts  == 0, eps, ref_counts  / len(reference))
    prod_pct = np.where(prod_counts == 0, eps, prod_counts / len(production))

    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def compute_ks(reference: np.ndarray,
               production: np.ndarray) -> tuple:
    """
    Two-sample Kolmogorov-Smirnov test.
    Returns (ks_stat, p_value).
    """
    return ks_2samp(reference, production)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_batch(path: str, feature_names: list) -> tuple:
    """
    Load and preprocess a production batch CSV.
    Applies identical preprocessing to the baseline notebook:
      - Drop 'day' column if present
      - Log-transform Amount
      - Reorder columns to match training schema
    Returns (df, X, y).
    """
    df = pd.read_csv(path)

    if "day" in df.columns:
        df = df.drop(columns=["day"])

    df["Amount"] = np.log(df["Amount"] + 0.001)

    y = df["Class"].values.astype(int) if "Class" in df.columns else None

    # Reorder to match training schema — guards against drift_5 column shuffle
    available = [f for f in feature_names if f in df.columns]
    X = df[available].values

    return df, X, y


# ── Main monitor function ──────────────────────────────────────────────────────
def run_drift_monitor(batch_path: str,
                      artifact_dir: str = "artifacts/",
                      n_bins: int = 10) -> dict:
    """
    Run drift detection for a new production batch against the
    training reference.

    Parameters
    ----------
    batch_path   : path to production batch CSV
    artifact_dir : directory containing training_artifacts.json
                   and X_train.npy
    n_bins       : number of PSI histogram bins

    Returns
    -------
    dict with keys:
      'batch_path'     : str
      'n_features'     : int
      'features'       : dict — per-feature PSI, KS stat, p-value,
                                ks_drifted, psi_level
      'summary'        : dict — counts of stable/warning/critical features,
                                KS flagged, actionable (two-key)
      'fraud_rate'     : float or None
    """
    # Load reference artifacts
    artifact_path = os.path.join(artifact_dir, "training_artifacts.json")
    with open(artifact_path) as f:
        training = json.load(f)

    feature_names = training["feature_names"]

    X_train = np.load(os.path.join(artifact_dir, "X_train.npy"))
    train_df = pd.DataFrame(X_train, columns=feature_names)

    # Features to monitor — V-features + Amount, Time excluded
    monitor_features = [f for f in feature_names
                        if f.startswith("V") or f == "Amount"]

    # Load and preprocess batch
    prod_df, X_prod, y_prod = preprocess_batch(batch_path, feature_names)

    # Run tests per feature
    feature_results = {}
    for feat in monitor_features:
        ref  = train_df[feat].values
        prod = prod_df[feat].values if feat in prod_df.columns else None

        if prod is None:
            continue

        psi     = compute_psi(ref, prod, n_bins)
        ks_stat, ks_pval = compute_ks(ref, prod)

        psi_level = (
            "critical" if psi >= PSI_CRITICAL else
            "warning"  if psi >= PSI_WARNING  else
            "stable"
        )

        feature_results[feat] = {
            "psi":       round(psi, 6),
            "psi_level": psi_level,
            "ks_stat":   round(float(ks_stat), 6),
            "ks_pval":   round(float(ks_pval), 6),
            "ks_drifted": bool(ks_pval < KS_ALPHA),
        }

    # Summary counts
    n_critical  = sum(1 for f in feature_results.values()
                      if f["psi_level"] == "critical")
    n_warning   = sum(1 for f in feature_results.values()
                      if f["psi_level"] == "warning")
    n_ks        = sum(1 for f in feature_results.values()
                      if f["ks_drifted"])
    # Two-key: must fail both KS and PSI critical
    actionable  = [
        feat for feat, r in feature_results.items()
        if r["ks_drifted"] and r["psi_level"] == "critical"
    ]

    fraud_rate = float(y_prod.mean() * 100) if y_prod is not None else None

    return {
        "batch_path":  batch_path,
        "n_features":  len(feature_results),
        "features":    feature_results,
        "summary": {
            "n_stable":     len(feature_results) - n_critical - n_warning,
            "n_warning":    n_warning,
            "n_critical":   n_critical,
            "n_ks_flagged": n_ks,
            "n_actionable": len(actionable),
            "actionable_features": actionable,
        },
        "fraud_rate": fraud_rate,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run drift detection on a production batch."
    )
    parser.add_argument("--batch",        required=True,
                        help="Path to production batch CSV")
    parser.add_argument("--artifacts",    default="artifacts/",
                        help="Path to artifact directory")
    parser.add_argument("--bins",         type=int, default=10,
                        help="Number of PSI histogram bins")
    args = parser.parse_args()

    report = run_drift_monitor(
        batch_path=args.batch,
        artifact_dir=args.artifacts,
        n_bins=args.bins,
    )

    print(f"\n=== DRIFT MONITOR REPORT ===")
    print(f"Batch       : {report['batch_path']}")
    print(f"Fraud rate  : {report['fraud_rate']:.3f}%"
          if report['fraud_rate'] else "Fraud rate  : unknown")
    print(f"\nFeature Summary:")
    print(f"  Stable    : {report['summary']['n_stable']}")
    print(f"  Warning   : {report['summary']['n_warning']}")
    print(f"  Critical  : {report['summary']['n_critical']}")
    print(f"  KS flagged: {report['summary']['n_ks_flagged']}")
    print(f"  Actionable: {report['summary']['n_actionable']}")
    if report['summary']['actionable_features']:
        print(f"  Features  : {report['summary']['actionable_features']}")