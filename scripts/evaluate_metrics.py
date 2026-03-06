"""
scripts/evaluate_metrics.py
Compare IntelliGrade-H AI scores against human teacher scores.

Metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Pearson Correlation
  - Cohen's Kappa (±1 mark tolerance)
  - Accuracy within ±0.5 / ±1.0 mark

Usage:
    python scripts/evaluate_metrics.py --csv results/grading_comparison.csv

CSV format:
    submission_id,teacher_score,max_marks
    uuid1,7.5,10
    ...

The script fetches AI scores from the API for each submission_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

API_BASE = "http://localhost:8000"


def fetch_ai_score(submission_id: str) -> float | None:
    try:
        r = requests.get(f"{API_BASE}/result/{submission_id}", timeout=30)
        r.raise_for_status()
        return r.json().get("final_score")
    except Exception as e:
        print(f"  ⚠️  Could not fetch {submission_id}: {e}")
        return None


def compute_metrics(teacher: np.ndarray, ai: np.ndarray, max_marks: np.ndarray) -> dict:
    diff = ai - teacher
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr, pval = pearsonr(teacher, ai)

    # Normalise to [0, max_marks] scale for kappa
    t_rounded = np.round(teacher).astype(int)
    ai_rounded = np.round(ai).astype(int)

    # Accuracy within tolerance
    acc_05 = float(np.mean(np.abs(diff) <= 0.5))
    acc_1 = float(np.mean(np.abs(diff) <= 1.0))

    # Kappa requires same possible labels
    all_labels = sorted(set(t_rounded.tolist() + ai_rounded.tolist()))
    kappa = cohen_kappa_score(t_rounded, ai_rounded, labels=all_labels)

    return {
        "n_samples": len(teacher),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Pearson_r": round(corr, 4),
        "Pearson_p": round(pval, 6),
        "Cohen_Kappa": round(kappa, 4),
        "Accuracy_within_0.5": round(acc_05, 4),
        "Accuracy_within_1.0": round(acc_1, 4),
        "Mean_teacher_score": round(float(np.mean(teacher)), 4),
        "Mean_AI_score": round(float(np.mean(ai)), 4),
    }


def run(csv_path: str, output_json: str | None = None):
    df = pd.read_csv(csv_path)
    assert {"submission_id", "teacher_score", "max_marks"}.issubset(df.columns), \
        "CSV must have: submission_id, teacher_score, max_marks"

    print(f"Fetching AI scores for {len(df)} submissions...")
    df["ai_score"] = df["submission_id"].apply(fetch_ai_score)

    missing = df["ai_score"].isna().sum()
    if missing:
        print(f"⚠️  {missing} submissions missing AI score — excluded.")
    df = df.dropna(subset=["ai_score"])

    teacher = df["teacher_score"].values
    ai = df["ai_score"].values
    max_m = df["max_marks"].values

    metrics = compute_metrics(teacher, ai, max_m)

    print("\n" + "═" * 50)
    print("  IntelliGrade-H  –  Evaluation Metrics")
    print("═" * 50)
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
    print("═" * 50 + "\n")

    if output_json:
        Path(output_json).write_text(json.dumps(metrics, indent=2))
        print(f"Metrics saved to: {output_json}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with submission_id, teacher_score, max_marks")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    run(args.csv, args.output)
