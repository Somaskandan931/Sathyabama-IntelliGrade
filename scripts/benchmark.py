"""
scripts/benchmark.py
Compare IntelliGrade-H AI scores against human teacher scores using the
metrics module (MAE, Pearson, Kappa, Accuracy within ±1).

Usage:
    python scripts/benchmark.py --csv data/teacher_scores.csv [--max-marks 10]

CSV format (no header):
    submission_id,teacher_score,question_type
    1,7.5,open_ended
    2,1,mcq
    ...

The script calls GET /result/{submission_id} for each row to fetch the AI score.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.metrics import compute_metrics, compute_mcq_metrics, print_metrics_report

API_BASE = "http://localhost:8000"


def fetch_ai_score(submission_id: int) -> tuple:
    """Return (final_score, question_type) from the API, or (None, None) on error."""
    try:
        r = requests.get(f"{API_BASE}/result/{submission_id}", timeout=15)
        r.raise_for_status()
        d = r.json()
        return d.get("final_score"), d.get("question_type", "open_ended")
    except Exception as e:
        print(f"  ⚠️  Could not fetch submission {submission_id}: {e}")
        return None, None


def run(csv_path: str, max_marks: float = 10.0):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rows.append(row)

    print(f"Loaded {len(rows)} rows from {csv_path}")

    open_ai, open_gt = [], []
    mcq_pred, mcq_correct = [], []
    skipped = 0

    for row in rows:
        sub_id = int(row[0].strip())
        teacher_score = float(row[1].strip())
        qtype = row[2].strip() if len(row) > 2 else "open_ended"

        ai_score, detected_type = fetch_ai_score(sub_id)
        if ai_score is None:
            skipped += 1
            continue

        resolved_type = detected_type or qtype
        if resolved_type == "mcq":
            # For MCQ: teacher_score=1 means correct, 0 = wrong
            mcq_pred.append(str(int(ai_score >= max_marks * 0.5)))
            mcq_correct.append(str(int(teacher_score >= max_marks * 0.5)))
        else:
            open_ai.append(ai_score)
            open_gt.append(teacher_score)

    print(f"  Open-ended samples: {len(open_ai)}")
    print(f"  MCQ samples:        {len(mcq_pred)}")
    print(f"  Skipped (no result): {skipped}\n")

    if open_ai:
        report = compute_metrics(open_ai, open_gt, max_marks=max_marks, question_type="open_ended")
        print_metrics_report(report)

    if mcq_pred:
        report = compute_mcq_metrics(mcq_pred, mcq_correct)
        print_metrics_report(report)

    if not open_ai and not mcq_pred:
        print("No results to compute. Make sure the API is running and submissions are evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to scores CSV")
    parser.add_argument("--max-marks", type=float, default=10.0)
    args = parser.parse_args()
    run(args.csv, args.max_marks)
