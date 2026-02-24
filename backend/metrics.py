"""
IntelliGrade-H - Evaluation Metrics Module
Validates AI scoring against teacher ground truth.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class MetricsReport:
    mae: float                # Mean Absolute Error
    pearson_r: float          # Pearson Correlation
    cohen_kappa: float        # Cohen's Kappa Agreement
    accuracy_within_1: float  # % of scores within ±1 mark
    accuracy_within_0_5: float
    n_samples: int
    mean_ai_score: float
    mean_teacher_score: float


def compute_metrics(
    ai_scores: List[float],
    teacher_scores: List[float],
    max_marks: float = 10.0
) -> MetricsReport:
    """
    Compare AI-assigned scores with teacher ground truth.

    Parameters
    ----------
    ai_scores      : list of AI predicted scores
    teacher_scores : list of teacher ground truth scores
    max_marks      : maximum marks per question
    """
    assert len(ai_scores) == len(teacher_scores), "Lists must be same length"
    ai = np.array(ai_scores, dtype=float)
    gt = np.array(teacher_scores, dtype=float)
    n = len(ai)

    # Mean Absolute Error
    mae = float(np.mean(np.abs(ai - gt)))

    # Pearson Correlation
    if np.std(ai) > 0 and np.std(gt) > 0:
        pearson_r = float(np.corrcoef(ai, gt)[0, 1])
    else:
        pearson_r = 0.0

    # Cohen's Kappa (rounded to nearest integer mark)
    kappa = _cohen_kappa(ai.round().astype(int), gt.round().astype(int),
                         max_marks=int(max_marks))

    # Accuracy within ±N marks
    acc_1 = float(np.mean(np.abs(ai - gt) <= 1.0))
    acc_05 = float(np.mean(np.abs(ai - gt) <= 0.5))

    return MetricsReport(
        mae=round(mae, 4),
        pearson_r=round(pearson_r, 4),
        cohen_kappa=round(kappa, 4),
        accuracy_within_1=round(acc_1, 4),
        accuracy_within_0_5=round(acc_05, 4),
        n_samples=n,
        mean_ai_score=round(float(np.mean(ai)), 4),
        mean_teacher_score=round(float(np.mean(gt)), 4)
    )


def _cohen_kappa(pred: np.ndarray, true: np.ndarray, max_marks: int) -> float:
    """Compute linear weighted Cohen's Kappa."""
    from sklearn.metrics import cohen_kappa_score
    try:
        labels = list(range(max_marks + 1))
        return cohen_kappa_score(true, pred, labels=labels, weights="linear")
    except Exception:
        return 0.0


def print_metrics_report(report: MetricsReport):
    """Pretty print a MetricsReport."""
    print("\n" + "=" * 45)
    print("   IntelliGrade-H — Evaluation Metrics")
    print("=" * 45)
    print(f"  Samples:                  {report.n_samples}")
    print(f"  Mean Teacher Score:       {report.mean_teacher_score}")
    print(f"  Mean AI Score:            {report.mean_ai_score}")
    print(f"  Mean Absolute Error:      {report.mae}")
    print(f"  Pearson Correlation:      {report.pearson_r}")
    print(f"  Cohen's Kappa:            {report.cohen_kappa}")
    print(f"  Accuracy within ±1 mark: {report.accuracy_within_1 * 100:.1f}%")
    print(f"  Accuracy within ±0.5:    {report.accuracy_within_0_5 * 100:.1f}%")
    print("=" * 45 + "\n")
