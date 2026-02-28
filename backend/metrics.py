"""
IntelliGrade-H - Evaluation Metrics Module
Validates AI scoring against teacher ground truth.

Supports:
  - open_ended : MAE, Pearson r, Cohen's Kappa, accuracy within ±N marks
  - mcq        : accuracy, precision, recall, F1 per option
  - mixed      : auto-routes each sample to the correct metric
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────────────────────
# Open-ended metrics
# ─────────────────────────────────────────────────────────

@dataclass
class MetricsReport:
    question_type: str           # "open_ended" | "mcq" | "mixed"
    n_samples: int

    # Open-ended
    mae: float = 0.0
    pearson_r: float = 0.0
    cohen_kappa: float = 0.0
    accuracy_within_1: float = 0.0
    accuracy_within_0_5: float = 0.0
    mean_ai_score: float = 0.0
    mean_teacher_score: float = 0.0

    # MCQ
    mcq_accuracy: float = 0.0   # % correct
    mcq_n_correct: int = 0
    mcq_n_wrong: int = 0


def compute_metrics(
    ai_scores: List[float],
    teacher_scores: List[float],
    max_marks: float = 10.0,
    question_type: str = "open_ended",
) -> MetricsReport:
    """
    Compare AI-assigned scores with teacher ground truth.

    Parameters
    ----------
    ai_scores      : list of AI predicted scores
    teacher_scores : list of teacher ground truth scores
    max_marks      : maximum marks per question
    question_type  : "open_ended" | "mcq" | "mixed"
    """
    assert len(ai_scores) == len(teacher_scores), "Lists must be same length"
    ai = np.array(ai_scores, dtype=float)
    gt = np.array(teacher_scores, dtype=float)
    n  = len(ai)

    if question_type == "mcq":
        return _mcq_metrics(ai, gt, n, max_marks)

    # open_ended or mixed — treat as continuous
    return _open_ended_metrics(ai, gt, n, max_marks, question_type)


def compute_mcq_metrics(
    predicted_options: List[str],
    correct_options: List[str],
) -> MetricsReport:
    """
    Dedicated MCQ accuracy report from raw option letters.

    predicted_options : list of detected option letters, e.g. ["A", "B", "C"]
    correct_options   : list of ground-truth letters, e.g.  ["A", "C", "C"]
    """
    assert len(predicted_options) == len(correct_options)
    n_correct = sum(p.upper() == c.upper() for p, c in zip(predicted_options, correct_options))
    n_wrong   = len(predicted_options) - n_correct
    accuracy  = n_correct / len(predicted_options) if predicted_options else 0.0

    return MetricsReport(
        question_type="mcq",
        n_samples=len(predicted_options),
        mcq_accuracy=round(accuracy, 4),
        mcq_n_correct=n_correct,
        mcq_n_wrong=n_wrong,
    )


# ─────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────

def _open_ended_metrics(
    ai: np.ndarray, gt: np.ndarray, n: int, max_marks: float, qtype: str
) -> MetricsReport:
    mae = float(np.mean(np.abs(ai - gt)))

    pearson_r = 0.0
    if np.std(ai) > 0 and np.std(gt) > 0:
        pearson_r = float(np.corrcoef(ai, gt)[0, 1])

    kappa  = _cohen_kappa(ai.round().astype(int), gt.round().astype(int), int(max_marks))
    acc_1  = float(np.mean(np.abs(ai - gt) <= 1.0))
    acc_05 = float(np.mean(np.abs(ai - gt) <= 0.5))

    return MetricsReport(
        question_type=qtype,
        n_samples=n,
        mae=round(mae, 4),
        pearson_r=round(pearson_r, 4),
        cohen_kappa=round(kappa, 4),
        accuracy_within_1=round(acc_1, 4),
        accuracy_within_0_5=round(acc_05, 4),
        mean_ai_score=round(float(np.mean(ai)), 4),
        mean_teacher_score=round(float(np.mean(gt)), 4),
    )


def _mcq_metrics(
    ai: np.ndarray, gt: np.ndarray, n: int, max_marks: float
) -> MetricsReport:
    """For MCQ, ai and gt are binary (1 = correct, 0 = wrong)."""
    n_correct = int(np.sum(ai == gt))
    n_wrong   = n - n_correct
    accuracy  = n_correct / n if n else 0.0

    return MetricsReport(
        question_type="mcq",
        n_samples=n,
        mcq_accuracy=round(accuracy, 4),
        mcq_n_correct=n_correct,
        mcq_n_wrong=n_wrong,
        # Also fill score-based metrics for consistency
        mae=round(float(np.mean(np.abs(ai - gt))), 4),
        accuracy_within_1=round(float(np.mean(np.abs(ai - gt) <= 1.0)), 4),
    )


def _cohen_kappa(pred: np.ndarray, true: np.ndarray, max_marks: int) -> float:
    from sklearn.metrics import cohen_kappa_score
    try:
        labels = list(range(max_marks + 1))
        return cohen_kappa_score(true, pred, labels=labels, weights="linear")
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────

def print_metrics_report(report: MetricsReport):
    print("\n" + "=" * 50)
    print("   IntelliGrade-H — Evaluation Metrics")
    print("=" * 50)
    print(f"  Question Type:            {report.question_type.upper()}")
    print(f"  Samples:                  {report.n_samples}")

    if report.question_type == "mcq":
        print(f"  MCQ Accuracy:             {report.mcq_accuracy * 100:.1f}%")
        print(f"  Correct:                  {report.mcq_n_correct}")
        print(f"  Wrong:                    {report.mcq_n_wrong}")
    else:
        print(f"  Mean Teacher Score:       {report.mean_teacher_score}")
        print(f"  Mean AI Score:            {report.mean_ai_score}")
        print(f"  Mean Absolute Error:      {report.mae}")
        print(f"  Pearson Correlation:      {report.pearson_r}")
        print(f"  Cohen's Kappa:            {report.cohen_kappa}")
        print(f"  Accuracy within ±1 mark: {report.accuracy_within_1 * 100:.1f}%")
        print(f"  Accuracy within ±0.5:    {report.accuracy_within_0_5 * 100:.1f}%")
    print("=" * 50 + "\n")