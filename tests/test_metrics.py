"""
IntelliGrade-H: Evaluation Metrics & Testing
Validates AI grading against teacher scores using MAE, Pearson r, Cohen's Kappa.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import math

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Metric Functions (pure Python, no heavy deps)
# ─────────────────────────────────────────────

def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """MAE between teacher scores and AI scores."""
    assert len(y_true) == len(y_pred), "Lists must be same length"
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def pearson_correlation(y_true: List[float], y_pred: List[float]) -> float:
    """Pearson r between teacher and AI scores."""
    n = len(y_true)
    mean_t = sum(y_true) / n
    mean_p = sum(y_pred) / n
    cov = sum((t - mean_t) * (p - mean_p) for t, p in zip(y_true, y_pred)) / n
    std_t = math.sqrt(sum((t - mean_t) ** 2 for t in y_true) / n)
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in y_pred) / n)
    if std_t == 0 or std_p == 0:
        return 0.0
    return round(cov / (std_t * std_p), 4)


def accuracy_within_margin(
    y_true: List[float], y_pred: List[float], margin: float = 1.0
) -> float:
    """Fraction of predictions within ±margin of teacher score."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= margin)
    return round(correct / len(y_true), 4)


def cohen_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """
    Cohen's Kappa for ordinal agreement.
    Scores should be discretized to integers first.
    """
    categories = sorted(set(y_true + y_pred))
    n = len(y_true)
    # Observed agreement
    p_o = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    # Expected agreement
    p_e = sum(
        (y_true.count(c) / n) * (y_pred.count(c) / n)
        for c in categories
    )
    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


def discretize(scores: List[float], step: float = 0.5) -> List[int]:
    """Round scores to nearest step and convert to int×10 for kappa."""
    return [round(int(s / step) * step * 10) for s in scores]


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class SystemEvaluator:
    """
    Compare AI-generated scores against teacher-provided ground truth.

    CSV format:
        student_id, question, teacher_score, max_marks, student_answer, teacher_answer
    """

    def __init__(self):
        pass

    def evaluate_from_csv(self, csv_path: str, engine=None) -> Dict:
        """
        Run end-to-end evaluation against a labelled CSV.
        If engine is None, reports must already have 'ai_score' column.
        """
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        teacher_scores = []
        ai_scores = []
        errors = []

        for i, row in enumerate(rows):
            teacher = float(row["teacher_score"])
            teacher_scores.append(teacher)

            if "ai_score" in row:
                ai_scores.append(float(row["ai_score"]))
            elif engine:
                try:
                    report = engine.grade_text(
                        question=row["question"],
                        teacher_answer=row["teacher_answer"],
                        student_answer=row["student_answer"],
                        max_marks=float(row["max_marks"]),
                        student_id=row.get("student_id", str(i)),
                    )
                    ai_scores.append(report.final_score)
                except Exception as e:
                    logger.error(f"Row {i} failed: {e}")
                    errors.append(i)
                    ai_scores.append(0.0)
            else:
                raise ValueError("Either provide 'ai_score' column or pass an engine.")

        metrics = self._compute_metrics(teacher_scores, ai_scores)
        metrics["errors"] = len(errors)
        metrics["total"] = len(rows)
        return metrics

    def _compute_metrics(
        self, teacher: List[float], ai: List[float]
    ) -> Dict:
        disc_t = discretize(teacher)
        disc_ai = discretize(ai)
        return {
            "mae": round(mean_absolute_error(teacher, ai), 4),
            "pearson_r": pearson_correlation(teacher, ai),
            "accuracy_within_1": accuracy_within_margin(teacher, ai, 1.0),
            "accuracy_within_05": accuracy_within_margin(teacher, ai, 0.5),
            "cohen_kappa": cohen_kappa(disc_t, disc_ai),
            "n": len(teacher),
        }

    def print_report(self, metrics: Dict):
        print("=" * 45)
        print("  IntelliGrade-H — Evaluation Report")
        print("=" * 45)
        print(f"  Samples evaluated    : {metrics['n']}")
        print(f"  Mean Absolute Error  : {metrics['mae']}")
        print(f"  Pearson Correlation  : {metrics['pearson_r']}")
        print(f"  Accuracy (±1 mark)   : {metrics['accuracy_within_1']:.1%}")
        print(f"  Accuracy (±0.5 mark) : {metrics['accuracy_within_05']:.1%}")
        print(f"  Cohen's Kappa        : {metrics['cohen_kappa']}")
        if "errors" in metrics:
            print(f"  Grading errors       : {metrics['errors']}")
        print("=" * 45)


# ─────────────────────────────────────────────
# Unit Tests (pytest)
# ─────────────────────────────────────────────

def test_mae():
    assert mean_absolute_error([8, 7, 9], [7.5, 7, 8.5]) == pytest.approx(0.333, abs=0.01)

def test_pearson():
    r = pearson_correlation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert r == 1.0

def test_accuracy_margin():
    acc = accuracy_within_margin([8, 7, 9, 6], [7.5, 6, 9, 5], margin=1.0)
    assert acc == 1.0

def test_cohen_kappa_perfect():
    labels = [1, 2, 3, 4, 5]
    assert cohen_kappa(labels, labels) == 1.0

def test_system_evaluator_no_engine():
    import tempfile, csv, os
    rows = [
        {"student_id": "s1", "question": "Q", "teacher_answer": "A",
         "student_answer": "A", "teacher_score": "8", "max_marks": "10", "ai_score": "7.5"},
        {"student_id": "s2", "question": "Q", "teacher_answer": "A",
         "student_answer": "B", "teacher_score": "5", "max_marks": "10", "ai_score": "5.2"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        tmp = f.name
    evaluator = SystemEvaluator()
    metrics = evaluator.evaluate_from_csv(tmp)
    os.unlink(tmp)
    assert metrics["n"] == 2
    assert metrics["mae"] < 1.0


if __name__ == "__main__":
    import sys
    # Simple CLI test
    teacher = [8, 7, 9, 6, 5, 10, 4, 7, 8, 6]
    ai      = [7.5, 6.8, 8.7, 6.2, 5.1, 9.8, 4.3, 7.2, 7.9, 6.0]
    ev = SystemEvaluator()
    metrics = ev._compute_metrics(teacher, ai)
    ev.print_report(metrics)

try:
    import pytest
except ImportError:
    pass
