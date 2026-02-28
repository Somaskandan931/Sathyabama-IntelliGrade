"""
IntelliGrade-H - Model Fine-tuning Script
Fine-tunes TrOCR on your collected handwriting dataset.

Usage:
  python models/train_trocr.py train --dataset datasets/training --output models/trocr-finetuned
  python models/train_trocr.py eval  --model models/trocr-finetuned --test-dir datasets/training/val
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_trocr")


# ─────────────────────────────────────────────────────────
# CER helper — no broken third-party imports
# ─────────────────────────────────────────────────────────

def _compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate via Levenshtein distance (pure Python fallback)."""
    if not gt:
        return 0.0 if not pred else 1.0
    try:
        import Levenshtein
        return Levenshtein.distance(pred, gt) / len(gt)
    except ImportError:
        pass
    # Pure-Python edit distance
    m, n = len(pred), len(gt)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if pred[i - 1] == gt[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / len(gt)


# ─────────────────────────────────────────────────────────
# train
# ─────────────────────────────────────────────────────────

def train(dataset_path: str, output_dir: str, epochs: int, batch_size: int):
    dataset_path = Path(dataset_path)
    output_dir   = Path(output_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_labels = dataset_path / "train" / "labels.txt"
    val_labels   = dataset_path / "val"   / "labels.txt"

    if not train_labels.exists():
        raise FileNotFoundError(
            "Training labels not found. Run: python datasets/collect_dataset.py export"
        )

    with open(train_labels) as f:
        n_train = sum(1 for l in f if l.strip())
    with open(val_labels) as f:
        n_val = sum(1 for l in f if l.strip())

    logger.info("Training samples : %d", n_train)
    logger.info("Validation samples: %d", n_val)
    logger.info("Epochs: %d  Batch size: %d", epochs, batch_size)

    from backend.ocr_module import TrOCREngine
    TrOCREngine().fine_tune(
        dataset_path=str(dataset_path / "train"),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
    )
    logger.info("Fine-tuning complete! Model saved to: %s", output_dir)


# ─────────────────────────────────────────────────────────
# eval
# ─────────────────────────────────────────────────────────

def evaluate_ocr(model_path: str, test_dir: str):
    from backend.ocr_module import OCRModule

    ocr = OCRModule(engine="trocr", trocr_model_path=model_path)
    test_dir    = Path(test_dir)
    labels_file = test_dir / "labels.txt"

    if not labels_file.exists():
        logger.error("labels.txt not found in test directory.")
        return

    with open(labels_file, encoding="utf-8") as f:
        samples = [l.strip().split("\t") for l in f if "\t" in l]

    cer_scores = []
    for filename, gt_text in samples:
        img_path = test_dir / "images" / filename
        if not img_path.exists():
            continue
        pred_text = ocr.extract_text(str(img_path), use_line_segmentation=False).text
        cer = _compute_cer(pred_text, gt_text)
        cer_scores.append(cer)
        logger.info(
            "%s: CER=%.3f | GT='%s' | Pred='%s'",
            filename, cer, gt_text[:40], pred_text[:40],
        )

    avg_cer = float(np.mean(cer_scores)) if cer_scores else 1.0
    logger.info("=" * 40)
    logger.info("Avg CER: %.4f  |  Accuracy: %.2f%%  |  Samples: %d",
                avg_cer, (1 - avg_cer) * 100, len(cer_scores))
    logger.info("=" * 40)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliGrade-H TrOCR Fine-tuning")
    sub = parser.add_subparsers(dest="command")

    tp = sub.add_parser("train", help="Fine-tune TrOCR")
    tp.add_argument("--dataset",    default="datasets/training")
    tp.add_argument("--output",     default="models/trocr-finetuned")
    tp.add_argument("--epochs",     type=int, default=5)
    tp.add_argument("--batch-size", type=int, default=8)

    ep = sub.add_parser("eval", help="Evaluate OCR accuracy")
    ep.add_argument("--model",    required=True)
    ep.add_argument("--test-dir", required=True)

    args = parser.parse_args()
    if args.command == "train":
        train(args.dataset, args.output, args.epochs, args.batch_size)
    elif args.command == "eval":
        evaluate_ocr(args.model, args.test_dir)
    else:
        parser.print_help()