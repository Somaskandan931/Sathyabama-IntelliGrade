"""
IntelliGrade-H - Model Fine-tuning Script
Fine-tunes TrOCR on your collected handwriting dataset.
Run: python models/train_trocr.py --dataset datasets/training --output models/trocr-finetuned
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_trocr")


def train(dataset_path: str, output_dir: str, epochs: int, batch_size: int):
    """Run TrOCR fine-tuning on a labeled handwriting dataset."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count samples
    train_labels = dataset_path / "train" / "labels.txt"
    val_labels = dataset_path / "val" / "labels.txt"

    if not train_labels.exists():
        raise FileNotFoundError(
            "Training labels not found. Run: python datasets/collect_dataset.py export"
        )

    with open(train_labels) as f:
        n_train = sum(1 for l in f if l.strip())
    with open(val_labels) as f:
        n_val = sum(1 for l in f if l.strip())

    logger.info(f"Training samples: {n_train}")
    logger.info(f"Validation samples: {n_val}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
    logger.info(f"Output directory: {output_dir}")

    from backend.ocr_module import TrOCREngine
    engine = TrOCREngine()
    engine.fine_tune(
        dataset_path=str(dataset_path / "train"),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size
    )

    logger.info("✅ Fine-tuning complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("Update OCR_ENGINE=trocr and set trocr_model_path in your .env to use this model.")


def evaluate_ocr(model_path: str, test_dir: str):
    """Evaluate OCR accuracy on a test set."""
    from backend.ocr_module import OCRModule
    from PIL import Image
    import numpy as np

    ocr = OCRModule(engine="trocr", trocr_model_path=model_path)

    test_dir = Path(test_dir)
    labels_file = test_dir / "labels.txt"

    if not labels_file.exists():
        logger.error("labels.txt not found in test directory.")
        return

    with open(labels_file, encoding="utf-8") as f:
        samples = [l.strip().split('\t') for l in f if '\t' in l]

    correct_chars = 0
    total_chars = 0
    cer_scores = []

    for filename, gt_text in samples:
        img_path = test_dir / "images" / filename
        if not img_path.exists():
            continue

        result = ocr.extract_text(str(img_path), use_line_segmentation=False)
        pred_text = result.text

        # Character Error Rate
        from e import editops
        try:
            import Levenshtein
            cer = Levenshtein.distance(pred_text, gt_text) / max(len(gt_text), 1)
        except ImportError:
            # fallback simple CER
            cer = abs(len(pred_text) - len(gt_text)) / max(len(gt_text), 1)

        cer_scores.append(cer)
        logger.info(f"{filename}: CER={cer:.3f} | GT='{gt_text[:40]}' | Pred='{pred_text[:40]}'")

    avg_cer = np.mean(cer_scores) if cer_scores else 1.0
    logger.info(f"\n{'=' * 40}")
    logger.info(f"Average Character Error Rate (CER): {avg_cer:.4f}")
    logger.info(f"Accuracy (1 - CER): {(1 - avg_cer) * 100:.2f}%")
    logger.info(f"Samples evaluated: {len(cer_scores)}")
    logger.info(f"{'=' * 40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliGrade-H TrOCR Fine-tuning")
    subparsers = parser.add_subparsers(dest="command")

    # train command
    train_parser = subparsers.add_parser("train", help="Fine-tune TrOCR")
    train_parser.add_argument("--dataset", default="datasets/training",
                               help="Path to dataset (with train/ and val/ subdirs)")
    train_parser.add_argument("--output", default="models/trocr-finetuned",
                               help="Output directory for fine-tuned model")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=8)

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate OCR accuracy")
    eval_parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    eval_parser.add_argument("--test-dir", required=True, help="Path to test set directory")

    args = parser.parse_args()

    if args.command == "train":
        train(args.dataset, args.output, args.epochs, args.batch_size)
    elif args.command == "eval":
        evaluate_ocr(args.model, args.test_dir)
    else:
        parser.print_help()
