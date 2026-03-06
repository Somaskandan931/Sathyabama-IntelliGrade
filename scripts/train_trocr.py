"""
IntelliGrade-H - TrOCR Fine-tuning Script (v2)
================================================
Enhanced training with:
- Data augmentation
- Learning rate scheduling
- Early stopping
- Evaluation metrics
- Model checkpointing
- TensorBoard logging
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback
)
from datasets import load_metric
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HandwritingDataset(Dataset):
    """Dataset for TrOCR fine-tuning"""

    def __init__(self, data_dir: str, processor: TrOCRProcessor,
                 max_length: int = 128, augment: bool = False):
        """
        Args:
            data_dir: Directory containing images/ and labels.txt
            processor: TrOCR processor
            max_length: Maximum sequence length
            augment: Whether to apply data augmentation
        """
        self.processor = processor
        self.max_length = max_length
        self.augment = augment

        # Load data
        self.data = []
        data_dir = Path(data_dir)
        labels_file = data_dir / "labels.txt"
        images_dir = data_dir / "images"

        if not labels_file.exists():
            raise FileNotFoundError(f"labels.txt not found in {data_dir}")

        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    filename, text = line.split('\t', 1)
                    img_path = images_dir / filename
                    if img_path.exists():
                        self.data.append((str(img_path), text.strip()))

        logger.info(f"Loaded {len(self.data)} samples from {data_dir}")

    def __len__(self):
        return len(self.data)

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Apply data augmentation"""
        from PIL import ImageEnhance, ImageFilter
        import random

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-3, 3)
            image = image.rotate(angle, expand=True, fillcolor=255)

        # Random brightness/contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random blur
        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        return image

    def __getitem__(self, idx):
        img_path, text = self.data[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply augmentation if enabled
        if self.augment:
            image = self._augment_image(image)

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Process text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Replace padding token id with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def compute_metrics(eval_pred):
    """Compute CER and WER metrics"""
    predictions, references = eval_pred

    # Decode predictions
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)

    # Decode references (remove -100)
    references = np.where(references != -100, references, processor.tokenizer.pad_token_id)
    ref_str = processor.batch_decode(references, skip_special_tokens=True)

    # Compute CER
    cer_metric = load_metric("cer")
    cer = cer_metric.compute(predictions=pred_str, references=ref_str)

    # Compute WER
    wer_metric = load_metric("wer")
    wer = wer_metric.compute(predictions=pred_str, references=ref_str)

    return {"cer": cer, "wer": wer}


def train_model(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    model_name: str = "microsoft/trocr-small-handwritten",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    early_stopping_patience: int = 3,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    augment: bool = True
):
    """
    Fine-tune TrOCR model.

    Args:
        train_dir: Directory with training data
        val_dir: Directory with validation data
        output_dir: Directory to save model
        model_name: Base model name
        epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        early_stopping_patience: Patience for early stopping
        gradient_accumulation_steps: Gradient accumulation steps
        fp16: Use mixed precision training
        augment: Apply data augmentation
    """
    global processor
    processor = TrOCRProcessor.from_pretrained(model_name)

    # Load datasets
    train_dataset = HandwritingDataset(train_dir, processor, augment=augment)
    val_dataset = HandwritingDataset(val_dir, processor, augment=False)

    # Load model
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Set decoder start token id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=fp16 and torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Save training config
    config = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "augment": augment,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "date": datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Training complete!")


def evaluate_model(
    model_path: str,
    test_dir: str,
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to trained model
        test_dir: Directory with test data
        batch_size: Batch size

    Returns:
        Dictionary with evaluation metrics
    """
    global processor

    # Load model and processor
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = HandwritingDataset(test_dir, processor, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Metrics
    cer_metric = load_metric("cer")
    wer_metric = load_metric("wer")

    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)

            # Generate
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

            # Decode
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Decode references
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            ref_text = processor.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(pred_text)
            all_refs.extend(ref_text)

    # Compute metrics
    cer = cer_metric.compute(predictions=all_preds, references=all_refs)
    wer = wer_metric.compute(predictions=all_preds, references=all_refs)

    # Character accuracy
    total_chars = sum(len(ref) for ref in all_refs)
    correct_chars = sum(
        sum(1 for p, r in zip(pred, ref) if p == r)
        for pred, ref in zip(all_preds, all_refs)
    )
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0

    # Word accuracy
    total_words = sum(len(ref.split()) for ref in all_refs)
    correct_words = sum(
        sum(1 for p, r in zip(pred.split(), ref.split()) if p == r)
        for pred, ref in zip(all_preds, all_refs)
    )
    word_accuracy = correct_words / total_words if total_words > 0 else 0

    results = {
        "cer": float(cer),
        "wer": float(wer),
        "char_accuracy": float(char_accuracy),
        "word_accuracy": float(word_accuracy),
        "samples": len(test_dataset)
    }

    # Print results
    print("\n" + "=" * 50)
    print("📊 Evaluation Results")
    print("=" * 50)
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"Character Accuracy: {char_accuracy*100:.2f}%")
    print(f"Word Accuracy: {word_accuracy*100:.2f}%")
    print(f"Test Samples: {len(test_dataset)}")
    print("=" * 50 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="IntelliGrade-H TrOCR Fine-tuning")
    parser.add_argument("mode", choices=["train", "eval", "tune"], help="Mode: train, eval, or tune")

    # Common arguments
    parser.add_argument("--model", default="microsoft/trocr-small-handwritten",
                       help="Base model name or path")
    parser.add_argument("--output", default="models/trocr-finetuned",
                       help="Output directory")

    # Training arguments
    parser.add_argument("--train-dir", default="datasets/training/train",
                       help="Training data directory")
    parser.add_argument("--val-dir", default="datasets/training/val",
                       help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")

    # Evaluation arguments
    parser.add_argument("--test-dir", default="datasets/training/test",
                       help="Test data directory")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            augment=not args.no_augment,
            fp16=not args.no_fp16 and torch.cuda.is_available()
        )

    elif args.mode == "eval":
        evaluate_model(
            model_path=args.model,
            test_dir=args.test_dir,
            batch_size=args.batch_size
        )

    elif args.mode == "tune":
        # Hyperparameter tuning
        from itertools import product

        learning_rates = [1e-5, 3e-5, 5e-5, 1e-4]
        batch_sizes = [4, 8, 16]

        best_cer = float('inf')
        best_config = None

        for lr, bs in product(learning_rates, batch_sizes):
            logger.info(f"Tuning with lr={lr}, bs={bs}")

            output_dir = f"{args.output}_lr{lr}_bs{bs}"

            train_model(
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                output_dir=output_dir,
                model_name=args.model,
                epochs=5,  # Quick epochs for tuning
                batch_size=bs,
                learning_rate=lr,
                augment=True
            )

            # Evaluate
            results = evaluate_model(
                model_path=output_dir,
                test_dir=args.val_dir,
                batch_size=bs
            )

            if results["cer"] < best_cer:
                best_cer = results["cer"]
                best_config = {"lr": lr, "batch_size": bs, "cer": best_cer}

        print("\n" + "=" * 50)
        print("🏆 Best Configuration")
        print("=" * 50)
        print(f"Learning Rate: {best_config['lr']}")
        print(f"Batch Size: {best_config['batch_size']}")
        print(f"CER: {best_config['cer']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()