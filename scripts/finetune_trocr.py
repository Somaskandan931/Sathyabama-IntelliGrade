"""
scripts/finetune_trocr.py
Fine-tune TrOCR on a local handwriting dataset.

Dataset format (CSV):
    image_path,text
    datasets/handwriting/img001.jpg,"Machine learning is a subset of AI"
    ...

Usage:
    python scripts/finetune_trocr.py \
        --dataset datasets/handwriting_dataset.csv \
        --output models/trocr-finetuned \
        --epochs 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────
class HandwritingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: TrOCRProcessor, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        text = str(row["text"])

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        # Replace padding token id with -100 so they are ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Training loop ─────────────────────────────────────────────────────────────
def train(
    dataset_csv: str,
    output_dir: str,
    base_model: str = "microsoft/trocr-base-handwritten",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-5,
    val_split: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load data
    df = pd.read_csv(dataset_csv)
    logger.info("Dataset size: %d samples", len(df))

    split = int(len(df) * (1 - val_split))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    # Load model & processor
    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    model.to(device)

    # Configure decoder
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    train_set = HandwritingDataset(train_df, processor)
    val_set = HandwritingDataset(val_df, processor)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                num_training_steps=total_steps)

    best_val_loss = float("inf")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train = train_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)
        logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f", epoch, epochs, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            logger.info("  ✅ Saved best model (val_loss=%.4f)", best_val_loss)

    logger.info("Training complete. Best model at: %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on handwriting data")
    parser.add_argument("--dataset", required=True, help="Path to CSV with columns: image_path, text")
    parser.add_argument("--output", default="models/trocr-finetuned", help="Output directory")
    parser.add_argument("--base-model", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    train(
        dataset_csv=args.dataset,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
