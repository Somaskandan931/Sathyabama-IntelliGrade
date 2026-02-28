"""
IntelliGrade-H - Dataset Collection Tool
Helps faculty collect and label handwriting samples for fine-tuning TrOCR.
Run: python datasets/collect_dataset.py
"""

import os
import csv
import sys
import uuid
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("datasets/handwriting_samples")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels.txt"


def setup():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not LABELS_FILE.exists():
        LABELS_FILE.write_text("")
    print(f"Dataset directory: {DATASET_DIR.resolve()}")


def add_sample(image_path: str, transcription: str):
    """Add a labeled sample to the dataset."""
    src = Path(image_path)
    if not src.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    ext = src.suffix
    unique_name = f"{uuid.uuid4().hex[:8]}{ext}"
    dest = IMAGES_DIR / unique_name

    # Copy and verify image
    try:
        img = Image.open(src).convert("RGB")
        img.save(dest)
    except Exception as e:
        print(f"ERROR: Cannot process image: {e}")
        return

    # Append to labels
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{unique_name}\t{transcription}\n")

    print(f"✅ Added: {unique_name} → '{transcription[:50]}...'")


def review_dataset():
    """Print dataset statistics."""
    if not LABELS_FILE.exists():
        print("No dataset found. Run 'add_sample' first.")
        return

    with open(LABELS_FILE, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\n📊 Dataset Statistics")
    print(f"   Total samples: {len(lines)}")
    print(f"   Image directory: {IMAGES_DIR}")

    valid = sum(1 for l in lines if (IMAGES_DIR / l.split('\t')[0]).exists())
    print(f"   Valid image-label pairs: {valid}")

    avg_len = sum(len(l.split('\t')[1]) for l in lines if '\t' in l) / max(len(lines), 1)
    print(f"   Average transcription length: {avg_len:.1f} chars\n")


def export_for_training(output_dir: str = "datasets/training"):
    """
    Export dataset in Hugging Face format for TrOCR fine-tuning.
    Creates train/ and val/ splits (80/20).
    """
    import shutil
    import random

    output = Path(output_dir)
    (output / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output / "val" / "images").mkdir(parents=True, exist_ok=True)

    with open(LABELS_FILE, encoding="utf-8") as f:
        samples = [l.strip() for l in f if '\t' in l]

    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        label_path = output / split_name / "labels.txt"
        with open(label_path, "w", encoding="utf-8") as lf:
            for s in split_samples:
                img_name, transcription = s.split('\t', 1)
                src = IMAGES_DIR / img_name
                if src.exists():
                    shutil.copy(src, output / split_name / "images" / img_name)
                    lf.write(f"{img_name}\t{transcription}\n")

    print(f"✅ Exported {len(train_samples)} train / {len(val_samples)} val samples to {output}")


def generate_synthetic_dataset(n: int = 100, output_dir: str = "datasets/synthetic"):
    """
    Generate synthetic handwriting samples using system fonts.
    Useful for bootstrapping when real data is limited.
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    import textwrap

    output = Path(output_dir)
    (output / "images").mkdir(parents=True, exist_ok=True)
    labels_out = output / "labels.txt"

    # Sample academic sentences
    sentences = [
        "Machine learning is a branch of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "Supervised learning uses labeled training data.",
        "The gradient descent algorithm minimizes the loss function.",
        "Convolutional neural networks excel at image recognition tasks.",
        "Natural language processing enables machines to understand text.",
        "Decision trees use if-else rules to classify data points.",
        "Regularization prevents overfitting in machine learning models.",
        "The backpropagation algorithm computes gradients efficiently.",
        "Transfer learning reuses pretrained models for new tasks.",
        "Attention mechanisms help models focus on relevant features.",
        "BERT is a transformer model trained on large text corpora.",
        "Recurrent neural networks handle sequential data effectively.",
        "Random forests combine multiple decision trees for better accuracy.",
        "K-means clustering partitions data into k distinct groups.",
    ]

    count = 0
    with open(labels_out, "w", encoding="utf-8") as lf:
        for i in range(n):
            sentence = random.choice(sentences)
            # Add random variation
            if random.random() < 0.3:
                sentence += " " + random.choice(sentences)

            # Create image
            img_w, img_h = 800, 60
            img = Image.new("L", (img_w, img_h), color=240)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            # Random slight offset for realism
            x = random.randint(5, 15)
            y = random.randint(5, 20)
            draw.text((x, y), sentence[:80], fill=random.randint(0, 40), font=font)

            filename = f"syn_{i:04d}.png"
            img.save(output / "images" / filename)
            lf.write(f"{filename}\t{sentence}\n")
            count += 1

    print(f"✅ Generated {count} synthetic samples in {output}")


if __name__ == "__main__":
    setup()

    import argparse
    parser = argparse.ArgumentParser(description="IntelliGrade-H Dataset Tool")
    parser.add_argument("action", choices=["add", "review", "export", "synthetic"],
                        help="Action to perform")
    parser.add_argument("--image", help="Path to image (for 'add')")
    parser.add_argument("--text", help="Transcription text (for 'add')")
    parser.add_argument("--n", type=int, default=200, help="Number of synthetic samples")
    args = parser.parse_args()

    if args.action == "add":
        if not args.image or not args.text:
            print("Usage: python collect_dataset.py add --image path.jpg --text 'transcription'")
        else:
            add_sample(args.image, args.text)

    elif args.action == "review":
        review_dataset()

    elif args.action == "export":
        export_for_training()

    elif args.action == "synthetic":
        generate_synthetic_dataset(n=args.n)
        review_dataset()