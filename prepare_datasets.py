"""
IntelliGrade-H — Dataset Preparation Script
=============================================
Converts IAM and CVL Kaggle datasets into the labels.txt + images/
format required by scripts/train_trocr.py.

Usage:
  python prepare_datasets.py `
    --iam  "datasets/raw/iam_words" `
    --cvl  "datasets/raw/cvl-database-1-1" `
    --out  "datasets/handwriting"

After running this, execute:
  python scripts/collect_dataset.py split
  python scripts/train_trocr.py train ...
"""

import os
import re
import shutil
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Filters ────────────────────────────────────────────────────────────────────
MIN_TEXT_LEN   = 2      # skip single-character labels
MAX_TEXT_LEN   = 60     # skip very long concatenated words
MIN_IMG_WIDTH  = 30     # skip tiny fragments (px)
MIN_IMG_HEIGHT = 15


def _valid_image(path: Path) -> bool:
    """Quick size check without loading full image."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            return w >= MIN_IMG_WIDTH and h >= MIN_IMG_HEIGHT
    except Exception:
        return False


# ── IAM converter ──────────────────────────────────────────────────────────────

def convert_iam(iam_root: str, out_images: Path, labels_fh) -> int:
    """
    IAM Kaggle structure:
      iam_words/
        words.txt          ← label file
        words/
          a01/
            a01-000u/
              a01-000u-00-00.png   ← word image
              ...

    words.txt format (skip lines starting with #):
      a01-000u-00-00 ok 154 1 408 768 27 51 AT A
      ^word_id       ^status                    ^transcription (last field)

    We only keep rows where status == "ok".
    """
    root       = Path(iam_root)
    words_txt  = root / "words.txt"
    words_dir  = root / "words"

    if not words_txt.exists():
        log.error("IAM: words.txt not found at %s", words_txt)
        return 0
    if not words_dir.exists():
        log.error("IAM: words/ folder not found at %s", words_dir)
        return 0

    # Build id → transcription map
    transcriptions = {}
    with open(words_txt, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) < 9:
                continue
            word_id    = parts[0]          # e.g. a01-000u-00-00
            status     = parts[1]          # ok | er
            transcript = parts[-1]         # last field
            if status == "ok" and MIN_TEXT_LEN <= len(transcript) <= MAX_TEXT_LEN:
                transcriptions[word_id] = transcript

    log.info("IAM: %d valid transcriptions loaded", len(transcriptions))

    count = 0
    for word_id, transcript in transcriptions.items():
        # Image path: words/a01/a01-000u/a01-000u-00-00.png
        parts    = word_id.split("-")
        folder1  = parts[0]                          # a01
        folder2  = "-".join(parts[:2])               # a01-000u (or a01-000ua for some)
        # Try both possible subfolder naming conventions
        img_path = words_dir / folder1 / folder2 / f"{word_id}.png"
        if not img_path.exists():
            # Some versions use just the first two parts as subfolder
            folder2b = parts[0] + parts[1] if len(parts) > 1 else folder2
            img_path = words_dir / folder1 / folder2b / f"{word_id}.png"
        if not img_path.exists():
            continue

        if not _valid_image(img_path):
            continue

        dest_name = f"iam_{word_id}.png"
        dest_path = out_images / dest_name
        shutil.copy2(img_path, dest_path)
        labels_fh.write(f"{dest_name}\t{transcript}\n")
        count += 1

        if count % 1000 == 0:
            log.info("  IAM: copied %d images...", count)

    log.info("✅ IAM: %d samples converted", count)
    return count


# ── CVL converter ──────────────────────────────────────────────────────────────

def convert_cvl(cvl_root: str, out_images: Path, labels_fh) -> int:
    """
    CVL Kaggle structure (cvl-database-1-1):
      cvl-database-1-1/
        cvl-database-public-1-1/
          testset/
            words/
              0001-1-0-0-word.tif    ← word image; last part before .tif = transcription
          trainset/
            words/
              0002-2-1-3-hello.tif

    Filename format:  writerID-textID-lineID-wordID-TRANSCRIPTION.tif
    The transcription is everything after the 4th dash.

    The Kaggle version (amrrsheta) may also contain a flat folder of
    pre-cropped word images with the same naming convention.
    We walk the entire tree and grab every .tif / .png that matches.
    """
    root  = Path(cvl_root)
    count = 0

    # Find all word images recursively
    image_files = list(root.rglob("*.tif")) + list(root.rglob("*.png"))
    log.info("CVL: found %d image files to scan", len(image_files))

    for img_path in image_files:
        stem = img_path.stem   # e.g. 0001-1-0-0-hello

        # Extract transcription: everything after the 4th dash
        parts = stem.split("-")
        if len(parts) < 5:
            continue   # not a word image

        transcript = "-".join(parts[4:])   # handles words that contain dashes

        # Clean up: remove any trailing writer/page metadata
        # CVL labels sometimes have trailing digits after a dot
        transcript = transcript.split(".")[0].strip()

        if not (MIN_TEXT_LEN <= len(transcript) <= MAX_TEXT_LEN):
            continue
        if not re.match(r'^[A-Za-z0-9\s\'\-\,\.\!\?]+$', transcript):
            continue   # skip lines with garbled characters

        if not _valid_image(img_path):
            continue

        dest_name = f"cvl_{img_path.stem}.png"
        dest_path = out_images / dest_name

        try:
            # Convert TIF → PNG for compatibility with TrOCR processor
            from PIL import Image
            with Image.open(img_path) as img:
                img.convert("RGB").save(dest_path)
        except Exception as e:
            log.warning("CVL: skipping %s — %s", img_path.name, e)
            continue

        labels_fh.write(f"{dest_name}\t{transcript}\n")
        count += 1

        if count % 500 == 0:
            log.info("  CVL: copied %d images...", count)

    log.info("✅ CVL: %d samples converted", count)
    return count


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert IAM + CVL Kaggle datasets to IntelliGrade-H training format"
    )
    parser.add_argument("--iam", default=None,
                        help="Path to extracted IAM Kaggle folder (iam_words/)")
    parser.add_argument("--cvl", default=None,
                        help="Path to extracted CVL Kaggle folder (cvl-database-1-1/)")
    parser.add_argument("--out", default="datasets/handwriting",
                        help="Output directory (default: datasets/handwriting)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples per dataset (0 = no limit, useful for quick tests)")
    args = parser.parse_args()

    if not args.iam and not args.cvl:
        parser.error("Provide at least one of --iam or --cvl")

    # Create output structure
    out_root   = Path(args.out)
    out_images = out_root / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    labels_path = out_root / "labels.txt"

    total = 0

    with open(labels_path, "w", encoding="utf-8") as lf:

        if args.iam:
            log.info("\n── Converting IAM dataset ──────────────────")
            n = convert_iam(args.iam, out_images, lf)
            total += n
            if args.limit and total >= args.limit:
                log.info("Reached sample limit (%d)", args.limit)

        if args.cvl:
            log.info("\n── Converting CVL dataset ──────────────────")
            n = convert_cvl(args.cvl, out_images, lf)
            total += n

    log.info("\n══════════════════════════════════════════")
    log.info("✅  Total samples prepared : %d", total)
    log.info("   Images → %s", out_images)
    log.info("   Labels → %s", labels_path)
    log.info("\nNext steps:")
    log.info("  python scripts/collect_dataset.py validate")
    log.info("  python scripts/collect_dataset.py split")
    log.info("  python scripts/train_trocr.py train \\")
    log.info("    --train-dir datasets/handwriting/train \\")
    log.info("    --val-dir   datasets/handwriting/val \\")
    log.info("    --output    models/trocr-finetuned \\")
    log.info("    --epochs 10 --batch-size 8")
    log.info("══════════════════════════════════════════")


if __name__ == "__main__":
    main()