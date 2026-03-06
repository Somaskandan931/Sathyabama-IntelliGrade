"""
IntelliGrade-H - Dataset Collection Tool (v2)
==============================================
Enhanced dataset collection with:
- Handwriting sample collection
- Synthetic data generation
- Data augmentation
- Quality validation
- Export to multiple formats
"""

import os
import sys
import json
import uuid
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import csv
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system("pip install Pillow numpy")
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HandwritingDataset:
    """
    Manages handwriting dataset collection and preparation.
    """

    def __init__(self, base_dir: str = "datasets/handwriting"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.labels_file = self.base_dir / "labels.txt"
        self.metadata_file = self.base_dir / "metadata.json"

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load existing dataset metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_samples": 0,
            "samples": [],
            "augmented": False
        }

    def _save_metadata(self):
        """Save dataset metadata"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.metadata["total_samples"] = len(self.metadata["samples"])

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_sample(self, image_path: str, transcription: str,
                   student_id: Optional[str] = None,
                   quality: float = 1.0) -> bool:
        """
        Add a labeled handwriting sample to the dataset.

        Args:
            image_path: Path to the handwriting image
            transcription: Correct text transcription
            student_id: Optional student identifier
            quality: Image quality rating (0-1)

        Returns:
            bool: Success status
        """
        src = Path(image_path)
        if not src.exists():
            logger.error(f"Image not found: {image_path}")
            return False

        # Generate unique filename
        ext = src.suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            logger.error(f"Unsupported image format: {ext}")
            return False

        unique_id = uuid.uuid4().hex[:8]
        filename = f"{unique_id}{ext}"
        dest = self.images_dir / filename

        try:
            # Load and validate image
            img = Image.open(src).convert("RGB")

            # Basic quality checks
            if img.size[0] < 100 or img.size[1] < 50:
                logger.warning(f"Image too small: {img.size}")

            # Save image
            img.save(dest, quality=95)

            # Append to labels
            with open(self.labels_file, "a", encoding="utf-8") as f:
                f.write(f"{filename}\t{transcription}\n")

            # Add to metadata
            sample = {
                "filename": filename,
                "transcription": transcription,
                "student_id": student_id,
                "quality": quality,
                "added": datetime.now().isoformat(),
                "image_size": img.size,
                "format": ext
            }
            self.metadata["samples"].append(sample)
            self._save_metadata()

            logger.info(f"✅ Added: {filename} ({len(transcription)} chars)")
            return True

        except Exception as e:
            logger.error(f"Failed to add sample: {e}")
            return False

    def add_batch(self, samples: List[Tuple[str, str]]) -> Tuple[int, int]:
        """
        Add multiple samples at once.

        Args:
            samples: List of (image_path, transcription) tuples

        Returns:
            (success_count, fail_count)
        """
        success = 0
        failed = 0

        for img_path, text in samples:
            if self.add_sample(img_path, text):
                success += 1
            else:
                failed += 1

            # Print progress
            if (success + failed) % 10 == 0:
                logger.info(f"Progress: {success + failed}/{len(samples)}")

        return success, failed

    def generate_synthetic(self, n_samples: int = 100,
                          variations: int = 3) -> int:
        """
        Generate synthetic handwriting samples using fonts.

        Args:
            n_samples: Number of base sentences to generate
            variations: Number of variations per sentence

        Returns:
            Number of samples generated
        """
        from .synthetic_generator import SyntheticHandwritingGenerator

        generator = SyntheticHandwritingGenerator()
        count = 0

        for i in range(n_samples):
            # Generate base sentence
            sentence = generator.generate_sentence()

            for v in range(variations):
                # Create variation
                img = generator.render_text(
                    sentence,
                    handwriting_style=random.choice(["neat", "messy", "cursive"]),
                    noise_level=random.uniform(0.1, 0.5)
                )

                # Save image
                filename = f"syn_{i:04d}_v{v}.png"
                img_path = self.images_dir / filename
                img.save(img_path)

                # Save label
                with open(self.labels_file, "a", encoding="utf-8") as f:
                    f.write(f"{filename}\t{sentence}\n")

                # Add to metadata
                sample = {
                    "filename": filename,
                    "transcription": sentence,
                    "synthetic": True,
                    "variation": v,
                    "added": datetime.now().isoformat()
                }
                self.metadata["samples"].append(sample)

                count += 1

                if count % 50 == 0:
                    logger.info(f"Generated {count} samples...")

        self._save_metadata()
        logger.info(f"✅ Generated {count} synthetic samples")
        return count

    def augment_dataset(self, techniques: List[str] = None) -> int:
        """
        Apply data augmentation to existing samples.

        Args:
            techniques: List of augmentation techniques

        Returns:
            Number of augmented samples created
        """
        if techniques is None:
            techniques = ["rotate", "noise", "blur", "contrast"]

        count = 0
        aug_dir = self.base_dir / "augmented"
        aug_dir.mkdir(exist_ok=True)

        # Load existing samples
        samples = self.metadata["samples"]

        for sample in samples:
            if sample.get("synthetic"):
                continue  # Skip synthetic for now

            img_path = self.images_dir / sample["filename"]
            if not img_path.exists():
                continue

            img = Image.open(img_path)
            text = sample["transcription"]

            # Apply each augmentation
            for tech in techniques:
                aug_img = self._apply_augmentation(img, tech)
                if aug_img:
                    aug_filename = f"aug_{tech}_{sample['filename']}"
                    aug_path = aug_dir / aug_filename
                    aug_img.save(aug_path)

                    # Add to dataset
                    self.add_sample(str(aug_path), text, quality=0.8)
                    count += 1

        logger.info(f"✅ Created {count} augmented samples")
        return count

    def _apply_augmentation(self, img: Image.Image, technique: str) -> Optional[Image.Image]:
        """Apply specific augmentation technique"""
        try:
            if technique == "rotate":
                angle = random.uniform(-5, 5)
                return img.rotate(angle, expand=True, fillcolor=255)

            elif technique == "noise":
                import numpy as np
                arr = np.array(img)
                noise = np.random.normal(0, 25, arr.shape).astype(np.uint8)
                noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy)

            elif technique == "blur":
                return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            elif technique == "contrast":
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img)
                return enhancer.enhance(random.uniform(0.8, 1.5))

        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            return None

        return None

    def split_dataset(self, train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     seed: int = 42) -> Dict[str, int]:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            seed: Random seed

        Returns:
            Dictionary with split counts
        """
        random.seed(seed)

        # Load all samples
        samples = []
        if self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        filename, text = line.split('\t', 1)
                        if (self.images_dir / filename).exists():
                            samples.append((filename, text))

        if not samples:
            logger.error("No samples found")
            return {"train": 0, "val": 0, "test": 0}

        # Shuffle
        random.shuffle(samples)

        # Split
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]

        # Create split directories
        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }

        for split_name, split_samples in splits.items():
            split_dir = self.base_dir / split_name
            split_dir.mkdir(exist_ok=True)
            images_dir = split_dir / "images"
            images_dir.mkdir(exist_ok=True)

            # Copy images and create labels file
            labels_path = split_dir / "labels.txt"
            with open(labels_path, 'w', encoding='utf-8') as lf:
                for filename, text in split_samples:
                    src = self.images_dir / filename
                    dst = images_dir / filename
                    shutil.copy2(src, dst)
                    lf.write(f"{filename}\t{text}\n")

            logger.info(f"Split {split_name}: {len(split_samples)} samples")

        return {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        }

    def export_for_trocr(self, output_dir: str = "datasets/training") -> str:
        """
        Export dataset in TrOCR training format.

        Args:
            output_dir: Output directory

        Returns:
            Path to exported dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # First split the dataset
        splits = self.split_dataset()

        # Create HuggingFace dataset structure
        hf_dir = output_path / "hf_dataset"
        hf_dir.mkdir(exist_ok=True)

        # Create metadata
        metadata = {
            "dataset_name": "IntelliGrade-H Handwriting Dataset",
            "description": "Handwriting samples for TrOCR fine-tuning",
            "samples": self.metadata["total_samples"],
            "splits": splits,
            "created": datetime.now().isoformat()
        }

        with open(hf_dir / "dataset_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Dataset exported to {hf_dir}")
        return str(hf_dir)

    def validate_dataset(self) -> Dict[str, any]:
        """
        Validate dataset quality and integrity.

        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "total_samples": 0,
            "issues": [],
            "statistics": {}
        }

        # Check if labels file exists
        if not self.labels_file.exists():
            report["valid"] = False
            report["issues"].append("labels.txt not found")
            return report

        # Check all samples
        valid_count = 0
        total_chars = 0
        image_sizes = []

        with open(self.labels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if '\t' not in line:
                    report["issues"].append(f"Line {line_num}: Missing tab separator")
                    continue

                filename, text = line.split('\t', 1)

                # Check image exists
                img_path = self.images_dir / filename
                if not img_path.exists():
                    report["issues"].append(f"Missing image: {filename}")
                    continue

                # Check image can be opened
                try:
                    img = Image.open(img_path)
                    image_sizes.append(img.size)

                    # Basic text validation
                    if not text.strip():
                        report["issues"].append(f"Empty transcription: {filename}")
                    else:
                        valid_count += 1
                        total_chars += len(text)

                except Exception as e:
                    report["issues"].append(f"Cannot open {filename}: {e}")

        # Calculate statistics
        report["total_samples"] = valid_count
        report["statistics"] = {
            "average_length": total_chars / valid_count if valid_count else 0,
            "total_characters": total_chars,
            "min_image_width": min(s[0] for s in image_sizes) if image_sizes else 0,
            "max_image_width": max(s[0] for s in image_sizes) if image_sizes else 0,
            "min_image_height": min(s[1] for s in image_sizes) if image_sizes else 0,
            "max_image_height": max(s[1] for s in image_sizes) if image_sizes else 0,
        }

        if valid_count == 0:
            report["valid"] = False
            report["issues"].append("No valid samples found")

        return report

    def review(self):
        """Display dataset statistics"""
        report = self.validate_dataset()

        print("\n" + "=" * 50)
        print("📊 Handwriting Dataset Review")
        print("=" * 50)

        if report["valid"]:
            print(f"✅ Dataset is valid")
        else:
            print(f"⚠️  Dataset has issues")

        print(f"\n📍 Location: {self.base_dir}")
        print(f"📝 Total samples: {report['total_samples']}")

        if report['statistics']:
            stats = report['statistics']
            print(f"\n📊 Statistics:")
            print(f"  • Average text length: {stats['average_length']:.1f} chars")
            print(f"  • Total characters: {stats['total_characters']}")
            print(f"  • Image width: {stats['min_image_width']}-{stats['max_image_width']} px")
            print(f"  • Image height: {stats['min_image_height']}-{stats['max_image_height']} px")

        if report['issues']:
            print(f"\n⚠️  Issues ({len(report['issues'])}):")
            for issue in report['issues'][:10]:
                print(f"  • {issue}")
            if len(report['issues']) > 10:
                print(f"  ... and {len(report['issues']) - 10} more")

        print("=" * 50)


class SyntheticHandwritingGenerator:
    """Generate synthetic handwriting samples"""

    ACADEMIC_SENTENCES = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "Supervised learning uses labeled training data.",
        "Gradient descent minimizes the loss function.",
        "Convolutional neural networks excel at image recognition.",
        "Natural language processing enables machines to understand text.",
        "Recurrent neural networks handle sequential data.",
        "Transfer learning reuses pretrained models.",
        "Attention mechanisms help models focus on relevant features.",
        "BERT is a transformer model trained on large text corpora.",
        "Backpropagation computes gradients efficiently.",
        "Regularization prevents overfitting in machine learning.",
        "Decision trees use if-else rules for classification.",
        "Random forests combine multiple decision trees.",
        "K-means clustering partitions data into k groups.",
        "Support vector machines find optimal hyperplanes.",
        "Principal component analysis reduces dimensionality.",
        "Cross-validation helps evaluate model performance.",
        "Hyperparameter tuning optimizes model configuration.",
        "Ensemble methods combine multiple models for better accuracy."
    ]

    def __init__(self):
        self.fonts = self._find_fonts()

    def _find_fonts(self) -> List[str]:
        """Find available handwriting-style fonts"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:\\Windows\\Fonts\\Arial.ttf",
            "C:\\Windows\\Fonts\\Calibri.ttf",
            "C:\\Windows\\Fonts\\Candara.ttf",
        ]

        available = []
        for path in font_paths:
            if Path(path).exists():
                available.append(path)

        # Add default if nothing found
        if not available:
            available = [None]  # Use default font

        return available

    def generate_sentence(self) -> str:
        """Generate a random academic sentence"""
        return random.choice(self.ACADEMIC_SENTENCES)

    def render_text(self, text: str,
                   handwriting_style: str = "neat",
                   noise_level: float = 0.2,
                   image_width: int = 800,
                   image_height: int = 100) -> Image.Image:
        """
        Render text as a synthetic handwriting image.

        Args:
            text: Text to render
            handwriting_style: "neat", "messy", or "cursive"
            noise_level: Amount of noise to add (0-1)
            image_width: Width of output image
            image_height: Height of output image

        Returns:
            PIL Image
        """
        # Create blank image
        img = Image.new("L", (image_width, image_height), color=255)
        draw = ImageDraw.Draw(img)

        # Select font
        font_size = random.randint(24, 32)
        font_path = random.choice(self.fonts) if self.fonts else None

        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Calculate text position
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = len(text) * font_size // 2
            text_height = font_size

        x = random.randint(10, 20)
        y = (image_height - text_height) // 2 + random.randint(-5, 5)

        # Draw text with variations based on style
        if handwriting_style == "neat":
            # Consistent spacing
            draw.text((x, y), text, fill=random.randint(0, 40), font=font)

        elif handwriting_style == "messy":
            # Draw character by character with random offsets
            char_x = x
            for char in text:
                offset_y = random.randint(-3, 3)
                rotation = random.uniform(-2, 2)
                draw.text((char_x + random.randint(-1, 1), y + offset_y),
                         char, fill=random.randint(0, 60), font=font)
                try:
                    char_width = draw.textbbox((0, 0), char, font=font)[2]
                except:
                    char_width = font_size // 2
                char_x += char_width + random.randint(-2, 2)

        else:  # cursive
            # Simulate connected writing
            draw.text((x, y), text, fill=random.randint(0, 50), font=font)

        # Add noise
        if noise_level > 0:
            import numpy as np
            arr = np.array(img)
            noise = np.random.normal(0, noise_level * 50, arr.shape)
            noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(noisy)

        # Add slight blur
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        return img


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="IntelliGrade-H Dataset Collection Tool")
    parser.add_argument("action", choices=["add", "review", "synthetic", "augment", "split", "export", "validate"],
                       help="Action to perform")
    parser.add_argument("--image", help="Path to image file (for 'add')")
    parser.add_argument("--text", help="Transcription text (for 'add')")
    parser.add_argument("--student", help="Student ID (optional)")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--input-dir", help="Directory containing images (for batch add)")
    parser.add_argument("--output", default="datasets/training", help="Output directory")

    args = parser.parse_args()

    # Initialize dataset
    dataset = HandwritingDataset()

    if args.action == "add":
        if not args.image or not args.text:
            print("Error: --image and --text required for 'add'")
            return

        dataset.add_sample(args.image, args.text, args.student)
        dataset.review()

    elif args.action == "review":
        dataset.review()

    elif args.action == "synthetic":
        generator = SyntheticHandwritingGenerator()
        count = 0
        for i in range(args.n):
            sentence = generator.generate_sentence()
            img = generator.render_text(sentence)

            # Save temporary file
            temp_path = f"/tmp/syn_{i}.png"
            img.save(temp_path)

            if dataset.add_sample(temp_path, sentence, quality=0.7):
                count += 1

            # Clean up
            Path(temp_path).unlink()

        print(f"✅ Generated {count} synthetic samples")
        dataset.review()

    elif args.action == "augment":
        count = dataset.augment_dataset()
        print(f"✅ Created {count} augmented samples")

    elif args.action == "split":
        splits = dataset.split_dataset()
        print(f"Split complete: {splits}")

    elif args.action == "export":
        path = dataset.export_for_trocr(args.output)
        print(f"✅ Dataset exported to {path}")

    elif args.action == "validate":
        report = dataset.validate_dataset()
        if report["valid"]:
            print("✅ Dataset validation passed")
        else:
            print("❌ Dataset validation failed")
            for issue in report["issues"]:
                print(f"  • {issue}")


if __name__ == "__main__":
    main()