"""
IntelliGrade-H: Dataset Preparation Utilities
Tools for collecting, augmenting, and structuring training datasets.
"""

import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


# ─────────────────────────────────────────────
# OCR Dataset Builder
# ─────────────────────────────────────────────

class OCRDatasetBuilder:
    """
    Build a TrOCR fine-tuning dataset from raw scanned images.

    Expected input structure:
        raw_data/
            img_001.jpg
            img_001.txt   ← transcription file (same stem)
            img_002.jpg
            img_002.txt
            ...

    Output CSV:
        image_path, transcription
    """

    def __init__(self, raw_dir: str, output_csv: str = "datasets/ocr_dataset.csv"):
        self.raw_dir = Path(raw_dir)
        self.output_csv = output_csv

    def build(self) -> int:
        """Scan raw_dir and write CSV. Returns number of pairs found."""
        pairs = []
        for img_path in sorted(self.raw_dir.glob("*.jpg")):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                transcription = txt_path.read_text(encoding="utf-8").strip()
                if transcription:
                    pairs.append((str(img_path), transcription))

        for img_path in sorted(self.raw_dir.glob("*.png")):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                transcription = txt_path.read_text(encoding="utf-8").strip()
                if transcription:
                    pairs.append((str(img_path), transcription))

        os.makedirs(Path(self.output_csv).parent, exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "transcription"])
            writer.writerows(pairs)

        print(f"OCR dataset: {len(pairs)} pairs → {self.output_csv}")
        return len(pairs)


# ─────────────────────────────────────────────
# Answer-Pair Dataset Builder
# ─────────────────────────────────────────────

class AnswerPairDatasetBuilder:
    """
    Build a similarity/evaluation dataset from teacher-graded answer pairs.

    CSV format:
        question, student_answer, teacher_answer, score, max_score
    """

    def __init__(self, output_csv: str = "datasets/answer_pairs.csv"):
        self.output_csv = output_csv
        self._rows: List[dict] = []

    def add(
        self,
        question: str,
        student_answer: str,
        teacher_answer: str,
        score: float,
        max_score: float,
        student_id: str = "",
    ):
        self._rows.append({
            "question": question,
            "student_answer": student_answer,
            "teacher_answer": teacher_answer,
            "score": score,
            "max_score": max_score,
            "student_id": student_id,
        })

    def load_from_json(self, json_path: str):
        """Load from a JSON array of answer pair objects."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.add(**item)

    def save(self) -> str:
        os.makedirs(Path(self.output_csv).parent, exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._rows[0].keys())
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"Answer-pair dataset: {len(self._rows)} rows → {self.output_csv}")
        return self.output_csv

    def train_test_split(
        self, test_ratio: float = 0.15
    ) -> Tuple[str, str]:
        """Split into train/test CSVs and save both."""
        random.shuffle(self._rows)
        split = int(len(self._rows) * (1 - test_ratio))
        train_rows, test_rows = self._rows[:split], self._rows[split:]

        def _write(rows, path):
            os.makedirs(Path(path).parent, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

        train_path = self.output_csv.replace(".csv", "_train.csv")
        test_path = self.output_csv.replace(".csv", "_test.csv")
        _write(train_rows, train_path)
        _write(test_rows, test_path)
        print(f"Train: {len(train_rows)} | Test: {len(test_rows)}")
        return train_path, test_path


# ─────────────────────────────────────────────
# Anonymiser
# ─────────────────────────────────────────────

class DataAnonymiser:
    """
    Strip personally identifiable information from student data
    before use in training or analysis.
    """

    REDACT_FIELDS = {"name", "email", "phone", "student_name", "roll_number"}

    def anonymise_csv(self, input_csv: str, output_csv: str):
        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        redact = [f for f in fieldnames if f.lower() in self.REDACT_FIELDS]
        for row in rows:
            for field in redact:
                row[field] = "[REDACTED]"

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Anonymised {len(redact)} fields in {len(rows)} rows → {output_csv}")

    def anonymise_image_filenames(self, image_dir: str, output_dir: str):
        """Rename image files to sequential IDs, removing student identifiers."""
        src = Path(image_dir)
        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)
        mapping = {}
        for i, img in enumerate(sorted(src.glob("*"))):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                new_name = f"img_{i:05d}{img.suffix}"
                shutil.copy2(img, dst / new_name)
                mapping[img.name] = new_name
        mapping_path = dst / "filename_mapping.json"
        mapping_path.write_text(json.dumps(mapping, indent=2))
        print(f"Renamed {len(mapping)} images. Mapping saved to {mapping_path}")
        return mapping


# ─────────────────────────────────────────────
# Sample Data Generator (for testing)
# ─────────────────────────────────────────────

class SampleDataGenerator:
    """Generate synthetic answer-pair data for unit testing."""

    QUESTIONS = [
        "Explain the concept of machine learning.",
        "What is backpropagation in neural networks?",
        "Describe the difference between supervised and unsupervised learning.",
        "What is the purpose of an activation function?",
        "Explain overfitting and how to prevent it.",
    ]

    TEACHER_ANSWERS = [
        "Machine learning is a subset of AI that enables systems to learn from data automatically.",
        "Backpropagation computes gradients via the chain rule and updates weights using gradient descent.",
        "Supervised learning uses labelled data; unsupervised learning finds patterns without labels.",
        "Activation functions introduce non-linearity, allowing networks to learn complex patterns.",
        "Overfitting occurs when a model learns noise. Prevention: dropout, regularisation, more data.",
    ]

    def generate(self, n_samples: int = 200, output_csv: str = "datasets/sample_pairs.csv") -> str:
        builder = AnswerPairDatasetBuilder(output_csv)
        for i in range(n_samples):
            q_idx = i % len(self.QUESTIONS)
            quality = random.uniform(0.3, 1.0)
            # Simulate partial answers by truncating the teacher answer
            words = self.TEACHER_ANSWERS[q_idx].split()
            cutoff = max(3, int(len(words) * quality))
            student_ans = " ".join(words[:cutoff])
            score = round(quality * 10, 1)
            builder.add(
                question=self.QUESTIONS[q_idx],
                student_answer=student_ans,
                teacher_answer=self.TEACHER_ANSWERS[q_idx],
                score=score,
                max_score=10.0,
                student_id=f"STU{i:04d}",
            )
        return builder.save()


if __name__ == "__main__":
    gen = SampleDataGenerator()
    path = gen.generate(n_samples=300)
    print(f"Generated sample dataset: {path}")
