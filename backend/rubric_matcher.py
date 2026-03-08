"""
IntelliGrade-H - Rubric Matching Module
Detects whether rubric criteria are present in the student answer.

NOTE: Rubric matching is applicable only to OPEN-ENDED questions.
      For MCQ questions, grading is binary (correct/incorrect) — rubrics are not used.

Uses zero-shot classification (no training required) or fine-tuned BERT.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RubricResult:
    criteria_scores: dict         # {criterion: 0 or 1}
    total_rubric_marks: float
    earned_rubric_marks: float
    coverage_ratio: float         # 0.0 – 1.0


class RubricMatcher:
    """
    Two-mode rubric matching for open-ended questions:
    1. Zero-shot NLI (default, no training) — uses facebook/bart-large-mnli
    2. Fine-tuned BERT classifier (if training data is available)
    """

    NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, use_zero_shot: bool = True,
                 finetuned_model_path: Optional[str] = None):
        self.use_zero_shot     = use_zero_shot
        self._finetuned_path   = finetuned_model_path
        self._zero_shot_pipeline = None
        self._finetuned_model  = None

    # ─────────────────────────────────────────────────────
    # Lazy loaders
    # ─────────────────────────────────────────────────────

    def _get_zero_shot_pipeline(self):
        if self._zero_shot_pipeline is None:
            from transformers import pipeline
            logger.info("Loading zero-shot NLI pipeline for rubric matching...")
            self._zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
        return self._zero_shot_pipeline

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def evaluate_rubric(
        self,
        student_answer: str,
        rubric_criteria: list,
        threshold: float = 0.5,
        question_type: str = "open_ended",
    ) -> RubricResult:
        """
        Check which rubric criteria are covered in the student answer.

        IMPORTANT: This method should only be called for open-ended questions.
        For MCQ questions, pass question_type="mcq" to receive an empty RubricResult.

        rubric_criteria example:
        [
            {"criterion": "definition of machine learning", "marks": 2.0},
            {"criterion": "example of supervised learning", "marks": 1.5},
        ]
        """
        # Rubric not applicable to deterministic question types
        if question_type in ("mcq", "true_false", "numerical"):
            logger.info("Rubric matching skipped for %s question.", question_type)
            return RubricResult(
                criteria_scores={},
                total_rubric_marks=0.0,
                earned_rubric_marks=0.0,
                coverage_ratio=0.0,
            )

        if not rubric_criteria:
            return RubricResult(
                criteria_scores={},
                total_rubric_marks=0.0,
                earned_rubric_marks=0.0,
                coverage_ratio=0.0,
            )

        criteria_labels = [c["criterion"] for c in rubric_criteria]
        marks_map   = {c["criterion"]: c["marks"] for c in rubric_criteria}
        total_marks = sum(c["marks"] for c in rubric_criteria)

        if self.use_zero_shot:
            scores = self._zero_shot_check(student_answer, criteria_labels, threshold)
        else:
            scores = self._finetuned_check(student_answer, criteria_labels)

        earned   = sum(marks_map[c] for c, present in scores.items() if present)
        coverage = earned / total_marks if total_marks > 0 else 0.0

        return RubricResult(
            criteria_scores={c: (1 if v else 0) for c, v in scores.items()},
            total_rubric_marks=total_marks,
            earned_rubric_marks=earned,
            coverage_ratio=round(coverage, 3),
        )

    # ─────────────────────────────────────────────────────
    # Zero-shot checking via NLI
    # ─────────────────────────────────────────────────────

    def _zero_shot_check(self, student_answer: str, criteria: list, threshold: float) -> dict:
        pipe = self._get_zero_shot_pipeline()
        result = pipe(student_answer, candidate_labels=criteria, multi_label=True)
        label_to_score = dict(zip(result["labels"], result["scores"]))
        return {c: label_to_score.get(c, 0.0) >= threshold for c in criteria}

    # ─────────────────────────────────────────────────────
    # Fine-tuned BERT (optional)
    # ─────────────────────────────────────────────────────

    def _finetuned_check(self, student_answer: str, criteria: list) -> dict:
        return self._zero_shot_check(student_answer, criteria, threshold=0.5)

    def train_finetuned(self, training_data: list, output_dir: str, epochs: int = 3):
        """
        Fine-tune BERT for rubric detection (open-ended only).

        training_data: list of dicts:
        {
            "answer": str,
            "criterion": str,
            "present": 0 or 1
        }
        """
        from transformers import (
            BertTokenizer, BertForSequenceClassification,
            Trainer, TrainingArguments,
        )
        from torch.utils.data import Dataset
        import torch

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        class RubricDataset(Dataset):
            def __init__(self, data, tokenizer):
                self.data      = data
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                text = f"{item['criterion']} [SEP] {item['answer']}"
                enc  = self.tokenizer(
                    text, truncation=True, padding="max_length",
                    max_length=256, return_tensors="pt"
                )
                return {
                    "input_ids":      enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels":         torch.tensor(item["present"], dtype=torch.long),
                }

        dataset = RubricDataset(training_data, tokenizer)
        model   = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=200,
            logging_steps=50,
        )

        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Rubric classifier saved to %s", output_dir)