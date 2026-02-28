"""
IntelliGrade-H - Semantic Similarity Module
Uses Sentence-BERT to compute how close a student answer is to the model answer.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    score: float          # 0.0 – 1.0 cosine similarity
    student_embedding: list
    teacher_embedding: list


class SemanticSimilarityModel:
    """
    Wraps SentenceTransformer to produce cosine-similarity scores
    between student and teacher answers.

    Default model: all-MiniLM-L6-v2 (fast, accurate)
    For better accuracy use: all-mpnet-base-v2
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("SentenceTransformer loaded.")

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def compute_similarity(
        self,
        student_answer: str,
        teacher_answer: str
    ) -> SimilarityResult:
        """
        Returns cosine similarity between student and teacher embeddings.
        Score range: 0.0 (unrelated) to 1.0 (identical meaning).
        """
        self._load()
        from sentence_transformers import util

        embeddings = self._model.encode(
            [student_answer, teacher_answer],
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        # clamp to [0, 1]
        cosine_score = max(0.0, min(1.0, cosine_score))

        return SimilarityResult(
            score=cosine_score,
            student_embedding=embeddings[0].cpu().tolist(),
            teacher_embedding=embeddings[1].cpu().tolist()
        )

    def compute_sentence_level(
        self,
        student_answer: str,
        teacher_answer: str
    ) -> dict:
        """
        Compute similarity at sentence level for granular feedback.
        Returns per-sentence similarity and best matches.
        """
        self._load()
        from sentence_transformers import util
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
            student_sents = [s.text for s in nlp(student_answer).sents]
            teacher_sents = [s.text for s in nlp(teacher_answer).sents]
        except Exception:
            student_sents = student_answer.split(". ")
            teacher_sents = teacher_answer.split(". ")

        if not student_sents or not teacher_sents:
            return {"overall": 0.0, "sentence_scores": []}

        s_embs = self._model.encode(student_sents, convert_to_tensor=True, normalize_embeddings=True)
        t_embs = self._model.encode(teacher_sents, convert_to_tensor=True, normalize_embeddings=True)

        cosine_matrix = util.cos_sim(s_embs, t_embs)

        sentence_scores = []
        for i, s_sent in enumerate(student_sents):
            best_score = cosine_matrix[i].max().item()
            best_teacher_idx = cosine_matrix[i].argmax().item()
            sentence_scores.append({
                "student_sentence": s_sent,
                "best_match_teacher": teacher_sents[best_teacher_idx],
                "similarity": round(max(0.0, best_score), 3)
            })

        overall = float(cosine_matrix.max(dim=1).values.mean().item())

        return {
            "overall": round(max(0.0, min(1.0, overall)), 3),
            "sentence_scores": sentence_scores
        }

    def fine_tune(
        self,
        training_data: list,
        output_dir: str,
        epochs: int = 4,
        batch_size: int = 16
    ):
        """
        Fine-tune the similarity model on QA scoring pairs.

        training_data: list of dicts with keys:
            - question: str
            - student_answer: str
            - teacher_answer: str
            - score: float  (normalized 0-1)

        Trains using CosineSimilarityLoss.
        """
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        self._load()

        examples = []
        for item in training_data:
            examples.append(InputExample(
                texts=[item["student_answer"], item["teacher_answer"]],
                label=float(item["score"])
            ))

        dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = losses.CosineSimilarityLoss(self._model)

        logger.info(f"Fine-tuning on {len(examples)} examples for {epochs} epochs...")
        self._model.fit(
            train_objectives=[(dataloader, loss)],
            epochs=epochs,
            output_path=output_dir,
            show_progress_bar=True
        )
        logger.info(f"Fine-tuned model saved to {output_dir}")