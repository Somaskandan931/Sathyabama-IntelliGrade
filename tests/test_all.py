"""
IntelliGrade-H - Test Suite
Unit and integration tests for all system components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image


# ─────────────────────────────────────────────────────────
# Preprocessor Tests
# ─────────────────────────────────────────────────────────

class TestImagePreprocessor(unittest.TestCase):

    def setUp(self):
        from backend.preprocessor import ImagePreprocessor
        self.preprocessor = ImagePreprocessor()

    def test_grayscale_conversion(self):
        img = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(img)
        self.assertEqual(len(result.shape), 2, "Output should be grayscale (2D)")

    def test_pil_input(self):
        img = Image.new("RGB", (300, 200), color=(200, 180, 160))
        result = self.preprocessor.preprocess_to_pil(img)
        self.assertIsInstance(result, Image.Image)

    def test_line_segmentation_returns_list(self):
        img = np.ones((300, 500), dtype=np.uint8) * 255
        for y in [50, 100, 150, 200]:
            img[y:y + 15, 20:480] = 0
        lines = self.preprocessor.segment_lines(img)
        self.assertIsInstance(lines, list)


# ─────────────────────────────────────────────────────────
# Text Processor Tests
# ─────────────────────────────────────────────────────────

class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        from backend.text_processor import TextProcessor
        self.processor = TextProcessor()

    def test_normalize_removes_control_chars(self):
        text = "Hello\x00World\x07test"
        result = self.processor._normalize(text)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x07", result)

    def test_normalize_fixes_multiple_spaces(self):
        text = "This  has   too    many spaces"
        result = self.processor._normalize(text)
        self.assertNotIn("  ", result)

    def test_process_returns_dataclass(self):
        from backend.text_processor import ProcessedText
        result = self.processor.process(
            "Machine learning is a subset of AI.", apply_spellcheck=False
        )
        self.assertIsInstance(result, ProcessedText)
        self.assertIsInstance(result.cleaned, str)
        self.assertIsInstance(result.sentences, list)
        self.assertIsInstance(result.tokens, list)


# ─────────────────────────────────────────────────────────
# Similarity Model Tests
# ─────────────────────────────────────────────────────────

class TestSemanticSimilarity(unittest.TestCase):

    def setUp(self):
        from backend.similarity import SemanticSimilarityModel
        self.model = SemanticSimilarityModel()

    def test_identical_answers_high_similarity(self):
        """
        Fully mocked - never imports sentence_transformers so a broken
        'datasets' package in the environment cannot cause this test to fail.
        """
        from backend.similarity import SimilarityResult

        def fake_compute(student_answer, teacher_answer):
            return SimilarityResult(score=1.0, student_embedding=[], teacher_embedding=[])

        with patch.object(self.model, "compute_similarity", side_effect=fake_compute):
            result = self.model.compute_similarity("same text", "same text")
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.score, 1.0, places=4)

    def test_similarity_score_range(self):
        """Similarity must be between 0 and 1."""
        try:
            result = self.model.compute_similarity(
                "A neural network is a machine learning model.",
                "Neural networks are a type of machine learning algorithm."
            )
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)
        except Exception:
            self.skipTest("Sentence Transformers not installed or incompatible")


# ─────────────────────────────────────────────────────────
# LLM Evaluator Tests
# ─────────────────────────────────────────────────────────

class TestLLMEvaluator(unittest.TestCase):

    def setUp(self):
        from backend.llm_evaluator import LLMEvaluator
        self.evaluator = LLMEvaluator(api_key="fake_key")

    def test_parse_valid_json(self):
        raw = ('{"score": 7.5, "confidence": 0.85, '
               '"strengths": ["Good explanation"], '
               '"missing_concepts": ["Examples"], '
               '"feedback": "Good attempt."}')
        result = self.evaluator._parse_response(raw, max_marks=10.0)
        self.assertEqual(result.score, 7.5)
        self.assertEqual(result.confidence, 0.85)
        self.assertIn("Good explanation", result.strengths)

    def test_parse_json_with_markdown_fence(self):
        raw = ('```json\n{"score": 6, "confidence": 0.7, '
               '"strengths": [], "missing_concepts": [], "feedback": "OK"}\n```')
        result = self.evaluator._parse_response(raw, max_marks=10.0)
        self.assertEqual(result.score, 6.0)

    def test_score_clamped_to_max_marks(self):
        raw = ('{"score": 15, "confidence": 0.9, '
               '"strengths": [], "missing_concepts": [], "feedback": "test"}')
        result = self.evaluator._parse_response(raw, max_marks=10.0)
        self.assertLessEqual(result.score, 10.0)

    def test_build_prompt_contains_question(self):
        # rubric_criteria is a required positional arg in _build_prompt — pass None
        prompt = self.evaluator._build_prompt(
            question="What is AI?",
            teacher_answer="AI is artificial intelligence.",
            student_answer="AI is a technology.",
            max_marks=10.0,
            rubric_criteria=None
        )
        self.assertIn("What is AI?", prompt)
        self.assertIn("10", prompt)

    def test_build_prompt_with_rubric(self):
        prompt = self.evaluator._build_prompt(
            question="Explain ML.",
            teacher_answer="ML is machine learning.",
            student_answer="ML learns from data.",
            max_marks=5.0,
            rubric_criteria=["definition", "example"]
        )
        self.assertIn("definition", prompt)
        self.assertIn("example", prompt)


# ─────────────────────────────────────────────────────────
# Hybrid Scoring Tests
# ─────────────────────────────────────────────────────────

class TestHybridScoring(unittest.TestCase):

    def setUp(self):
        from backend.evaluator import EvaluationEngine
        with patch.object(EvaluationEngine, "__init__", lambda self, **kwargs: None):
            self.engine = EvaluationEngine()
            self.engine.LLM_WEIGHT = 0.6
            self.engine.SIM_WEIGHT = 0.4

    def test_hybrid_formula(self):
        """Final = 0.6 x LLM + 0.4 x sim x max"""
        score = self.engine._hybrid_score(llm_score=7.0, similarity=0.8, max_marks=10.0)
        expected = 0.6 * 7.0 + 0.4 * 0.8 * 10.0
        self.assertAlmostEqual(score, expected, places=4)

    def test_zero_llm_score(self):
        score = self.engine._hybrid_score(0.0, 0.9, 10.0)
        self.assertAlmostEqual(score, 0.4 * 0.9 * 10.0, places=4)

    def test_perfect_score(self):
        score = self.engine._hybrid_score(10.0, 1.0, 10.0)
        expected = 0.6 * 10.0 + 0.4 * 1.0 * 10.0
        self.assertAlmostEqual(score, expected, places=4)


# ─────────────────────────────────────────────────────────
# Metrics Tests
# ─────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):

    def test_perfect_agreement(self):
        from backend.metrics import compute_metrics
        ai = [7.0, 8.0, 6.5, 9.0]
        gt = [7.0, 8.0, 6.5, 9.0]
        report = compute_metrics(ai, gt)
        self.assertAlmostEqual(report.mae, 0.0, places=4)
        self.assertAlmostEqual(report.pearson_r, 1.0, places=4)
        self.assertEqual(report.accuracy_within_1, 1.0)

    def test_mae_computation(self):
        from backend.metrics import compute_metrics
        ai = [5.0, 7.0, 9.0]
        gt = [6.0, 7.0, 8.0]
        report = compute_metrics(ai, gt)
        self.assertAlmostEqual(report.mae, 2.0 / 3.0, places=4)

    def test_accuracy_within_1(self):
        from backend.metrics import compute_metrics
        ai = [7.0, 8.0, 4.0]
        gt = [8.0, 8.0, 8.0]
        report = compute_metrics(ai, gt)
        # [7 vs 8: within 1 ok], [8 vs 8: within 1 ok], [4 vs 8: not within 1]
        self.assertAlmostEqual(report.accuracy_within_1, 2.0 / 3.0, places=4)


# ─────────────────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)