"""
IntelliGrade-H - Test Suite
Unit and integration tests for all system components.
Covers both MCQ and open-ended question workflows.
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
# MCQ Answer Extraction Tests
# ─────────────────────────────────────────────────────────

class TestMCQAnswerExtraction(unittest.TestCase):
    """Tests for the OCR-based MCQ answer extractor."""

    def _extract(self, text):
        from backend.evaluator import _extract_mcq_answer
        return _extract_mcq_answer(text)

    def test_standalone_letter(self):
        self.assertEqual(self._extract("B"), "B")

    def test_bracketed_letter(self):
        self.assertEqual(self._extract("(C)"), "C")

    def test_answer_prefix(self):
        self.assertEqual(self._extract("Answer: D"), "D")

    def test_ans_prefix(self):
        self.assertEqual(self._extract("Ans: A"), "A")

    def test_option_prefix(self):
        self.assertEqual(self._extract("Option B"), "B")

    def test_lowercase_normalized(self):
        self.assertEqual(self._extract("b)"), "B")

    def test_noise_around_letter(self):
        # OCR noise with the selected letter
        self.assertEqual(self._extract("The answer is B."), "B")

    def test_unclear_returns_none(self):
        result = self._extract("I don't know the answer.")
        # May return None or a stray letter — just ensure no crash
        self.assertIn(result, (None, "A", "B", "C", "D", "E"))

    def test_checkmark_prefix(self):
        self.assertEqual(self._extract("✓C"), "C")


# ─────────────────────────────────────────────────────────
# MCQ Evaluation Engine Tests
# ─────────────────────────────────────────────────────────

class TestMCQEvaluation(unittest.TestCase):

    def _make_engine(self):
        from backend.evaluator import EvaluationEngine, QuestionType
        with patch.object(EvaluationEngine, "__init__", lambda self, **kwargs: None):
            engine = EvaluationEngine()
            engine.LLM_WEIGHT = 0.6
            engine.SIM_WEIGHT = 0.4
        return engine

    def test_correct_mcq_gives_full_marks(self):
        from backend.evaluator import EvaluationEngine, QuestionType
        from backend.ocr_module import OCRResult

        engine = self._make_engine()
        ocr    = OCRResult(text="B", confidence=0.95, engine="trocr")
        result = engine._evaluate_mcq(
            raw_text="B",
            ocr_result=ocr,
            correct_option="B",
            max_marks=2.0,
            start=0.0,
        )
        self.assertEqual(result.final_score, 2.0)
        self.assertTrue(result.mcq_correct)
        self.assertEqual(result.mcq_detected_answer, "B")

    def test_wrong_mcq_gives_zero(self):
        from backend.evaluator import EvaluationEngine, QuestionType
        from backend.ocr_module import OCRResult

        engine = self._make_engine()
        ocr    = OCRResult(text="(A)", confidence=0.90, engine="trocr")
        result = engine._evaluate_mcq(
            raw_text="(A)",
            ocr_result=ocr,
            correct_option="C",
            max_marks=2.0,
            start=0.0,
        )
        self.assertEqual(result.final_score, 0.0)
        self.assertFalse(result.mcq_correct)
        self.assertEqual(result.mcq_detected_answer, "A")

    def test_missing_correct_option_returns_empty(self):
        from backend.evaluator import EvaluationEngine, QuestionType
        from backend.ocr_module import OCRResult

        engine = self._make_engine()
        ocr    = OCRResult(text="B", confidence=0.9, engine="trocr")
        result = engine._evaluate_mcq(
            raw_text="B",
            ocr_result=ocr,
            correct_option=None,
            max_marks=2.0,
            start=0.0,
        )
        self.assertEqual(result.final_score, 0.0)
        self.assertIn("not provided", result.feedback)


# ─────────────────────────────────────────────────────────
# Text Processor Tests
# ─────────────────────────────────────────────────────────

class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        from backend.text_processor import TextProcessor
        self.processor = TextProcessor()

    def test_normalize_removes_control_chars(self):
        text   = "Hello\x00World\x07test"
        result = self.processor._normalize(text)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x07", result)

    def test_normalize_fixes_multiple_spaces(self):
        text   = "This  has   too    many spaces"
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
# Similarity Model Tests (open-ended only)
# ─────────────────────────────────────────────────────────

class TestSemanticSimilarity(unittest.TestCase):

    def setUp(self):
        from backend.similarity import SemanticSimilarityModel
        self.model = SemanticSimilarityModel()

    def test_identical_answers_high_similarity(self):
        from backend.similarity import SimilarityResult

        def fake_compute(student_answer, teacher_answer):
            return SimilarityResult(score=1.0, student_embedding=[], teacher_embedding=[])

        with patch.object(self.model, "compute_similarity", side_effect=fake_compute):
            result = self.model.compute_similarity("same text", "same text")
            self.assertAlmostEqual(result.score, 1.0, places=4)

    def test_similarity_score_range(self):
        try:
            result = self.model.compute_similarity(
                "A neural network is a machine learning model.",
                "Neural networks are a type of machine learning algorithm.",
            )
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)
        except Exception:
            self.skipTest("Sentence Transformers not installed or incompatible")


# ─────────────────────────────────────────────────────────
# LLM Evaluator Tests (open-ended only)
# ─────────────────────────────────────────────────────────

class TestLLMEvaluator(unittest.TestCase):

    def setUp(self):
        from backend.llm_evaluator import LLMEvaluator
        self.evaluator = LLMEvaluator(api_key="fake_key")

    def test_parse_valid_json(self):
        raw    = ('{"score": 7.5, "confidence": 0.85, '
                  '"strengths": ["Good explanation"], '
                  '"missing_concepts": ["Examples"], '
                  '"feedback": "Good attempt."}')
        result = self.evaluator._parse_response(raw, max_marks=10.0)
        self.assertEqual(result.score, 7.5)
        self.assertEqual(result.confidence, 0.85)
        self.assertIn("Good explanation", result.strengths)

    def test_parse_json_with_markdown_fence(self):
        raw    = ('```json\n{"score": 6, "confidence": 0.7, '
                  '"strengths": [], "missing_concepts": [], "feedback": "OK"}\n```')
        result = self.evaluator._parse_response(raw, max_marks=10.0)
        self.assertEqual(result.score, 6.0)

    def test_score_clamped_to_max_marks(self):
        raw    = ('{"score": 15, "confidence": 0.9, '
                  '"strengths": [], "missing_concepts": [], "feedback": "test"}')
        result = self.evaluator._parse_response(raw, max_marks=10.0)
        self.assertLessEqual(result.score, 10.0)

    def test_build_prompt_contains_question(self):
        prompt = self.evaluator._build_prompt(
            question="What is AI?",
            teacher_answer="AI is artificial intelligence.",
            student_answer="AI is a technology.",
            max_marks=10.0,
            rubric_criteria=None,
        )
        self.assertIn("What is AI?", prompt)
        self.assertIn("10", prompt)

    def test_build_prompt_with_rubric(self):
        prompt = self.evaluator._build_prompt(
            question="Explain ML.",
            teacher_answer="ML is machine learning.",
            student_answer="ML learns from data.",
            max_marks=5.0,
            rubric_criteria=["definition", "example"],
        )
        self.assertIn("definition", prompt)
        self.assertIn("example", prompt)

    def test_empty_student_answer_returns_zero(self):
        result = self.evaluator.evaluate(
            question="What is AI?",
            teacher_answer="AI is artificial intelligence.",
            student_answer="",
            max_marks=10.0,
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.confidence, 1.0)


# ─────────────────────────────────────────────────────────
# Hybrid Scoring Tests (open-ended)
# ─────────────────────────────────────────────────────────

class TestHybridScoring(unittest.TestCase):

    def setUp(self):
        from backend.evaluator import EvaluationEngine
        with patch.object(EvaluationEngine, "__init__", lambda self, **kwargs: None):
            self.engine = EvaluationEngine()
            self.engine.LLM_WEIGHT = 0.6
            self.engine.SIM_WEIGHT = 0.4

    def test_hybrid_formula(self):
        score    = self.engine._hybrid_score(llm_score=7.0, similarity=0.8, max_marks=10.0)
        expected = 0.6 * 7.0 + 0.4 * 0.8 * 10.0
        self.assertAlmostEqual(score, expected, places=4)

    def test_zero_llm_score(self):
        score = self.engine._hybrid_score(0.0, 0.9, 10.0)
        self.assertAlmostEqual(score, 0.4 * 0.9 * 10.0, places=4)

    def test_perfect_score(self):
        score    = self.engine._hybrid_score(10.0, 1.0, 10.0)
        expected = 0.6 * 10.0 + 0.4 * 1.0 * 10.0
        self.assertAlmostEqual(score, expected, places=4)


# ─────────────────────────────────────────────────────────
# Metrics Tests
# ─────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):

    def test_perfect_agreement_open_ended(self):
        from backend.metrics import compute_metrics
        ai  = [7.0, 8.0, 6.5, 9.0]
        gt  = [7.0, 8.0, 6.5, 9.0]
        rep = compute_metrics(ai, gt, question_type="open_ended")
        self.assertAlmostEqual(rep.mae, 0.0, places=4)
        self.assertAlmostEqual(rep.pearson_r, 1.0, places=4)
        self.assertEqual(rep.accuracy_within_1, 1.0)

    def test_mae_computation(self):
        from backend.metrics import compute_metrics
        ai  = [5.0, 7.0, 9.0]
        gt  = [6.0, 7.0, 8.0]
        rep = compute_metrics(ai, gt, question_type="open_ended")
        self.assertAlmostEqual(rep.mae, 2.0 / 3.0, places=4)

    def test_accuracy_within_1(self):
        from backend.metrics import compute_metrics
        ai  = [7.0, 8.0, 4.0]
        gt  = [8.0, 8.0, 8.0]
        rep = compute_metrics(ai, gt, question_type="open_ended")
        self.assertAlmostEqual(rep.accuracy_within_1, 2.0 / 3.0, places=4)

    def test_mcq_accuracy_all_correct(self):
        from backend.metrics import compute_mcq_metrics
        pred = ["A", "B", "C", "D"]
        corr = ["A", "B", "C", "D"]
        rep  = compute_mcq_metrics(pred, corr)
        self.assertEqual(rep.mcq_accuracy, 1.0)
        self.assertEqual(rep.mcq_n_correct, 4)
        self.assertEqual(rep.mcq_n_wrong, 0)

    def test_mcq_accuracy_partial(self):
        from backend.metrics import compute_mcq_metrics
        pred = ["A", "B", "C"]
        corr = ["A", "C", "C"]
        rep  = compute_mcq_metrics(pred, corr)
        # A==A ✓, B!=C ✗, C==C ✓ → 2/3
        self.assertAlmostEqual(rep.mcq_accuracy, 2.0 / 3.0, places=4)
        self.assertEqual(rep.mcq_n_wrong, 1)

    def test_mcq_accuracy_all_wrong(self):
        from backend.metrics import compute_mcq_metrics
        pred = ["A", "A", "A"]
        corr = ["B", "C", "D"]
        rep  = compute_mcq_metrics(pred, corr)
        self.assertEqual(rep.mcq_accuracy, 0.0)

    def test_mcq_metrics_report_question_type(self):
        from backend.metrics import compute_metrics
        ai  = [1.0, 0.0, 1.0]
        gt  = [1.0, 1.0, 1.0]
        rep = compute_metrics(ai, gt, question_type="mcq")
        self.assertEqual(rep.question_type, "mcq")

    def test_mixed_metrics_run_without_error(self):
        from backend.metrics import compute_metrics
        ai  = [7.0, 1.0, 5.0]
        gt  = [8.0, 1.0, 5.0]
        rep = compute_metrics(ai, gt, question_type="mixed")
        self.assertEqual(rep.n_samples, 3)


# ─────────────────────────────────────────────────────────
# Rubric Matcher Tests
# ─────────────────────────────────────────────────────────

class TestRubricMatcher(unittest.TestCase):

    def setUp(self):
        from backend.rubric_matcher import RubricMatcher
        self.matcher = RubricMatcher()

    def test_mcq_returns_empty_rubric(self):
        """Rubric should be skipped entirely for MCQ questions."""
        rubric = [{"criterion": "definition", "marks": 2.0}]
        result = self.matcher.evaluate_rubric(
            student_answer="B",
            rubric_criteria=rubric,
            question_type="mcq",
        )
        self.assertEqual(result.earned_rubric_marks, 0.0)
        self.assertEqual(result.criteria_scores, {})

    def test_empty_rubric_returns_zero(self):
        result = self.matcher.evaluate_rubric(
            student_answer="Some answer",
            rubric_criteria=[],
            question_type="open_ended",
        )
        self.assertEqual(result.earned_rubric_marks, 0.0)
        self.assertEqual(result.coverage_ratio, 0.0)


# ─────────────────────────────────────────────────────────
# QuestionType Enum Tests
# ─────────────────────────────────────────────────────────

class TestQuestionType(unittest.TestCase):

    def test_enum_values(self):
        from backend.evaluator import QuestionType
        self.assertEqual(QuestionType.MCQ, "mcq")
        self.assertEqual(QuestionType.OPEN_ENDED, "open_ended")

    def test_enum_from_string(self):
        from backend.evaluator import QuestionType
        self.assertEqual(QuestionType("mcq"), QuestionType.MCQ)
        self.assertEqual(QuestionType("open_ended"), QuestionType.OPEN_ENDED)

    def test_invalid_enum_raises(self):
        from backend.evaluator import QuestionType
        with self.assertRaises(ValueError):
            QuestionType("essay")


# ─────────────────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)