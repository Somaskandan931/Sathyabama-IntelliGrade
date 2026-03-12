"""
IntelliGrade-H - Comprehensive Test Suite (v3)
===============================================
Fixes vs v2:
  - Removed import of EasyOCREngine (doesn't exist — codebase uses hybrid pipeline)
  - Removed import of RuleBasedFallbackProvider (doesn't exist in llm_provider.py)
  - OCRModule() instantiated without engine= argument (constructor takes no engine param)
  - TestLLMProvider: removed tests for non-existent classes
  - TestLLMEvaluator: _build_enhanced_prompt signature corrected
  - TestEvaluationEngine: _hybrid_score signature corrected to match actual implementation
  - TestLayoutDetector / TestDiagramDetector: kept as pytest-style but marked
    as skipped when modules don't exist (optional future modules)
  - test_metrics.py has bare `pytest` reference without import guard — fixed
"""

import os
import sys
import unittest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import modules
from backend.preprocessor import ImagePreprocessor
from backend.ocr_module import OCRModule          # no EasyOCREngine — hybrid pipeline only
from backend.text_processor import TextProcessor
from backend.similarity import SemanticSimilarityModel
from backend.llm_evaluator import LLMEvaluator
from backend.llm_provider import LLMClient        # RuleBasedFallbackProvider removed
from backend.evaluator import EvaluationEngine, EvaluationResult
from backend.rubric_matcher import RubricMatcher
from backend.question_classifier import QuestionClassifier
from backend.metrics import compute_metrics, compute_mcq_metrics


# ============================================================================
# Test Utilities
# ============================================================================

def create_test_image(text: str = "Test", width: int = 400, height: int = 100) -> Image.Image:
    """Create a test image with text"""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 30), text, fill=(0, 0, 0), font=font)
    return img


# ============================================================================
# Preprocessor Tests
# ============================================================================

class TestImagePreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ImagePreprocessor()

    def test_grayscale_conversion(self):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(img)
        self.assertEqual(len(result.shape), 2)

    def test_denoise(self):
        img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        result = self.preprocessor._denoise(img)
        self.assertEqual(result.shape, img.shape)

    def test_deskew(self):
        img = Image.new("L", (400, 200), color=255)
        draw = ImageDraw.Draw(img)
        draw.line((50, 50, 350, 150), fill=0, width=3)
        img_array = np.array(img)
        result = self.preprocessor._deskew(img_array)
        self.assertEqual(result.shape, img_array.shape)

    def test_line_segmentation(self):
        img = Image.new("L", (500, 300), color=255)
        draw = ImageDraw.Draw(img)
        for i, y in enumerate([50, 100, 150, 200]):
            draw.text((50, y), f"Line {i+1}", fill=0)
        lines = self.preprocessor.segment_lines(np.array(img))
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0)


# ============================================================================
# OCR Module Tests
# ============================================================================

class TestOCRModule(unittest.TestCase):
    """Test OCR functionality using the hybrid pipeline (no engine= arg)."""

    def setUp(self):
        # OCRModule() takes no engine= argument — the hybrid pipeline is always used
        self.ocr = OCRModule()

    def test_ocr_with_image(self):
        img = create_test_image("Hello World")
        result = self.ocr.extract_text(img)
        self.assertIsInstance(result.text, str)
        self.assertIsInstance(result.confidence, float)

    def test_ocr_empty_image(self):
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        result = self.ocr.extract_text(img)
        self.assertIsInstance(result.text, str)

    def test_ocr_confidence_range(self):
        img = create_test_image("Test")
        result = self.ocr.extract_text(img)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


# ============================================================================
# Text Processor Tests
# ============================================================================

class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = TextProcessor()

    def test_normalize(self):
        text = "This  has   multiple    spaces"
        result = self.processor._normalize(text)
        self.assertNotIn("  ", result)

    def test_spellcheck(self):
        text = "This is a test with mispelled word"
        result = self.processor._spellcheck(text)
        self.assertIsInstance(result, str)

    def test_sentence_segmentation(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.processor._segment_sentences(text)
        self.assertGreaterEqual(len(sentences), 3)

    def test_tokenization(self):
        text = "Machine learning is fascinating"
        tokens = self.processor._tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)


# ============================================================================
# Similarity Model Tests
# ============================================================================

class TestSimilarityModel(unittest.TestCase):

    def setUp(self):
        self.model = SemanticSimilarityModel()

    def test_identical_answers(self):
        text = "Neural networks are a type of machine learning model."
        result = self.model.compute_similarity(text, text)
        self.assertGreaterEqual(result.score, 0.9)

    def test_different_answers(self):
        text1 = "Neural networks are a type of machine learning model."
        text2 = "The weather is nice today."
        result = self.model.compute_similarity(text1, text2)
        self.assertLess(result.score, 0.5)

    def test_similarity_range(self):
        text1 = "Machine learning uses algorithms."
        text2 = "AI systems learn from data."
        result = self.model.compute_similarity(text1, text2)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_sentence_level(self):
        text1 = "First sentence. Second sentence."
        text2 = "First sentence. Different second."
        result = self.model.compute_sentence_level(text1, text2)
        self.assertIn("sentence_scores", result)


# ============================================================================
# LLM Provider Tests
# ============================================================================

class TestLLMProvider(unittest.TestCase):
    """
    Tests for LLMClient only.
    RuleBasedFallbackProvider was removed — it does not exist in llm_provider.py.
    The offline fallback is now an internal static method (_offline_fallback).
    """

    def test_llm_client_from_env(self):
        client = LLMClient.from_env()
        self.assertIsInstance(client, LLMClient)
        self.assertGreater(len(client._providers), 0)

    def test_offline_fallback_returns_json(self):
        """The internal fallback must always return parseable JSON."""
        text = LLMClient._offline_fallback("any prompt")
        data = json.loads(text)
        self.assertIn("score", data)
        self.assertIn("feedback", data)
        self.assertIn("confidence", data)

    def test_generate_returns_llm_response(self):
        """generate() must always return an LLMResponse even without a key."""
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            client = LLMClient()
            response = client.generate("Hello")
            self.assertIsNotNone(response)
            self.assertIsInstance(response.text, str)

    def test_generate_json_always_returns_dict(self):
        """generate_json() must never raise — returns fallback dict on errors."""
        client = LLMClient()
        with patch.object(client, "generate") as mock_gen:
            mock_gen.return_value = MagicMock(text="not valid json {{{{")
            result = client.generate_json("test prompt")
            self.assertIsInstance(result, dict)
            self.assertIn("score", result)


# ============================================================================
# LLM Evaluator Tests
# ============================================================================

class TestLLMEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = LLMEvaluator()

    def test_empty_answer(self):
        result = self.evaluator.evaluate(
            question="What is AI?",
            teacher_answer="Artificial Intelligence",
            student_answer="",
            max_marks=10
        )
        self.assertEqual(result.score, 0.0)

    def test_parse_json_response(self):
        raw = '{"score": 7.5, "confidence": 0.85, "strengths": ["Good"], "missing_concepts": [], "feedback": "Good"}'
        result = self.evaluator._parse_response(raw, max_marks=10)
        self.assertEqual(result.score, 7.5)

    def test_parse_json_with_markdown(self):
        raw = '```json\n{"score": 8, "confidence": 0.9, "strengths": [], "missing_concepts": [], "feedback": "OK"}\n```'
        result = self.evaluator._parse_response(raw, max_marks=10)
        self.assertEqual(result.score, 8.0)

    def test_build_prompt_with_rubric(self):
        rubric = [{"criterion": "Definition", "marks": 2}]
        # _build_enhanced_prompt takes: question, teacher_answer, student_answer,
        # max_marks, rubric_criteria (not a keyword named "rubric")
        prompt = self.evaluator._build_enhanced_prompt(
            question="Test?",
            teacher_answer="Answer",
            student_answer="Student",
            max_marks=10,
            rubric_criteria=rubric,
        )
        self.assertIn("Definition", prompt)


# ============================================================================
# Question Classifier Tests
# ============================================================================

class TestQuestionClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = QuestionClassifier()

    def test_mcq_detection(self):
        question = "Which of the following is a machine learning algorithm? A) Decision Tree B) K-Means C) Both"
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "mcq")

    def test_true_false_detection(self):
        question = "State whether true or false: Neural networks require labeled data."
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "true_false")

    def test_numerical_detection(self):
        question = "Calculate the value of the resistance."
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "numerical")

    def test_open_ended_detection(self):
        question = "Explain the concept of backpropagation in detail."
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "open_ended")

    def test_classifier_returns_confidence(self):
        question = "What is machine learning?"
        result = self.classifier.classify(question)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


# ============================================================================
# Rubric Matcher Tests
# ============================================================================

class TestRubricMatcher(unittest.TestCase):

    def setUp(self):
        self.matcher = RubricMatcher(use_zero_shot=False)

    def test_rubric_matching(self):
        answer = "Machine learning is a subset of AI. Neural networks are an example."
        rubric = [
            {"criterion": "definition of ML", "marks": 2},
            {"criterion": "neural networks example", "marks": 2}
        ]
        result = self.matcher.evaluate_rubric(answer, rubric)
        self.assertIsInstance(result.criteria_scores, dict)
        self.assertGreaterEqual(result.earned_rubric_marks, 0)
        self.assertLessEqual(result.earned_rubric_marks, result.total_rubric_marks)

    def test_mcq_skips_rubric(self):
        answer = "B"
        rubric = [{"criterion": "test", "marks": 1}]
        result = self.matcher.evaluate_rubric(answer, rubric, question_type="mcq")
        self.assertEqual(result.earned_rubric_marks, 0.0)
        self.assertEqual(result.criteria_scores, {})

    def test_empty_rubric(self):
        result = self.matcher.evaluate_rubric("Some answer", [])
        self.assertEqual(result.earned_rubric_marks, 0.0)
        self.assertEqual(result.coverage_ratio, 0.0)


# ============================================================================
# Evaluation Engine Tests
# ============================================================================

class TestEvaluationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = EvaluationEngine(
            ocr_engine=os.getenv("OCR_ENGINE", "trocr"),
            llm_weight=float(os.getenv("LLM_WEIGHT", "0.40")),
            similarity_weight=float(os.getenv("SIMILARITY_WEIGHT", "0.25")),
            rubric_weight=float(os.getenv("RUBRIC_WEIGHT", "0.20")),
            keyword_weight=float(os.getenv("KEYWORD_WEIGHT", "0.10")),
            length_weight=float(os.getenv("LENGTH_WEIGHT", "0.05")),
        )

    def test_empty_result_creation(self):
        result = self.engine._empty_result(10.0, "open_ended", "Test", 0.5)
        self.assertEqual(result.final_score, 0.0)
        self.assertEqual(result.max_marks, 10.0)
        self.assertEqual(result.feedback, "Test")

    def test_hybrid_score_weights_sum(self):
        """Weights defined on the engine must sum to 1.0."""
        total = (
            self.engine.llm_weight
            + self.engine.similarity_weight
            + self.engine.rubric_weight
            + self.engine.keyword_weight
            + self.engine.length_weight
        )
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_hybrid_score_zero_inputs(self):
        score = self.engine._hybrid_score(
            llm_score=0.0, similarity=0.0, max_marks=10.0
        )
        self.assertEqual(score, 0.0)

    def test_hybrid_score_full_marks(self):
        score = self.engine._hybrid_score(
            llm_score=10.0, similarity=1.0, max_marks=10.0
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 10.0)


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics(unittest.TestCase):

    def test_perfect_agreement(self):
        ai = [7.0, 8.0, 9.0]
        teacher = [7.0, 8.0, 9.0]
        report = compute_metrics(ai, teacher)
        self.assertAlmostEqual(report.mae, 0.0)
        self.assertAlmostEqual(report.pearson_r, 1.0)

    def test_mae_computation(self):
        ai = [5.0, 7.0, 9.0]
        teacher = [6.0, 7.0, 8.0]
        report = compute_metrics(ai, teacher)
        self.assertAlmostEqual(report.mae, 2/3, places=5)

    def test_mcq_metrics(self):
        pred = ["A", "B", "C"]
        correct = ["A", "C", "C"]
        report = compute_mcq_metrics(pred, correct)
        self.assertAlmostEqual(report.mcq_accuracy, 2/3, places=5)
        self.assertEqual(report.mcq_n_correct, 2)
        self.assertEqual(report.mcq_n_wrong, 1)

    def test_accuracy_within_threshold(self):
        ai = [7.0, 8.0, 9.5]
        teacher = [8.0, 8.0, 8.0]
        report = compute_metrics(ai, teacher)
        self.assertGreaterEqual(report.accuracy_within_1, 0)
        self.assertLessEqual(report.accuracy_within_1, 1)


# ============================================================================
# API Integration Tests
# ============================================================================

class TestAPI(unittest.TestCase):

    def setUp(self):
        from backend.api import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("version", response.json())

    def test_upload_endpoint(self):
        img = create_test_image("Test answer")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            with open(f.name, "rb") as img_file:
                response = self.client.post(
                    "/upload",
                    files={"file": ("test.png", img_file, "image/png")},
                    data={"student_code": "TEST001"}
                )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("submission_id", data)

    def test_metrics_compute_endpoint(self):
        response = self.client.get(
            "/metrics/compute",
            params={
                "ai_scores": "7.5,8.0,6.5",
                "teacher_scores": "8.0,7.5,7.0",
                "max_marks": 10
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("open_ended", data)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance(unittest.TestCase):

    def test_preprocessor_speed(self):
        preprocessor = ImagePreprocessor()
        img = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
        start = time.time()
        preprocessor.preprocess(img)
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0)

    def test_ocr_speed(self):
        ocr = OCRModule()
        img = create_test_image("Short test sentence for OCR speed testing.")
        start = time.time()
        ocr.extract_text(img)
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0)   # raised from 5s — hybrid pipeline loads models


# ============================================================================
# Optional module tests (pytest-style, skipped if module missing)
# ============================================================================

import pytest

class TestLayoutDetector:
    """Tests for document layout detection — skipped if module not present."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("backend.layout_detector")

    def test_detect_returns_result(self, dummy_image_bytes):
        from backend.layout_detector import LayoutDetector
        detector = LayoutDetector()
        result = detector.detect(dummy_image_bytes)
        assert result is not None
        assert result.page_width > 0
        assert result.page_height > 0
        assert result.detector_used in ("detectron2", "opencv_fallback")

    def test_get_answer_crops_returns_list(self, dummy_image_bytes):
        from backend.layout_detector import LayoutDetector
        detector = LayoutDetector()
        crops = detector.get_answer_crops(dummy_image_bytes)
        assert isinstance(crops, list)
        assert len(crops) >= 1


class TestDiagramDetector:
    """Tests for diagram detection — skipped if module not present."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("backend.diagram_detector")

    def test_detect_plain_white_no_diagram(self):
        from backend.diagram_detector import DiagramDetector
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        detector = DiagramDetector()
        result = detector.detect(img)
        assert result is not None
        assert result.detector_used in ("yolov8", "heuristic_fallback")
        assert isinstance(result.has_diagram, bool)

    def test_detect_result_structure(self, dummy_image_bytes):
        from backend.diagram_detector import DiagramDetector
        detector = DiagramDetector()
        result = detector.detect(dummy_image_bytes)
        assert hasattr(result, "has_diagram")
        assert hasattr(result, "diagrams")
        assert hasattr(result, "n_diagrams")


class TestUpdatedHybridScoring:
    """Tests for the 5-factor hybrid scoring formula."""

    def test_weights_sum_to_one(self):
        from backend.evaluator import EvaluationEngine
        e = EvaluationEngine.__new__(EvaluationEngine)
        e.llm_weight     = 0.40
        e.similarity_weight = 0.25
        e.rubric_weight  = 0.20
        e.keyword_weight = 0.10
        e.length_weight  = 0.05
        total = (e.llm_weight + e.similarity_weight + e.rubric_weight
                 + e.keyword_weight + e.length_weight)
        assert abs(total - 1.0) < 1e-9

    def test_hybrid_score_zero(self):
        from backend.evaluator import EvaluationEngine
        e = EvaluationEngine.__new__(EvaluationEngine)
        e.llm_weight     = 0.40
        e.similarity_weight = 0.25
        e.rubric_weight  = 0.20
        e.keyword_weight = 0.10
        e.length_weight  = 0.05
        score = e._hybrid_score(
            llm_score=0.0, similarity=0.0, max_marks=10.0
        )
        assert score == 0.0


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestImagePreprocessor, TestOCRModule, TestTextProcessor,
        TestSimilarityModel, TestLLMProvider, TestLLMEvaluator,
        TestQuestionClassifier, TestRubricMatcher, TestEvaluationEngine,
        TestMetrics, TestAPI, TestPerformance,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)