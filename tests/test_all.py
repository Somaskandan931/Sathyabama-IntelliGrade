"""
IntelliGrade-H - Comprehensive Test Suite (v2)
===============================================
Tests all components with:
- Unit tests
- Integration tests
- Performance tests
- Error handling tests
- Explainability tests
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import modules
from backend.preprocessor import ImagePreprocessor
from backend.ocr_module import OCRModule, EasyOCREngine
from backend.text_processor import TextProcessor
from backend.similarity import SemanticSimilarityModel
from backend.llm_evaluator import LLMEvaluator
from backend.llm_provider import LLMClient, RuleBasedFallbackProvider
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
    except:
        font = ImageFont.load_default()

    draw.text((10, 30), text, fill=(0, 0, 0), font=font)
    return img


# ============================================================================
# Preprocessor Tests
# ============================================================================

class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing pipeline"""

    def setUp(self):
        self.preprocessor = ImagePreprocessor()

    def test_grayscale_conversion(self):
        """Test grayscale conversion"""
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(img)
        self.assertEqual(len(result.shape), 2)

    def test_denoise(self):
        """Test denoising"""
        img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        result = self.preprocessor._denoise(img)
        self.assertEqual(result.shape, img.shape)

    def test_deskew(self):
        """Test deskewing"""
        # Create skewed image
        img = Image.new("L", (400, 200), color=255)
        draw = ImageDraw.Draw(img)
        draw.line((50, 50, 350, 150), fill=0, width=3)
        img_array = np.array(img)

        result = self.preprocessor._deskew(img_array)
        self.assertEqual(result.shape, img_array.shape)

    def test_line_segmentation(self):
        """Test line segmentation"""
        # Create image with multiple lines
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
    """Test OCR functionality"""

    def setUp(self):
        self.ocr = OCRModule(engine="easyocr")

    def test_ocr_with_image(self):
        """Test OCR on simple image"""
        img = create_test_image("Hello World")
        result = self.ocr.extract_text(img)
        self.assertIsInstance(result.text, str)
        self.assertIsInstance(result.confidence, float)

    def test_ocr_empty_image(self):
        """Test OCR on empty image"""
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        result = self.ocr.extract_text(img)
        self.assertIsInstance(result.text, str)

    def test_ocr_confidence_range(self):
        """Test confidence score is between 0 and 1"""
        img = create_test_image("Test")
        result = self.ocr.extract_text(img)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    @patch('backend.ocr_module.EasyOCREngine.recognize')
    def test_ocr_fallback(self, mock_recognize):
        """Test fallback mechanism"""
        mock_recognize.side_effect = Exception("OCR failed")

        img = create_test_image("Test")
        result = self.ocr.extract_text(img)
        # Should still return something
        self.assertIsNotNone(result)


# ============================================================================
# Text Processor Tests
# ============================================================================

class TestTextProcessor(unittest.TestCase):
    """Test text processing"""

    def setUp(self):
        self.processor = TextProcessor()

    def test_normalize(self):
        """Test text normalization"""
        text = "This  has   multiple    spaces"
        result = self.processor._normalize(text)
        self.assertNotIn("  ", result)

    def test_spellcheck(self):
        """Test spell checking"""
        text = "This is a test with mispelled word"
        result = self.processor._spellcheck(text)
        self.assertIsInstance(result, str)

    def test_sentence_segmentation(self):
        """Test sentence segmentation"""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.processor._segment_sentences(text)
        self.assertGreaterEqual(len(sentences), 3)

    def test_tokenization(self):
        """Test tokenization"""
        text = "Machine learning is fascinating"
        tokens = self.processor._tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)


# ============================================================================
# Similarity Model Tests
# ============================================================================

class TestSimilarityModel(unittest.TestCase):
    """Test semantic similarity"""

    def setUp(self):
        self.model = SemanticSimilarityModel()

    def test_identical_answers(self):
        """Test identical answers give high similarity"""
        text = "Neural networks are a type of machine learning model."
        result = self.model.compute_similarity(text, text)
        self.assertGreaterEqual(result.score, 0.9)

    def test_different_answers(self):
        """Test different answers give lower similarity"""
        text1 = "Neural networks are a type of machine learning model."
        text2 = "The weather is nice today."
        result = self.model.compute_similarity(text1, text2)
        self.assertLess(result.score, 0.5)

    def test_similarity_range(self):
        """Test similarity is between 0 and 1"""
        text1 = "Machine learning uses algorithms."
        text2 = "AI systems learn from data."
        result = self.model.compute_similarity(text1, text2)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_sentence_level(self):
        """Test sentence-level similarity"""
        text1 = "First sentence. Second sentence."
        text2 = "First sentence. Different second."
        result = self.model.compute_sentence_level(text1, text2)
        self.assertIn("sentence_scores", result)


# ============================================================================
# LLM Provider Tests
# ============================================================================

class TestLLMProvider(unittest.TestCase):
    """Test LLM providers"""

    def test_rule_based_fallback(self):
        """Test rule-based fallback provider"""
        provider = RuleBasedFallbackProvider()
        self.assertTrue(provider.is_available())

        prompt = """
QUESTION: What is machine learning?
MODEL ANSWER: Machine learning is a subset of AI.
STUDENT ANSWER: ML is a field of AI.
MAXIMUM MARKS: 10
"""
        response = provider.generate(prompt)
        self.assertIsInstance(response.text, str)

        # Try to parse as JSON
        try:
            data = json.loads(response.text)
            self.assertIn("score", data)
            self.assertIn("confidence", data)
        except:
            self.fail("Response should be valid JSON")

    def test_llm_client_from_env(self):
        """Test LLM client creation"""
        client = LLMClient.from_env()
        self.assertIsInstance(client, LLMClient)
        self.assertGreater(len(client._providers), 0)

    def test_llm_client_fallback(self):
        """Test LLM client fallback chain"""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "invalid",
            "ANTHROPIC_API_KEY": "invalid"
        }):
            client = LLMClient.from_env()
            prompt = "Test prompt"
            response = client.generate(prompt)
            self.assertIsNotNone(response)


# ============================================================================
# LLM Evaluator Tests
# ============================================================================

class TestLLMEvaluator(unittest.TestCase):
    """Test LLM evaluation"""

    def setUp(self):
        self.evaluator = LLMEvaluator(api_key="dummy")

    def test_empty_answer(self):
        """Test empty answer handling"""
        result = self.evaluator.evaluate(
            question="What is AI?",
            teacher_answer="Artificial Intelligence",
            student_answer="",
            max_marks=10
        )
        self.assertEqual(result.score, 0.0)
        self.assertIsNotNone(result.explanation)

    def test_parse_json_response(self):
        """Test JSON parsing"""
        raw = '{"score": 7.5, "confidence": 0.85, "strengths": ["Good"], "missing_concepts": [], "feedback": "Good"}'
        result = self.evaluator._parse_response(raw, max_marks=10)
        self.assertEqual(result.score, 7.5)

    def test_parse_json_with_markdown(self):
        """Test JSON parsing with markdown fences"""
        raw = '```json\n{"score": 8, "confidence": 0.9, "strengths": [], "missing_concepts": [], "feedback": "OK"}\n```'
        result = self.evaluator._parse_response(raw, max_marks=10)
        self.assertEqual(result.score, 8.0)

    def test_build_prompt_with_rubric(self):
        """Test prompt building with rubric"""
        rubric = [{"criterion": "Definition", "marks": 2}]
        prompt = self.evaluator._build_enhanced_prompt(
            question="Test?",
            teacher_answer="Answer",
            student_answer="Student",
            max_marks=10,
            rubric_criteria=rubric
        )
        self.assertIn("Definition", prompt)
        self.assertIn("explanation", prompt.lower())


# ============================================================================
# Question Classifier Tests
# ============================================================================

class TestQuestionClassifier(unittest.TestCase):
    """Test question type classification"""

    def setUp(self):
        self.classifier = QuestionClassifier()

    def test_mcq_detection(self):
        """Test MCQ detection"""
        question = "Which of the following is a machine learning algorithm? A) Decision Tree B) K-Means C) Both"
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "mcq")

    def test_true_false_detection(self):
        """Test true/false detection"""
        question = "State whether true or false: Neural networks require labeled data."
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "true_false")

    def test_numerical_detection(self):
        """Test numerical detection"""
        question = "Calculate the value of 2+2."
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "numerical")

    def test_open_ended_detection(self):
        """Test open-ended detection"""
        question = "Explain the concept of backpropagation in detail."
        result = self.classifier._classify_rule_based(question)
        self.assertEqual(result.question_type, "open_ended")

    def test_classifier_returns_confidence(self):
        """Test classifier returns confidence score"""
        question = "What is machine learning?"
        result = self.classifier.classify(question)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


# ============================================================================
# Rubric Matcher Tests
# ============================================================================

class TestRubricMatcher(unittest.TestCase):
    """Test rubric matching"""

    def setUp(self):
        self.matcher = RubricMatcher(use_zero_shot=False)  # Use rule-based for testing

    def test_rubric_matching(self):
        """Test rubric matching"""
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
        """Test MCQ skips rubric matching"""
        answer = "B"
        rubric = [{"criterion": "test", "marks": 1}]

        result = self.matcher.evaluate_rubric(answer, rubric, question_type="mcq")
        self.assertEqual(result.earned_rubric_marks, 0.0)
        self.assertEqual(result.criteria_scores, {})

    def test_empty_rubric(self):
        """Test empty rubric"""
        result = self.matcher.evaluate_rubric("Some answer", [])
        self.assertEqual(result.earned_rubric_marks, 0.0)
        self.assertEqual(result.coverage_ratio, 0.0)


# ============================================================================
# Evaluation Engine Tests
# ============================================================================

class TestEvaluationEngine(unittest.TestCase):
    """Test main evaluation engine"""

    def setUp(self):
        self.engine = EvaluationEngine(
            ocr_engine="easyocr",
            llm_weight=0.6,
            similarity_weight=0.4
        )

    def test_mcq_evaluation(self):
        """Test MCQ evaluation"""
        img = create_test_image("B")
        result = self.engine.evaluate(
            student_image=img,
            question="Test MCQ?",
            question_type="mcq",
            correct_option="B",
            max_marks=2
        )
        self.assertEqual(result.question_type, "mcq")
        self.assertIn(result.final_score, [0.0, 2.0])

    def test_true_false_evaluation(self):
        """Test true/false evaluation"""
        img = create_test_image("True")
        result = self.engine.evaluate(
            student_image=img,
            question="Test T/F?",
            question_type="true_false",
            correct_answer="True",
            max_marks=1
        )
        self.assertEqual(result.question_type, "true_false")

    def test_hybrid_score(self):
        """Test hybrid score calculation"""
        score = self.engine._hybrid_score(7.0, 0.8, 10.0)
        expected = 0.6 * 7.0 + 0.4 * 0.8 * 10.0
        self.assertAlmostEqual(score, expected)

    def test_empty_result_creation(self):
        """Test empty result creation"""
        result = self.engine._empty_result(10.0, "open_ended", "Test", 0.5)
        self.assertEqual(result.final_score, 0.0)
        self.assertEqual(result.max_marks, 10.0)
        self.assertEqual(result.feedback, "Test")


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics(unittest.TestCase):
    """Test metrics computation"""

    def test_perfect_agreement(self):
        """Test perfect agreement"""
        ai = [7.0, 8.0, 9.0]
        teacher = [7.0, 8.0, 9.0]
        report = compute_metrics(ai, teacher)
        self.assertAlmostEqual(report.mae, 0.0)
        self.assertAlmostEqual(report.pearson_r, 1.0)

    def test_mae_computation(self):
        """Test MAE computation"""
        ai = [5.0, 7.0, 9.0]
        teacher = [6.0, 7.0, 8.0]
        report = compute_metrics(ai, teacher)
        self.assertAlmostEqual(report.mae, 1.0)

    def test_mcq_metrics(self):
        """Test MCQ metrics"""
        pred = ["A", "B", "C"]
        correct = ["A", "C", "C"]
        report = compute_mcq_metrics(pred, correct)
        self.assertAlmostEqual(report.mcq_accuracy, 2/3)
        self.assertEqual(report.mcq_n_correct, 2)
        self.assertEqual(report.mcq_n_wrong, 1)

    def test_accuracy_within_threshold(self):
        """Test accuracy within threshold"""
        ai = [7.0, 8.0, 9.5]
        teacher = [8.0, 8.0, 8.0]
        report = compute_metrics(ai, teacher)
        self.assertGreaterEqual(report.accuracy_within_1, 0)
        self.assertLessEqual(report.accuracy_within_1, 1)


# ============================================================================
# API Integration Tests
# ============================================================================

class TestAPI(unittest.TestCase):
    """Test API endpoints"""

    def setUp(self):
        from backend.api import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("version", response.json())

    def test_upload_endpoint(self):
        """Test upload endpoint"""
        # Create test image
        img = create_test_image("Test answer")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)

            # Upload
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
        """Test metrics compute endpoint"""
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
    """Test system performance"""

    def test_preprocessor_speed(self):
        """Test preprocessor speed"""
        preprocessor = ImagePreprocessor()
        img = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

        start = time.time()
        result = preprocessor.preprocess(img)
        elapsed = time.time() - start

        self.assertLess(elapsed, 1.0)  # Should process in under 1 second

    def test_ocr_speed(self):
        """Test OCR speed"""
        ocr = OCRModule(engine="easyocr")
        img = create_test_image("Short test sentence for OCR speed testing.")

        start = time.time()
        result = ocr.extract_text(img)
        elapsed = time.time() - start

        self.assertLess(elapsed, 5.0)  # Should OCR in under 5 seconds


# ============================================================================
# Explainability Tests
# ============================================================================

class TestExplainability(unittest.TestCase):
    """Test explainability features"""

    def test_llm_evaluation_includes_explanation(self):
        """Test LLM evaluation includes explanation"""
        evaluator = LLMEvaluator(api_key="dummy")

        # Mock response with explanation
        with patch.object(evaluator, '_get_client') as mock_client:
            mock_client.return_value.generate_json.return_value = {
                "score": 7.5,
                "confidence": 0.85,
                "strengths": ["Good understanding"],
                "missing_concepts": ["Examples"],
                "feedback": "Good attempt",
                "explanation": {
                    "score_rationale": "Student demonstrated good understanding but missed examples",
                    "key_factors": ["Understanding of concepts", "Missing examples"],
                    "comparison_with_model": "Covered main points but lacked depth",
                    "improvement_suggestions": ["Add more examples"],
                    "confidence_factors": {"length": 100, "coverage": 0.8}
                }
            }

            result = evaluator.evaluate(
                question="Test?",
                teacher_answer="Answer",
                student_answer="Student",
                max_marks=10
            )

            self.assertIsNotNone(result.explanation)
            self.assertEqual(result.explanation.score_rationale,
                           "Student demonstrated good understanding but missed examples")

    def test_rule_based_fallback_explanation(self):
        """Test rule-based fallback includes explanation"""
        provider = RuleBasedFallbackProvider()

        prompt = """
QUESTION: What is machine learning?
MODEL ANSWER: Machine learning is a subset of AI that learns from data.
STUDENT ANSWER: ML is a field of AI.
MAXIMUM MARKS: 10
"""

        response = provider.generate(prompt)
        data = json.loads(response.text)

        self.assertIn("explanation", data)
        explanation = data["explanation"]
        self.assertIn("score_rationale", explanation)
        self.assertIn("key_factors", explanation)
        self.assertIn("improvement_suggestions", explanation)


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImagePreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestOCRModule))
    suite.addTests(loader.loadTestsFromTestCase(TestTextProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestSimilarityModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestQuestionClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestRubricMatcher))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainability))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)