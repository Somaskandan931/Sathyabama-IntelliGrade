"""
pytest configuration and shared fixtures for IntelliGrade-H tests.
"""
import os
import sys
import pytest

# Make sure the project root is on sys.path so that `backend.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_student_answer():
    return (
        "Gradient descent is an optimization algorithm used to minimize "
        "the loss function by iteratively updating model parameters in the "
        "direction of the negative gradient. The learning rate controls the "
        "step size. When the gradient is zero, the algorithm converges."
    )


@pytest.fixture
def sample_teacher_answer():
    return (
        "Gradient descent is an iterative optimization algorithm that minimizes "
        "a loss function by computing the gradient and moving parameters in the "
        "opposite direction. The learning rate (alpha) determines the step size. "
        "Variants include batch, mini-batch, and stochastic gradient descent."
    )


@pytest.fixture
def sample_rubric():
    return [
        {"criterion": "definition of gradient descent", "marks": 3.0},
        {"criterion": "role of learning rate", "marks": 3.0},
        {"criterion": "convergence explanation", "marks": 2.0},
        {"criterion": "real-world example or application", "marks": 2.0},
    ]


@pytest.fixture
def dummy_image_bytes():
    """Returns minimal 1×1 white PNG bytes for OCR/preprocessing tests."""
    import io
    from PIL import Image
    img = Image.new("RGB", (100, 40), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
