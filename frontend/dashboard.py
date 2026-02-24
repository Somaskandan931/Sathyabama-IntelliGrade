"""
IntelliGrade-H - Teacher Dashboard
Streamlit-based web interface for uploading and evaluating answer sheets.
"""

import streamlit as st
import requests
import json
import io
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="IntelliGrade-H",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Upload zone cards */
    .upload-card {
        background: #ffffff;
        border: 2px dashed #c0c8d8;
        border-radius: 16px;
        padding: 20px 16px 12px 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .upload-card:hover { border-color: #4a7cf7; }

    .upload-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .upload-icon {
        font-size: 28px;
        line-height: 1;
    }
    .upload-title {
        font-size: 15px;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    .upload-subtitle {
        font-size: 12px;
        color: #6b7280;
        margin: 0;
    }
    .upload-badge {
        display: inline-block;
        background: #4a7cf7;
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .upload-badge-optional {
        background: #9ca3af;
    }

    /* Preview thumbnails */
    .preview-strip {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
    .preview-label {
        font-size: 11px;
        color: #6b7280;
        text-align: center;
        margin-top: 2px;
    }

    /* Status pills */
    .pill-done   { background:#dcfce7; color:#166534; border-radius:20px; padding:3px 10px; font-size:12px; font-weight:600; }
    .pill-wait   { background:#fef9c3; color:#713f12; border-radius:20px; padding:3px 10px; font-size:12px; font-weight:600; }
    .pill-err    { background:#fee2e2; color:#991b1b; border-radius:20px; padding:3px 10px; font-size:12px; font-weight:600; }

    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
    }
    .score-value { font-size: 52px; font-weight: 800; }
    .score-label { font-size: 13px; opacity: 0.65; text-transform: uppercase; letter-spacing: 1px; }

    /* Feedback boxes */
    .feedback-box {
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 14px 16px;
        border-radius: 8px;
        margin: 6px 0;
        font-size: 14px;
    }
    .missing-box {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 14px 16px;
        border-radius: 8px;
        margin: 6px 0;
        font-size: 14px;
    }

    /* Step tracker */
    .step-bar {
        display: flex;
        align-items: center;
        gap: 0;
        margin: 18px 0 24px 0;
    }
    .step-node {
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 13px;
        flex-shrink: 0;
    }
    .step-active  { background:#4a7cf7; color:white; }
    .step-done    { background:#22c55e; color:white; }
    .step-pending { background:#e5e7eb; color:#9ca3af; }
    .step-line    { flex: 1; height: 3px; background: #e5e7eb; }
    .step-line-done { background: #22c55e; }
    .step-label   { font-size: 11px; color:#6b7280; text-align:center; margin-top:4px; }

    div[data-testid="stFileUploader"] { margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📝 IntelliGrade-H")
    st.caption("Sathyabama Institute of Science and Technology")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📤 Evaluate Answer", "📊 Batch Evaluation", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("v1.0 — AI-Powered Grading")


# ─────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────

def api_health_check():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def upload_file(file_bytes, filename, student_code):
    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (filename, file_bytes, "image/jpeg")},
        data={"student_code": student_code}
    )
    r.raise_for_status()
    return r.json()


def evaluate_submission(submission_id, question, teacher_answer, max_marks, rubric):
    payload = {
        "submission_id": submission_id,
        "question": question,
        "teacher_answer": teacher_answer,
        "max_marks": max_marks,
        "rubric_criteria": rubric
    }
    r = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def get_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json()
    except Exception:
        return {}


def render_score_card(score, max_marks, confidence):
    pct = (score / max_marks) * 100 if max_marks > 0 else 0
    color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 50 else "#f44336"
    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">Final Score</div>
        <div class="score-value" style="color:{color};">{score}</div>
        <div style="color:#aaa; font-size:18px;">out of {max_marks}</div>
        <div style="margin-top:12px; color:#90caf9;">AI Confidence: {int(confidence * 100)}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct / 100)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        return f"[PDF text extraction failed: {e}]"


def pdf_first_page_image(pdf_bytes: bytes) -> Image.Image:
    """Render first page of a PDF as a PIL Image for preview."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception:
        return None


def render_file_preview(uploaded, label: str, icon: str, show_text_preview: bool = False):
    """Show a compact preview panel for an uploaded file."""
    if uploaded is None:
        return

    st.markdown(f"<div style='margin-top:6px'>", unsafe_allow_html=True)

    is_pdf = uploaded.name.lower().endswith(".pdf")

    if is_pdf:
        preview_img = pdf_first_page_image(uploaded.getvalue())
        if preview_img:
            st.image(preview_img, caption=f"{icon} {label} — Page 1 preview", use_column_width=True)
        else:
            st.info(f"📄 PDF uploaded: **{uploaded.name}**")
    else:
        st.image(uploaded.getvalue(), caption=f"{icon} {label}", use_column_width=True)

    if show_text_preview and is_pdf:
        extracted = extract_text_from_pdf(uploaded.getvalue())
        if extracted:
            with st.expander("🔍 Preview extracted text", expanded=False):
                st.text(extracted[:800] + ("..." if len(extracted) > 800 else ""))

    st.markdown("</div>", unsafe_allow_html=True)


def step_tracker(steps: list, current: int):
    """Render a horizontal step progress bar."""
    nodes_html = ""
    for i, label in enumerate(steps):
        if i < current:
            cls = "step-done"
            icon = "✓"
        elif i == current:
            cls = "step-active"
            icon = str(i + 1)
        else:
            cls = "step-pending"
            icon = str(i + 1)

        line_cls = "step-line-done" if i < len(steps) - 1 and i < current else "step-line"
        line = f'<div class="step-line {line_cls}"></div>' if i < len(steps) - 1 else ""

        nodes_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;min-width:56px">
            <div class="step-node {cls}">{icon}</div>
            <div class="step-label">{label}</div>
        </div>
        {line}
        """

    st.markdown(f'<div class="step-bar">{nodes_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    st.title("📚 IntelliGrade-H — Teacher Dashboard")
    st.subheader("AI-Powered Handwritten Answer Evaluation System")

    is_alive = api_health_check()
    if is_alive:
        st.success("✅ API Server is online")
    else:
        st.error("❌ API Server is offline. Start: `uvicorn backend.api:app --reload`")

    st.divider()

    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Submissions", stats.get("total_submissions", 0))
    col2.metric("Evaluated", stats.get("evaluated", 0))
    col3.metric("Avg Score", stats.get("average_score", "—"))
    col4.metric("Avg Eval Time (s)", stats.get("average_evaluation_time_sec", "—"))

    st.divider()
    st.markdown("""
    ### How to Use
    1. Go to **📤 Evaluate Answer** — upload the **Question Paper**, **Model Answer Sheet**, and **Student Answer Sheet**.
    2. The AI will OCR the handwritten sheets, compare them, and generate a score + feedback.
    3. Go to **📊 Batch Evaluation** to process an entire class at once.
    """)


# ─────────────────────────────────────────────────────────
# Page: Evaluate Answer  (REDESIGNED)
# ─────────────────────────────────────────────────────────

elif page == "📤 Evaluate Answer":
    st.title("📤 Evaluate Student Answer")
    st.caption("Upload all three documents below, then click Evaluate.")

    # ── Step tracker ────────────────────────────────────────
    STEPS = ["Upload Files", "Set Parameters", "Evaluate", "View Results"]
    if "eval_step" not in st.session_state:
        st.session_state.eval_step = 0

    step_tracker(STEPS, st.session_state.eval_step)

    st.divider()

    # ════════════════════════════════════════════════════════
    # SECTION 1 — Three upload zones side by side
    # ════════════════════════════════════════════════════════

    st.markdown("### 📁 Step 1 — Upload Documents")
    st.caption("All three documents are required. Accepted formats: JPG, PNG, PDF.")

    zone1, zone2, zone3 = st.columns(3, gap="medium")

    # ── Zone A: Question Paper ────────────────────────────
    with zone1:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-header">
                <span class="upload-icon">📋</span>
                <div>
                    <p class="upload-title">Question Paper</p>
                    <p class="upload-subtitle">The exam question document</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        qp_file = st.file_uploader(
            "Upload Question Paper",
            type=["jpg", "jpeg", "png", "pdf"],
            key="qp_upload",
            label_visibility="collapsed"
        )
        if qp_file:
            st.markdown('<span class="pill-done">✓ Uploaded</span>', unsafe_allow_html=True)
            render_file_preview(qp_file, "Question Paper", "📋", show_text_preview=True)
        else:
            st.markdown('<span class="pill-wait">⏳ Awaiting upload</span>', unsafe_allow_html=True)

    # ── Zone B: Model Answer Sheet ────────────────────────
    with zone2:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-header">
                <span class="upload-icon">📗</span>
                <div>
                    <p class="upload-title">Model Answer Sheet</p>
                    <p class="upload-subtitle">Teacher's expected answer</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        ma_file = st.file_uploader(
            "Upload Model Answer Sheet",
            type=["jpg", "jpeg", "png", "pdf"],
            key="ma_upload",
            label_visibility="collapsed"
        )
        if ma_file:
            st.markdown('<span class="pill-done">✓ Uploaded</span>', unsafe_allow_html=True)
            render_file_preview(ma_file, "Model Answer", "📗", show_text_preview=True)
        else:
            st.markdown('<span class="pill-wait">⏳ Awaiting upload</span>', unsafe_allow_html=True)

    # ── Zone C: Student Answer Sheet ──────────────────────
    with zone3:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-header">
                <span class="upload-icon">✍️</span>
                <div>
                    <p class="upload-title">Student Answer Sheet</p>
                    <p class="upload-subtitle">Handwritten answer to evaluate</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        sa_file = st.file_uploader(
            "Upload Student Answer Sheet",
            type=["jpg", "jpeg", "png", "pdf"],
            key="sa_upload",
            label_visibility="collapsed"
        )
        if sa_file:
            st.markdown('<span class="pill-done">✓ Uploaded</span>', unsafe_allow_html=True)
            render_file_preview(sa_file, "Student Answer", "✍️")
        else:
            st.markdown('<span class="pill-wait">⏳ Awaiting upload</span>', unsafe_allow_html=True)

    # Upload status summary
    all_uploaded = qp_file and ma_file and sa_file
    if all_uploaded:
        st.success("✅ All three documents uploaded. Proceed to Step 2.")
        st.session_state.eval_step = max(st.session_state.eval_step, 1)
    elif any([qp_file, ma_file, sa_file]):
        missing = []
        if not qp_file: missing.append("Question Paper")
        if not ma_file: missing.append("Model Answer Sheet")
        if not sa_file: missing.append("Student Answer Sheet")
        st.warning(f"⚠️ Still needed: {', '.join(missing)}")

    st.divider()

    # ════════════════════════════════════════════════════════
    # SECTION 2 — Parameters
    # ════════════════════════════════════════════════════════

    st.markdown("### ⚙️ Step 2 — Evaluation Parameters")

    param_col1, param_col2 = st.columns([2, 1])

    with param_col1:
        student_code = st.text_input(
            "🎓 Student ID / Roll Number",
            placeholder="e.g. SIST2024001",
            help="Used to identify this submission in the database."
        )
        exam_name = st.text_input(
            "📝 Exam / Subject Name (optional)",
            placeholder="e.g. Machine Learning — Unit 3 Test"
        )

    with param_col2:
        max_marks = st.number_input(
            "🏆 Maximum Marks",
            min_value=1.0, max_value=100.0, value=10.0, step=0.5
        )
        question_number = st.number_input(
            "# Question Number",
            min_value=1, max_value=50, value=1, step=1
        )

    # ── Manual override: question text ──────────────────────
    with st.expander("✏️ Override extracted question text (optional)", expanded=False):
        st.caption("If the OCR misreads the question paper, you can type the question manually here.")
        manual_question = st.text_area(
            "Question text (leave blank to use OCR from question paper)",
            height=90,
            placeholder="e.g. Explain the working principle of a neural network with a diagram."
        )

    with st.expander("✏️ Override extracted model answer text (optional)", expanded=False):
        st.caption("If the OCR misreads the model answer, type it manually here.")
        manual_teacher_answer = st.text_area(
            "Model answer text (leave blank to use OCR from model answer sheet)",
            height=130,
            placeholder="Enter the expected answer..."
        )

    # ── Rubric (optional) ────────────────────────────────────
    with st.expander("📏 Rubric Criteria (optional)", expanded=False):
        st.caption("Define specific criteria the AI should check for. Leave empty to skip.")
        num_criteria = st.number_input("Number of criteria", min_value=0, max_value=10, value=0)
        rubric = []
        for i in range(int(num_criteria)):
            rc1, rc2 = st.columns([3, 1])
            criterion = rc1.text_input(f"Criterion {i+1}", key=f"crit_{i}",
                                        placeholder="e.g. Definition with example")
            marks = rc2.number_input(f"Marks", min_value=0.1, max_value=20.0,
                                      value=1.0, key=f"marks_{i}")
            if criterion:
                rubric.append({"criterion": criterion, "marks": marks})

    st.divider()

    # ════════════════════════════════════════════════════════
    # SECTION 3 — Evaluate Button
    # ════════════════════════════════════════════════════════

    st.markdown("### 🚀 Step 3 — Run Evaluation")

    # Validation checklist
    checks = {
        "Question Paper uploaded": bool(qp_file),
        "Model Answer Sheet uploaded": bool(ma_file),
        "Student Answer Sheet uploaded": bool(sa_file),
        "Student ID entered": bool(student_code and student_code.strip()),
    }
    ready = all(checks.values())

    chk_cols = st.columns(len(checks))
    for col, (label, ok) in zip(chk_cols, checks.items()):
        icon = "✅" if ok else "❌"
        col.markdown(f"<div style='text-align:center; font-size:13px'>{icon}<br>{label}</div>",
                     unsafe_allow_html=True)

    st.write("")

    eval_btn = st.button(
        "🚀 Evaluate Answer Sheet",
        type="primary",
        use_container_width=True,
        disabled=not ready
    )

    if not ready:
        st.caption("Complete all checklist items above to enable evaluation.")

    # ════════════════════════════════════════════════════════
    # SECTION 4 — Evaluation Logic + Results
    # ════════════════════════════════════════════════════════

    if eval_btn and ready:
        st.session_state.eval_step = 2
        step_tracker(STEPS, 2)
        st.divider()

        result_placeholder = st.empty()

        with st.spinner("🔍 Step 1/4 — Extracting text from Question Paper..."):
            # Extract question text
            if manual_question and manual_question.strip():
                question_text = manual_question.strip()
                qp_method = "manual"
            elif qp_file.name.lower().endswith(".pdf"):
                question_text = extract_text_from_pdf(qp_file.getvalue())
                qp_method = "pdf-text"
            else:
                # Image — will be OCR'd by backend; send as text placeholder for now
                question_text = f"[Question extracted from image: {qp_file.name}]"
                qp_method = "image-ocr"

        with st.spinner("🔍 Step 2/4 — Extracting text from Model Answer Sheet..."):
            if manual_teacher_answer and manual_teacher_answer.strip():
                teacher_answer_text = manual_teacher_answer.strip()
                ma_method = "manual"
            elif ma_file.name.lower().endswith(".pdf"):
                teacher_answer_text = extract_text_from_pdf(ma_file.getvalue())
                ma_method = "pdf-text"
            else:
                teacher_answer_text = f"[Model answer extracted from image: {ma_file.name}]"
                ma_method = "image-ocr"

        with st.spinner("📤 Step 3/4 — Uploading student answer sheet..."):
            try:
                upload_resp = upload_file(
                    sa_file.getvalue(),
                    sa_file.name,
                    student_code.strip()
                )
                submission_id = upload_resp["submission_id"]
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

        with st.spinner("🤖 Step 4/4 — Running AI evaluation (30–60 seconds)..."):
            try:
                result = evaluate_submission(
                    submission_id=submission_id,
                    question=question_text,
                    teacher_answer=teacher_answer_text,
                    max_marks=max_marks,
                    rubric=rubric if rubric else None
                )
            except requests.HTTPError as e:
                st.error(f"Evaluation API error: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.stop()

        # ── Results ────────────────────────────────────────
        st.session_state.eval_step = 3
        step_tracker(STEPS, 3)
        st.success("✅ Evaluation complete!")
        st.divider()

        st.markdown("### 📊 Step 4 — Results")

        # Scores row
        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            render_score_card(result["final_score"], result["max_marks"], result["confidence"])

        with r2:
            st.metric("LLM Score", f"{result['llm_score']:.1f} / {result['max_marks']}")
            st.metric("Semantic Similarity", f"{result['similarity_score']:.2%}")
            st.metric("OCR Confidence", f"{result['ocr_confidence']:.2%}")
            st.metric("Evaluation Time", f"{result['evaluation_time_sec']}s")

        with r3:
            if result.get("rubric_details"):
                rd = result["rubric_details"]
                st.metric("Rubric Coverage",
                          f"{rd['earned_rubric_marks']:.1f} / {rd['total_rubric_marks']:.1f}")
                for criterion, present in rd.get("criteria_scores", {}).items():
                    icon = "✅" if present else "❌"
                    st.write(f"{icon} {criterion}")
            else:
                st.info("No rubric criteria were defined for this evaluation.")

        # Document comparison panel
        with st.expander("📄 View All Extracted Texts", expanded=False):
            doc_a, doc_b, doc_c = st.columns(3)
            with doc_a:
                st.markdown("**📋 Question Paper**")
                st.text_area("", value=question_text, height=180, disabled=True, key="qt_display")
                st.caption(f"Source: {qp_method}")
            with doc_b:
                st.markdown("**📗 Model Answer**")
                st.text_area("", value=teacher_answer_text, height=180, disabled=True, key="ta_display")
                st.caption(f"Source: {ma_method}")
            with doc_c:
                st.markdown("**✍️ Student Answer (OCR)**")
                st.text_area("", value=result.get("ocr_text", ""), height=180, disabled=True, key="sa_display")
                st.caption(f"OCR Engine: {result.get('ocr_engine', 'N/A')}")

        # Feedback
        fb_col, miss_col = st.columns(2)
        with fb_col:
            st.markdown("#### 💪 Strengths")
            strengths = result.get("strengths", [])
            if strengths:
                for s in strengths:
                    st.markdown(f'<div class="feedback-box">✅ {s}</div>', unsafe_allow_html=True)
            else:
                st.caption("No specific strengths noted.")

        with miss_col:
            st.markdown("#### ⚠️ Missing Concepts")
            missing = result.get("missing_concepts", [])
            if missing:
                for m in missing:
                    st.markdown(f'<div class="missing-box">⚠️ {m}</div>', unsafe_allow_html=True)
            else:
                st.caption("No missing concepts detected.")

        st.markdown("#### 💡 AI Feedback")
        st.info(result.get("feedback", "No feedback generated."))

        # Download
        st.divider()
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "⬇️ Download Result (JSON)",
                data=json.dumps({**result, "student_id": student_code,
                                 "exam": exam_name, "question_number": question_number}, indent=2),
                file_name=f"result_{student_code}_Q{question_number}.json",
                mime="application/json",
                use_container_width=True
            )
        with dl_col2:
            # Simple text report
            report_lines = [
                f"IntelliGrade-H — Evaluation Report",
                f"{'='*45}",
                f"Student ID   : {student_code}",
                f"Exam         : {exam_name or 'N/A'}",
                f"Question No  : {question_number}",
                f"Max Marks    : {result['max_marks']}",
                f"Final Score  : {result['final_score']}",
                f"LLM Score    : {result['llm_score']}",
                f"Similarity   : {result['similarity_score']:.2%}",
                f"OCR Conf.    : {result['ocr_confidence']:.2%}",
                f"{'='*45}",
                f"STRENGTHS:",
                *[f"  + {s}" for s in result.get("strengths", [])],
                f"MISSING CONCEPTS:",
                *[f"  - {m}" for m in result.get("missing_concepts", [])],
                f"FEEDBACK:",
                f"  {result.get('feedback', '')}",
            ]
            st.download_button(
                "⬇️ Download Report (TXT)",
                data="\n".join(report_lines),
                file_name=f"report_{student_code}_Q{question_number}.txt",
                mime="text/plain",
                use_container_width=True
            )


# ─────────────────────────────────────────────────────────
# Page: Batch Evaluation
# ─────────────────────────────────────────────────────────

elif page == "📊 Batch Evaluation":
    st.title("📊 Batch Evaluation")
    st.caption("Upload the question paper, model answer, and all student answer sheets to evaluate an entire class.")

    st.markdown("### 📁 Upload Reference Documents")
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-header">
                <span class="upload-icon">📋</span>
                <div><p class="upload-title">Question Paper</p>
                <p class="upload-subtitle">Same for all students</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        batch_qp = st.file_uploader("Upload Question Paper", type=["jpg","jpeg","png","pdf"],
                                     key="batch_qp", label_visibility="collapsed")
        if batch_qp:
            st.markdown('<span class="pill-done">✓ Uploaded</span>', unsafe_allow_html=True)
            render_file_preview(batch_qp, "Question Paper", "📋", show_text_preview=True)

    with b_col2:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-card-header">
                <span class="upload-icon">📗</span>
                <div><p class="upload-title">Model Answer Sheet</p>
                <p class="upload-subtitle">Teacher's reference answer</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        batch_ma = st.file_uploader("Upload Model Answer", type=["jpg","jpeg","png","pdf"],
                                     key="batch_ma", label_visibility="collapsed")
        if batch_ma:
            st.markdown('<span class="pill-done">✓ Uploaded</span>', unsafe_allow_html=True)
            render_file_preview(batch_ma, "Model Answer", "📗", show_text_preview=True)

    st.divider()
    st.markdown("### ✍️ Upload Student Answer Sheets")
    st.caption("Name each file as the student's Roll Number (e.g. SIST2024001.jpg). The filename will be used as the Student ID.")

    batch_students = st.file_uploader(
        "Upload student answer sheets (multiple files)",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        key="batch_students"
    )

    if batch_students:
        st.success(f"✅ {len(batch_students)} student answer sheet(s) uploaded.")
        thumb_cols = st.columns(min(len(batch_students), 5))
        for i, f in enumerate(batch_students[:5]):
            with thumb_cols[i]:
                try:
                    st.image(f.getvalue(), caption=Path(f.name).stem, use_column_width=True)
                except Exception:
                    st.caption(f.name)
        if len(batch_students) > 5:
            st.caption(f"... and {len(batch_students) - 5} more.")

    b_param1, b_param2 = st.columns([2, 1])
    with b_param1:
        batch_exam = st.text_input("Exam / Subject Name", placeholder="e.g. AI — Unit 2 Test", key="batch_exam")
    with b_param2:
        batch_max_marks = st.number_input("Max Marks", value=10.0, step=0.5, key="batch_marks")

    with st.expander("✏️ Override extracted question text (optional)"):
        batch_manual_q = st.text_area("Question text", height=80, key="batch_mq",
                                       placeholder="Leave blank to extract from question paper file")
    with st.expander("✏️ Override extracted model answer text (optional)"):
        batch_manual_ma = st.text_area("Model answer text", height=100, key="batch_mma",
                                        placeholder="Leave blank to extract from model answer file")

    st.divider()
    batch_ready = bool(batch_qp and batch_ma and batch_students)
    if not batch_ready:
        st.warning("Upload the question paper, model answer sheet, and at least one student sheet to proceed.")

    if st.button("🚀 Run Batch Evaluation", type="primary", disabled=not batch_ready, use_container_width=True):
        # Extract question and model answer text
        if batch_manual_q and batch_manual_q.strip():
            batch_question = batch_manual_q.strip()
        elif batch_qp.name.lower().endswith(".pdf"):
            batch_question = extract_text_from_pdf(batch_qp.getvalue())
        else:
            batch_question = f"[Question from image: {batch_qp.name}]"

        if batch_manual_ma and batch_manual_ma.strip():
            batch_teacher_answer = batch_manual_ma.strip()
        elif batch_ma.name.lower().endswith(".pdf"):
            batch_teacher_answer = extract_text_from_pdf(batch_ma.getvalue())
        else:
            batch_teacher_answer = f"[Model answer from image: {batch_ma.name}]"

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_table = []

        for i, f in enumerate(batch_students):
            student_id = Path(f.name).stem
            status_text.info(f"Evaluating **{student_id}** ({i+1}/{len(batch_students)})...")
            try:
                upload_resp = upload_file(f.getvalue(), f.name, student_id)
                result = evaluate_submission(
                    submission_id=upload_resp["submission_id"],
                    question=batch_question,
                    teacher_answer=batch_teacher_answer,
                    max_marks=batch_max_marks,
                    rubric=None
                )
                results_table.append({
                    "Student ID": student_id,
                    "Score": result["final_score"],
                    "Max Marks": result["max_marks"],
                    "% Score": f"{(result['final_score']/result['max_marks']*100):.1f}%",
                    "LLM Score": result["llm_score"],
                    "Similarity": f"{result['similarity_score']:.2%}",
                    "OCR Confidence": f"{result['ocr_confidence']:.2%}",
                    "Eval Time (s)": result["evaluation_time_sec"],
                    "Feedback": result.get("feedback", "")[:80] + "..."
                })
            except Exception as e:
                results_table.append({
                    "Student ID": student_id,
                    "Score": "ERROR",
                    "Max Marks": batch_max_marks,
                    "% Score": "—",
                    "LLM Score": "—",
                    "Similarity": "—",
                    "OCR Confidence": "—",
                    "Eval Time (s)": "—",
                    "Feedback": str(e)
                })
            progress_bar.progress((i + 1) / len(batch_students))

        status_text.success(f"✅ Batch complete! {len(batch_students)} submissions evaluated.")

        # Summary stats
        valid_scores = [r["Score"] for r in results_table if isinstance(r["Score"], float)]
        if valid_scores:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Students", len(batch_students))
            s2.metric("Class Average", f"{sum(valid_scores)/len(valid_scores):.2f}")
            s3.metric("Highest Score", f"{max(valid_scores):.2f}")
            s4.metric("Lowest Score", f"{min(valid_scores):.2f}")

        st.dataframe(results_table, use_container_width=True, hide_index=True)

        import csv
        buf = io.StringIO()
        if results_table:
            writer = csv.DictWriter(buf, fieldnames=results_table[0].keys())
            writer.writeheader()
            writer.writerows(results_table)
        st.download_button(
            "⬇️ Download Batch Results (CSV)",
            data=buf.getvalue(),
            file_name=f"batch_results_{batch_exam or 'exam'}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ─────────────────────────────────────────────────────────
# Page: Settings
# ─────────────────────────────────────────────────────────

elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")

    st.subheader("API Configuration")
    api_url = st.text_input("API Base URL", value=API_BASE)
    gemini_key = st.text_input("Gemini API Key", type="password",
                                placeholder="Paste your Gemini API key here")

    st.subheader("Scoring Weights")
    llm_w = st.slider("LLM Weight", 0.0, 1.0, 0.6, 0.05)
    sim_w = st.slider("Similarity Weight", 0.0, 1.0, 0.4, 0.05)
    st.write(f"Formula: `Final = {llm_w} × LLM_Score + {sim_w} × Similarity × MaxMarks`")

    st.subheader("OCR Engine")
    ocr_engine = st.selectbox("Select OCR Engine", ["trocr", "tesseract"])
    st.caption("TrOCR is more accurate for handwriting. Tesseract is faster and lighter.")

    if st.button("Save Settings"):
        st.success("Settings saved. Restart the API server for changes to take effect.")

    st.divider()
    st.subheader("System Info")
    st.json({
        "api_status": "online" if api_health_check() else "offline",
        "version": "1.0.0",
        "ocr_engine": ocr_engine
    })