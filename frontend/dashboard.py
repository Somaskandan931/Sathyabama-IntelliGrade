"""
IntelliGrade-H — Professional Teacher Dashboard
Streamlit UI for AI-powered handwritten answer evaluation.
"""

import streamlit as st
import requests
import json
import io
import csv
from pathlib import Path
from PIL import Image

# ── Optional PyMuPDF ─────────────────────────────────────
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# App Config
# ─────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="IntelliGrade-H",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,700;1,400&display=swap');

/* ── Base ────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem 2.5rem !important; }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #c8ccd8 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,130,255,0.12) !important;
    color: #ffffff !important;
}

/* ── Page title ──────────────────────────────────── */
.ig-title {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #0f1117;
    line-height: 1.1;
    margin-bottom: 4px;
}
.ig-sub {
    font-size: 14px;
    color: #6b7280;
    font-weight: 400;
    margin-bottom: 0;
}

/* ── Step Tracker ────────────────────────────────── */
.stepper {
    display: flex;
    align-items: center;
    padding: 20px 0 8px 0;
    gap: 0;
}
.step-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 80px;
}
.step-circle {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    transition: all 0.3s;
}
.step-done   { background: #22c55e; color: white; box-shadow: 0 0 0 4px #dcfce7; }
.step-active { background: #6382ff; color: white; box-shadow: 0 0 0 4px #e0e7ff; }
.step-idle   { background: #f1f3f9; color: #9ca3af; border: 2px solid #e5e7eb; }
.step-lbl    { font-size: 10px; font-weight: 600; color: #9ca3af; margin-top: 6px;
               text-transform: uppercase; letter-spacing: .5px; text-align: center; }
.step-lbl-active { color: #6382ff; }
.step-lbl-done   { color: #22c55e; }
.step-line  { flex: 1; height: 2px; background: #e5e7eb; min-width: 20px; }
.step-line-done { background: #22c55e; }

/* ── Upload Card ─────────────────────────────────── */
.ucard {
    background: #ffffff;
    border: 1.5px dashed #d1d5db;
    border-radius: 14px;
    padding: 18px 16px 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
    min-height: 80px;
}
.ucard:hover { border-color: #6382ff; box-shadow: 0 0 0 3px #e8ecff; }
.ucard-done  { border-color: #22c55e !important; border-style: solid !important;
               box-shadow: 0 0 0 3px #dcfce7 !important; }
.ucard-icon  { font-size: 26px; margin-right: 10px; }
.ucard-title { font-size: 14px; font-weight: 700; color: #111827; margin: 0; }
.ucard-sub   { font-size: 11px; color: #9ca3af; margin: 0; }

/* ── Pills ───────────────────────────────────────── */
.pill {
    display: inline-block;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    margin-top: 6px;
}
.pill-green  { background: #dcfce7; color: #166534; }
.pill-yellow { background: #fef9c3; color: #713f12; }
.pill-red    { background: #fee2e2; color: #991b1b; }
.pill-blue   { background: #e0e7ff; color: #3730a3; }

/* ── Score Card ──────────────────────────────────── */
.score-wrap {
    background: linear-gradient(145deg, #0f1117 0%, #1a1f35 100%);
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-wrap::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,130,255,0.18) 0%, transparent 70%);
}
.score-lbl {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    color: #6b7280; text-transform: uppercase; margin-bottom: 8px;
}
.score-num {
    font-family: 'Fraunces', serif;
    font-size: 56px; font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.score-denom { font-size: 16px; color: #6b7280; margin-bottom: 14px; }
.score-conf  { font-size: 12px; color: #818cf8; }

/* ── Metric Tile ─────────────────────────────────── */
.mtile {
    background: #f9fafb;
    border: 1px solid #f0f0f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.mtile-label { font-size: 10px; font-weight: 600; color: #9ca3af;
               text-transform: uppercase; letter-spacing: .5px; }
.mtile-value { font-size: 20px; font-weight: 700; color: #111827;
               font-family: 'DM Mono', monospace; margin-top: 2px; }

/* ── Feedback Boxes ──────────────────────────────── */
.fb-green {
    background: #f0fdf4;
    border-left: 3px solid #22c55e;
    border-radius: 0 10px 10px 0;
    padding: 12px 14px;
    margin: 6px 0;
    font-size: 13.5px;
    color: #166534;
}
.fb-orange {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 12px 14px;
    margin: 6px 0;
    font-size: 13.5px;
    color: #92400e;
}

/* ── Checklist Row ───────────────────────────────── */
.chk-row {
    display: flex; align-items: center; gap: 8px;
    background: #f9fafb;
    border: 1px solid #f0f0f0;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px; font-weight: 500;
    color: #374151;
}
.chk-ok   { border-left: 3px solid #22c55e; }
.chk-fail { border-left: 3px solid #ef4444; }

/* ── Stat Card ───────────────────────────────────── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
}
.stat-num {
    font-family: 'Fraunces', serif;
    font-size: 36px; font-weight: 700; color: #111827;
}
.stat-lbl {
    font-size: 12px; font-weight: 600; color: #9ca3af;
    text-transform: uppercase; letter-spacing: .5px; margin-top: 2px;
}

/* ── Section Header ──────────────────────────────── */
.sec-hdr {
    font-size: 13px; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 24px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}
.sec-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #f0f0f0;
}

/* ── Divider ─────────────────────────────────────── */
.ig-divider {
    border: none; border-top: 1px solid #f0f0f0;
    margin: 20px 0;
}

/* ── Text area override ──────────────────────────── */
textarea { font-family: 'DM Mono', monospace !important; font-size: 12px !important; }

/* ── Button override ─────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #6382ff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4f6cf0 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(99,130,255,0.35) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: #e5e7eb !important;
    color: #9ca3af !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Expander ────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #f0f0f0 !important;
    border-radius: 10px !important;
}

/* ── Progress bar ────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6382ff, #22c55e) !important;
    border-radius: 4px;
}

/* ── Alerts ──────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────

def api_health_check() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def api_upload_file(file_bytes: bytes, filename: str, student_code: str) -> dict:
    mime = "application/pdf" if filename.lower().endswith(".pdf") else "image/jpeg"
    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (filename, file_bytes, mime)},
        data={"student_code": student_code},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_evaluate(submission_id: int, question: str, teacher_answer: str,
                 max_marks: float, rubric: list) -> dict:
    payload = {
        "submission_id": submission_id,
        "question": question,
        "teacher_answer": teacher_answer,
        "max_marks": max_marks,
        "rubric_criteria": rubric or None,
    }
    r = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def api_stats() -> dict:
    try:
        return requests.get(f"{API_BASE}/stats", timeout=5).json()
    except Exception:
        return {}


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not PYMUPDF_AVAILABLE:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(p.get_text() for p in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def pdf_preview_image(pdf_bytes: bytes):
    if not PYMUPDF_AVAILABLE:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception:
        return None


def show_file_preview(uploaded_file, label: str, icon: str, show_text: bool = False):
    """Renders a preview thumbnail for image or PDF."""
    if uploaded_file is None:
        return
    is_pdf = uploaded_file.name.lower().endswith(".pdf")
    if is_pdf:
        img = pdf_preview_image(uploaded_file.getvalue())
        if img:
            st.image(img, caption=f"{icon} {label} — p.1", use_column_width=True)
        else:
            st.info(f"📄 {uploaded_file.name}")
        if show_text:
            txt = extract_pdf_text(uploaded_file.getvalue())
            if txt:
                with st.expander("🔍 Extracted text preview"):
                    st.code(txt[:600] + ("…" if len(txt) > 600 else ""), language=None)
    else:
        try:
            st.image(uploaded_file.getvalue(), caption=f"{icon} {label}", use_column_width=True)
        except Exception:
            st.caption(f"⚠️ Could not preview {uploaded_file.name}")


def step_tracker(steps: list, current: int):
    """Render horizontal stepper."""
    nodes = ""
    for i, lbl in enumerate(steps):
        if i < current:
            c, t, lc = "step-done", "✓", "step-lbl-done"
        elif i == current:
            c, t, lc = "step-active", str(i + 1), "step-lbl-active"
        else:
            c, t, lc = "step-idle", str(i + 1), ""
        line = ""
        if i < len(steps) - 1:
            ldc = "step-line-done" if i < current else ""
            line = f'<div class="step-line {ldc}"></div>'
        nodes += f"""
        <div class="step-wrap">
            <div class="step-circle {c}">{t}</div>
            <div class="step-lbl {lc}">{lbl}</div>
        </div>{line}"""
    st.markdown(f'<div class="stepper">{nodes}</div>', unsafe_allow_html=True)


def score_card(score: float, max_marks: float, confidence: float):
    pct = score / max_marks if max_marks else 0
    color = "#22c55e" if pct >= 0.7 else "#f59e0b" if pct >= 0.5 else "#ef4444"
    st.markdown(f"""
    <div class="score-wrap">
        <div class="score-lbl">Final Score</div>
        <div class="score-num" style="color:{color}">{score}</div>
        <div class="score-denom">out of {max_marks}</div>
        <div style="margin:8px auto;width:80%;height:5px;background:#1e2130;border-radius:4px;">
            <div style="width:{pct*100:.1f}%;height:100%;background:{color};border-radius:4px;
                        transition:width 1s ease;"></div>
        </div>
        <div class="score-conf">AI Confidence: {int(confidence * 100)}%</div>
    </div>
    """, unsafe_allow_html=True)


def metric_tile(label: str, value: str):
    st.markdown(f"""
    <div class="mtile">
        <div class="mtile-label">{label}</div>
        <div class="mtile-value">{value}</div>
    </div>""", unsafe_allow_html=True)


def section_header(icon: str, title: str):
    st.markdown(f'<div class="sec-hdr">{icon} {title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 8px 12px 8px;">
        <div style="font-family:'Fraunces',serif;font-size:22px;font-weight:700;color:#fff;
                    letter-spacing:-0.5px;">IntelliGrade‑H</div>
        <div style="font-size:10px;color:#4b5563;font-weight:600;letter-spacing:1.5px;
                    text-transform:uppercase;margin-top:2px;">Sathyabama IST</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "nav",
        ["🏠  Dashboard", "📤  Evaluate Answer", "📊  Batch Evaluation", "⚙️  Settings"],
        label_visibility="collapsed",
    )

    st.divider()

    # Live API status indicator
    alive = api_health_check()
    dot = "#22c55e" if alive else "#ef4444"
    label = "API Online" if alive else "API Offline"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;padding:0 8px;">
        <div style="width:8px;height:8px;border-radius:50%;background:{dot};
                    box-shadow:0 0 6px {dot};"></div>
        <span style="font-size:11px;font-weight:600;color:#6b7280;">{label}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:16px 8px 0 8px;">
        <div style="font-size:10px;color:#374151;font-weight:600;letter-spacing:1px;
                    text-transform:uppercase;">v1.0.0</div>
        <div style="font-size:10px;color:#4b5563;margin-top:2px;">AI-Powered Grading</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE: Dashboard
# ─────────────────────────────────────────────────────────
if page == "🏠  Dashboard":
    st.markdown('<div class="ig-title">Teacher Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="ig-sub">AI-powered evaluation of handwritten student answers</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not alive:
        st.error("⚠️  Backend API is offline.  Run: `uvicorn backend.api:app --reload --port 8000`")
    else:
        st.success("✅  Backend API is online and ready.")

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    # Stats
    stats = api_stats()
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for col, num, lbl in [
        (c1, stats.get("total_submissions", 0), "Submissions"),
        (c2, stats.get("evaluated", 0), "Evaluated"),
        (c3, stats.get("average_score", "—"), "Avg Score"),
        (c4, stats.get("average_evaluation_time_sec", "—"), "Avg Time (s)"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-num">{num}</div>
            <div class="stat-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    section_header("📖", "How It Works")
    hw1, hw2, hw3 = st.columns(3, gap="medium")
    for col, icon, title, body in [
        (hw1, "📁", "Upload 3 Documents",
         "Question Paper, Model Answer Sheet, and Student's handwritten answer — all as image or PDF."),
        (hw2, "🤖", "AI Evaluates",
         "OCR extracts text, Sentence-BERT computes similarity, Gemini scores content and generates feedback."),
        (hw3, "📊", "Get Results",
         "Receive a score, strengths, missing concepts, and detailed feedback — instantly downloadable."),
    ]:
        col.markdown(f"""
        <div style="background:#f9fafb;border:1px solid #f0f0f0;border-radius:14px;padding:20px;">
            <div style="font-size:28px;margin-bottom:10px;">{icon}</div>
            <div style="font-weight:700;font-size:14px;color:#111827;margin-bottom:6px;">{title}</div>
            <div style="font-size:13px;color:#6b7280;line-height:1.5;">{body}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE: Evaluate Answer
# ─────────────────────────────────────────────────────────
elif page == "📤  Evaluate Answer":
    st.markdown('<div class="ig-title">Evaluate Answer Sheet</div>', unsafe_allow_html=True)
    st.markdown('<div class="ig-sub">Upload all three documents and run AI evaluation</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    STEPS = ["Upload", "Parameters", "Evaluate", "Results"]
    if "eval_step" not in st.session_state:
        st.session_state.eval_step = 0

    step_tracker(STEPS, st.session_state.eval_step)

    # ══════════════════════════════════════════════════
    # STEP 1 — Upload
    # ══════════════════════════════════════════════════
    section_header("📁", "Step 1 — Upload Documents")
    st.caption("Upload all three files. Accepted: JPG, PNG, PDF.")

    z1, z2, z3 = st.columns(3, gap="medium")

    def upload_zone(col, key, icon, title, subtitle, show_text=False):
        with col:
            f = st.session_state.get(key)
            done_cls = "ucard-done" if f else ""
            st.markdown(f"""
            <div class="ucard {done_cls}">
                <div style="display:flex;align-items:center;">
                    <span class="ucard-icon">{icon}</span>
                    <div>
                        <p class="ucard-title">{title}</p>
                        <p class="ucard-sub">{subtitle}</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            uploaded = st.file_uploader(
                title, type=["jpg", "jpeg", "png", "pdf"],
                key=key, label_visibility="collapsed"
            )
            if uploaded:
                st.markdown('<span class="pill pill-green">✓ Uploaded</span>',
                            unsafe_allow_html=True)
                show_file_preview(uploaded, title, icon, show_text)
            else:
                st.markdown('<span class="pill pill-yellow">⏳ Awaiting</span>',
                            unsafe_allow_html=True)
            return uploaded

    qp_file = upload_zone(z1, "up_qp", "📋", "Question Paper",
                          "Exam question document", show_text=True)
    ma_file = upload_zone(z2, "up_ma", "📗", "Model Answer Sheet",
                          "Teacher's reference answer", show_text=True)
    sa_file = upload_zone(z3, "up_sa", "✍️", "Student Answer Sheet",
                          "Handwritten answer to grade")

    all_uploaded = bool(qp_file and ma_file and sa_file)
    if all_uploaded:
        st.success("✅  All three documents uploaded.")
        st.session_state.eval_step = max(st.session_state.eval_step, 1)
    elif any([qp_file, ma_file, sa_file]):
        missing = [n for n, f in [("Question Paper", qp_file),
                                   ("Model Answer", ma_file),
                                   ("Student Answer", sa_file)] if not f]
        st.warning(f"Still needed: **{', '.join(missing)}**")

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # STEP 2 — Parameters
    # ══════════════════════════════════════════════════
    section_header("⚙️", "Step 2 — Evaluation Parameters")

    pc1, pc2 = st.columns([3, 1], gap="medium")
    with pc1:
        student_code = st.text_input(
            "Student ID / Roll Number *",
            placeholder="e.g. SIST2024001",
            help="Stored in the database to identify this submission."
        )
        exam_name = st.text_input(
            "Exam / Subject Name",
            placeholder="e.g. Machine Learning — Unit 3 Test (optional)"
        )
    with pc2:
        max_marks = st.number_input("Max Marks *", min_value=1.0,
                                    max_value=100.0, value=10.0, step=0.5)
        q_number = st.number_input("Question No.", min_value=1,
                                   max_value=100, value=1, step=1)

    with st.expander("✏️  Override question text  (if OCR is inaccurate)"):
        manual_q = st.text_area("Type question here — leave blank to auto-extract",
                                height=80,
                                placeholder="e.g. Explain the working principle of a neural network.")

    with st.expander("✏️  Override model answer text  (if OCR is inaccurate)"):
        manual_ma = st.text_area("Type model answer here — leave blank to auto-extract",
                                 height=120,
                                 placeholder="Type the expected answer text here...")

    with st.expander("📏  Rubric Criteria  (optional)"):
        st.caption("Define specific criteria the AI checks for — leave empty to skip.")
        n_crit = st.number_input("Number of criteria", min_value=0, max_value=10, value=0, step=1)
        rubric = []
        for i in range(int(n_crit)):
            r1, r2 = st.columns([4, 1])
            crit = r1.text_input(f"Criterion {i+1}", key=f"crit_{i}",
                                 placeholder="e.g. Correct definition with example")
            mks  = r2.number_input("Marks", min_value=0.1, max_value=20.0,
                                   value=1.0, key=f"rmk_{i}")
            if crit.strip():
                rubric.append({"criterion": crit.strip(), "marks": float(mks)})

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # STEP 3 — Run Evaluation
    # ══════════════════════════════════════════════════
    section_header("🚀", "Step 3 — Run Evaluation")

    checks = {
        "📋 Question Paper uploaded": bool(qp_file),
        "📗 Model Answer uploaded":   bool(ma_file),
        "✍️ Student Answer uploaded": bool(sa_file),
        "🎓 Student ID entered":      bool(student_code and student_code.strip()),
    }
    all_ready = all(checks.values())

    chk_cols = st.columns(len(checks), gap="small")
    for col, (lbl, ok) in zip(chk_cols, checks.items()):
        ok_cls = "chk-ok" if ok else "chk-fail"
        icon   = "✅" if ok else "❌"
        col.markdown(
            f'<div class="chk-row {ok_cls}">{icon} <span style="font-size:12px">{lbl}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    eval_btn = st.button(
        "🚀  Evaluate Answer Sheet",
        type="primary",
        use_container_width=True,
        disabled=not all_ready,
    )
    if not all_ready:
        st.caption("Complete all checklist items above to enable evaluation.")

    # ══════════════════════════════════════════════════
    # Evaluation Logic
    # ══════════════════════════════════════════════════
    if eval_btn and all_ready:
        st.session_state.eval_step = 2
        st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

        # ── Extract question text ────────────────────
        prog = st.progress(0, text="Extracting text from Question Paper…")
        if manual_q and manual_q.strip():
            q_text = manual_q.strip(); q_src = "manual"
        elif qp_file.name.lower().endswith(".pdf"):
            q_text = extract_pdf_text(qp_file.getvalue()); q_src = "pdf"
        else:
            q_text = f"[Image question: {qp_file.name}]"; q_src = "image"

        # ── Extract model answer text ────────────────
        prog.progress(25, text="Extracting text from Model Answer Sheet…")
        if manual_ma and manual_ma.strip():
            ma_text = manual_ma.strip(); ma_src = "manual"
        elif ma_file.name.lower().endswith(".pdf"):
            ma_text = extract_pdf_text(ma_file.getvalue()); ma_src = "pdf"
        else:
            ma_text = f"[Image model answer: {ma_file.name}]"; ma_src = "image"

        # Warn if both are image placeholders (won't evaluate well)
        if q_src == "image" and ma_src == "image":
            st.warning("⚠️  Both Question Paper and Model Answer are images — "
                       "OCR will be used but accuracy may be limited. "
                       "Consider uploading PDFs or using the manual override fields above.")

        # ── Upload student sheet ─────────────────────
        prog.progress(50, text="Uploading student answer sheet…")
        try:
            up = api_upload_file(sa_file.getvalue(), sa_file.name, student_code.strip())
            sid = up["submission_id"]
        except requests.HTTPError as e:
            prog.empty()
            st.error(f"Upload failed: {e.response.text}")
            st.stop()
        except Exception as e:
            prog.empty()
            st.error(f"Upload failed: {e}")
            st.stop()

        # ── Run AI evaluation ────────────────────────
        prog.progress(75, text="Running AI evaluation — please wait (30–90 s)…")
        try:
            result = api_evaluate(sid, q_text, ma_text, max_marks,
                                  rubric if rubric else None)
        except requests.HTTPError as e:
            prog.empty()
            st.error(f"Evaluation error: {e.response.text}")
            st.stop()
        except Exception as e:
            prog.empty()
            st.error(f"Evaluation failed: {e}")
            st.stop()

        prog.progress(100, text="Done!")
        prog.empty()

        st.session_state.eval_step = 3
        step_tracker(STEPS, 3)
        st.success("✅  Evaluation complete!")

        # ══════════════════════════════════════════════
        # STEP 4 — Results
        # ══════════════════════════════════════════════
        section_header("📊", "Step 4 — Results")

        res_a, res_b, res_c = st.columns([1, 1, 1], gap="medium")

        with res_a:
            score_card(result["final_score"], result["max_marks"], result["confidence"])

        with res_b:
            metric_tile("LLM Score",
                        f"{result['llm_score']:.1f} / {result['max_marks']}")
            metric_tile("Semantic Similarity",
                        f"{result['similarity_score']:.2%}")
            metric_tile("OCR Confidence",
                        f"{result['ocr_confidence']:.2%}")
            metric_tile("Evaluation Time",
                        f"{result['evaluation_time_sec']} s")

        with res_c:
            rd = result.get("rubric_details")
            if rd:
                metric_tile("Rubric Coverage",
                            f"{rd['earned_rubric_marks']:.1f} / {rd['total_rubric_marks']:.1f}")
                for criterion, present in rd.get("criteria_scores", {}).items():
                    ic = "✅" if present else "❌"
                    st.markdown(
                        f'<div style="font-size:12px;padding:4px 0;color:#374151;">'
                        f'{ic} {criterion}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("""
                <div style="background:#f9fafb;border:1px solid #f0f0f0;border-radius:12px;
                             padding:20px;text-align:center;color:#9ca3af;font-size:13px;">
                    No rubric criteria<br>were defined.
                </div>""", unsafe_allow_html=True)

        # ── Extracted Texts ──────────────────────────
        with st.expander("📄  View All Extracted Texts"):
            ta, tb, tc = st.columns(3, gap="small")
            with ta:
                st.markdown("**📋 Question Paper**")
                st.text_area("", value=q_text, height=160, disabled=True, key="disp_q")
                st.caption(f"Source: {q_src}")
            with tb:
                st.markdown("**📗 Model Answer**")
                st.text_area("", value=ma_text, height=160, disabled=True, key="disp_ma")
                st.caption(f"Source: {ma_src}")
            with tc:
                st.markdown("**✍️ Student Answer (OCR)**")
                st.text_area("", value=result.get("ocr_text", ""),
                             height=160, disabled=True, key="disp_sa")
                st.caption(f"Engine: {result.get('ocr_engine', 'N/A')}")

        # ── Strengths & Missing ──────────────────────
        fb_col, ms_col = st.columns(2, gap="medium")
        with fb_col:
            section_header("💪", "Strengths")
            strengths = result.get("strengths", [])
            if strengths:
                for s in strengths:
                    st.markdown(f'<div class="fb-green">✅ {s}</div>',
                                unsafe_allow_html=True)
            else:
                st.caption("No specific strengths noted.")

        with ms_col:
            section_header("⚠️", "Missing Concepts")
            missing = result.get("missing_concepts", [])
            if missing:
                for m in missing:
                    st.markdown(f'<div class="fb-orange">⚠️ {m}</div>',
                                unsafe_allow_html=True)
            else:
                st.caption("No missing concepts detected.")

        # ── AI Feedback ──────────────────────────────
        section_header("💡", "AI Feedback")
        st.markdown(f"""
        <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:12px;
                    padding:18px 20px;font-size:14px;line-height:1.65;color:#1e40af;">
            {result.get("feedback", "No feedback generated.")}
        </div>""", unsafe_allow_html=True)

        # ── Downloads ────────────────────────────────
        st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)
        section_header("⬇️", "Download Results")

        dl1, dl2 = st.columns(2, gap="medium")

        json_data = json.dumps({
            **result,
            "student_id": student_code,
            "exam": exam_name or "",
            "question_number": q_number,
        }, indent=2)

        report_txt = "\n".join([
            "IntelliGrade-H — Evaluation Report",
            "=" * 45,
            f"Student ID    : {student_code}",
            f"Exam          : {exam_name or 'N/A'}",
            f"Question No   : {q_number}",
            f"Max Marks     : {result['max_marks']}",
            f"Final Score   : {result['final_score']}",
            f"LLM Score     : {result['llm_score']}",
            f"Similarity    : {result['similarity_score']:.2%}",
            f"OCR Conf.     : {result['ocr_confidence']:.2%}",
            f"Eval Time     : {result['evaluation_time_sec']} s",
            "=" * 45,
            "STRENGTHS:",
            *[f"  + {s}" for s in result.get("strengths", [])],
            "",
            "MISSING CONCEPTS:",
            *[f"  - {m}" for m in result.get("missing_concepts", [])],
            "",
            "AI FEEDBACK:",
            f"  {result.get('feedback', '')}",
        ])

        dl1.download_button("⬇️  Download JSON",
                            data=json_data,
                            file_name=f"result_{student_code}_Q{q_number}.json",
                            mime="application/json",
                            use_container_width=True)
        dl2.download_button("⬇️  Download TXT Report",
                            data=report_txt,
                            file_name=f"report_{student_code}_Q{q_number}.txt",
                            mime="text/plain",
                            use_container_width=True)


# ─────────────────────────────────────────────────────────
# PAGE: Batch Evaluation
# ─────────────────────────────────────────────────────────
elif page == "📊  Batch Evaluation":
    st.markdown('<div class="ig-title">Batch Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="ig-sub">Evaluate an entire class in one go</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Reference Documents ──────────────────────────
    section_header("📁", "Reference Documents")
    st.caption("Upload the question paper and model answer — shared across all students.")

    bc1, bc2 = st.columns(2, gap="medium")

    def batch_upload_zone(col, key, icon, title, subtitle):
        with col:
            f = st.session_state.get(key)
            done_cls = "ucard-done" if f else ""
            st.markdown(f"""
            <div class="ucard {done_cls}">
                <div style="display:flex;align-items:center;">
                    <span class="ucard-icon">{icon}</span>
                    <div>
                        <p class="ucard-title">{title}</p>
                        <p class="ucard-sub">{subtitle}</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            uploaded = st.file_uploader(title, type=["jpg","jpeg","png","pdf"],
                                        key=key, label_visibility="collapsed")
            if uploaded:
                st.markdown('<span class="pill pill-green">✓ Uploaded</span>',
                            unsafe_allow_html=True)
                show_file_preview(uploaded, title, icon, show_text=True)
            else:
                st.markdown('<span class="pill pill-yellow">⏳ Awaiting</span>',
                            unsafe_allow_html=True)
            return uploaded

    b_qp = batch_upload_zone(bc1, "bup_qp", "📋", "Question Paper", "Same for all students")
    b_ma = batch_upload_zone(bc2, "bup_ma", "📗", "Model Answer Sheet", "Teacher's reference")

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    # ── Student Sheets ───────────────────────────────
    section_header("✍️", "Student Answer Sheets")
    st.caption("Name files as Roll Number (e.g. SIST2024001.jpg). Filename → Student ID.")

    b_students = st.file_uploader(
        "Upload student answer sheets",
        type=["jpg","jpeg","png","pdf"],
        accept_multiple_files=True,
        key="bup_students",
        label_visibility="collapsed",
    )
    if b_students:
        st.markdown(
            f'<span class="pill pill-blue">📚 {len(b_students)} file(s) loaded</span>',
            unsafe_allow_html=True
        )
        # Thumbnail strip (up to 6)
        n_show = min(len(b_students), 6)
        thumb_cols = st.columns(n_show)
        for i, f in enumerate(b_students[:n_show]):
            with thumb_cols[i]:
                try:
                    st.image(f.getvalue(), caption=Path(f.name).stem,
                             use_column_width=True)
                except Exception:
                    st.caption(f.name)
        if len(b_students) > n_show:
            st.caption(f"… and {len(b_students) - n_show} more")

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    # ── Parameters ───────────────────────────────────
    section_header("⚙️", "Parameters")
    bpc1, bpc2 = st.columns([3, 1], gap="medium")
    with bpc1:
        b_exam = st.text_input("Exam / Subject Name",
                               placeholder="e.g. AI — Unit 2 Test", key="b_exam")
    with bpc2:
        b_max  = st.number_input("Max Marks", value=10.0, step=0.5, key="b_max")

    with st.expander("✏️  Override question text"):
        b_mq = st.text_area("Leave blank to extract from question paper file",
                            height=70, key="b_mq")
    with st.expander("✏️  Override model answer text"):
        b_mma = st.text_area("Leave blank to extract from model answer file",
                             height=90, key="b_mma")

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    batch_ready = bool(b_qp and b_ma and b_students)
    if not batch_ready:
        st.warning("Upload the question paper, model answer, and at least one student sheet.")

    if st.button("🚀  Run Batch Evaluation", type="primary",
                 disabled=not batch_ready, use_container_width=True):

        # Extract shared texts
        b_q_text  = (b_mq.strip() if b_mq and b_mq.strip() else
                     extract_pdf_text(b_qp.getvalue()) if b_qp.name.lower().endswith(".pdf")
                     else f"[Image: {b_qp.name}]")
        b_ma_text = (b_mma.strip() if b_mma and b_mma.strip() else
                     extract_pdf_text(b_ma.getvalue()) if b_ma.name.lower().endswith(".pdf")
                     else f"[Image: {b_ma.name}]")

        prog_bar   = st.progress(0)
        status_box = st.empty()
        results    = []

        for i, f in enumerate(b_students):
            sid_name = Path(f.name).stem
            status_box.info(f"⏳  Evaluating **{sid_name}** ({i+1} / {len(b_students)})…")
            try:
                up  = api_upload_file(f.getvalue(), f.name, sid_name)
                res = api_evaluate(up["submission_id"], b_q_text, b_ma_text,
                                   b_max, None)
                pct = (res["final_score"] / b_max * 100) if b_max else 0
                results.append({
                    "Student ID"      : sid_name,
                    "Score"           : res["final_score"],
                    "Max Marks"       : b_max,
                    "% Score"         : f"{pct:.1f}%",
                    "LLM Score"       : res["llm_score"],
                    "Similarity"      : f"{res['similarity_score']:.2%}",
                    "OCR Confidence"  : f"{res['ocr_confidence']:.2%}",
                    "Eval Time (s)"   : res["evaluation_time_sec"],
                    "Feedback"        : (res.get("feedback", "") or "")[:100] + "…",
                })
            except Exception as e:
                results.append({
                    "Student ID"     : sid_name,
                    "Score"          : "ERROR",
                    "Max Marks"      : b_max,
                    "% Score"        : "—",
                    "LLM Score"      : "—",
                    "Similarity"     : "—",
                    "OCR Confidence" : "—",
                    "Eval Time (s)"  : "—",
                    "Feedback"       : str(e),
                })
            prog_bar.progress((i + 1) / len(b_students))

        prog_bar.empty()
        status_box.success(f"✅  Batch complete — {len(b_students)} submission(s) processed.")

        # Summary stats
        valid = [r["Score"] for r in results if isinstance(r["Score"], (int, float))]
        if valid:
            s1, s2, s3, s4 = st.columns(4, gap="medium")
            for col, num, lbl in [
                (s1, len(b_students), "Total"),
                (s2, f"{sum(valid)/len(valid):.2f}", "Average"),
                (s3, f"{max(valid):.2f}", "Highest"),
                (s4, f"{min(valid):.2f}", "Lowest"),
            ]:
                col.markdown(f"""
                <div class="stat-card">
                    <div class="stat-num">{num}</div>
                    <div class="stat-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True, hide_index=True)

        # CSV download
        buf = io.StringIO()
        if results:
            writer = csv.DictWriter(buf, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        st.download_button(
            "⬇️  Download Results CSV",
            data=buf.getvalue(),
            file_name=f"batch_{b_exam or 'results'}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────
# PAGE: Settings
# ─────────────────────────────────────────────────────────
elif page == "⚙️  Settings":
    st.markdown('<div class="ig-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="ig-sub">System configuration and diagnostics</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        section_header("🔌", "API Configuration")
        st.text_input("API Base URL", value=API_BASE,
                      help="URL where the FastAPI backend is running.")
        st.text_input("Gemini API Key", type="password",
                      placeholder="Paste your key — stored only in .env")
        st.caption("Get a free key at https://aistudio.google.com/")

        section_header("🤖", "OCR Engine")
        ocr_choice = st.selectbox(
            "Select OCR Engine",
            ["tesseract", "trocr"],
            help="Tesseract is fast and lightweight. TrOCR is more accurate but requires ~1.5 GB download on first run."
        )
        st.caption("After changing, update `OCR_ENGINE` in your `.env` and restart the API.")

    with sc2:
        section_header("⚖️", "Scoring Weights")
        llm_w = st.slider("LLM Weight", 0.0, 1.0, 0.6, 0.05,
                          help="How much the Gemini LLM score contributes.")
        sim_w = st.slider("Similarity Weight", 0.0, 1.0, 0.4, 0.05,
                          help="How much semantic similarity contributes.")
        total = llm_w + sim_w
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f}. Recommended: sum = 1.0")
        st.code(
            f"Final = {llm_w:.2f} × LLM_Score\n"
            f"      + {sim_w:.2f} × Similarity × MaxMarks",
            language="python"
        )

        section_header("📊", "System Diagnostics")
        stats = api_stats()
        st.json({
            "api_status"        : "online" if alive else "offline",
            "total_submissions" : stats.get("total_submissions", 0),
            "evaluated"         : stats.get("evaluated", 0),
            "avg_score"         : stats.get("average_score", 0),
            "avg_eval_time_s"   : stats.get("average_evaluation_time_sec", 0),
            "ocr_engine"        : ocr_choice,
            "pymupdf_available" : PYMUPDF_AVAILABLE,
            "version"           : "1.0.0",
        })

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    if st.button("💾  Save Settings", type="primary"):
        st.success("Settings noted. Update your `.env` file and restart the API server to apply.")