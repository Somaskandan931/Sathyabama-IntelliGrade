"""
IntelliGrade-H — Teacher Dashboard (v9 — Production Fixed)
===========================================================
Fixes vs v8:
  • Timeout: replaced single 600s blocking request with async polling loop
    (uploads/OCR use short timeouts; /evaluate fires async, UI polls /result/{id})
  • max_marks: extracted from question paper OCR text via LLM, then passed to
    /evaluate so the score denominator is correct (2 for Part A, 12 for Part B)
  • Claude 400 is fixed in backend/llm_provider.py (model env var + correct API format)
  • Batch: uses same polling approach per student
  • Student Code field moved inside the upload card
"""

import streamlit as st
import os, json, time, requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Timeouts ───────────────────────────────────────────────────────────────────
UPLOAD_TIMEOUT   = 30    # seconds — file upload
OCR_TIMEOUT      = 180   # seconds — OCR (3 min max per doc)
EVALUATE_TIMEOUT = 180   # seconds — wait for evaluate response (LLM can take 2-3 min)
POLL_TIMEOUT     = 600   # seconds — total wait for result polling
POLL_INTERVAL    = 4     # seconds — between polls

st.set_page_config(
    page_title="IntelliGrade-H",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #21262d;
    --accent:    #1f6feb;
    --accent2:   #388bfd;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --success:   #3fb950;
    --danger:    #f85149;
    --warn:      #d29922;
    --radius:    10px;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--muted) !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.83rem !important; padding: 0.1rem 0; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 0.1rem !important; }

.main > div { background: var(--bg) !important; }
.main .block-container { padding: 2.5rem 2.5rem 4rem !important; max-width: 1100px !important; }

.logo-block { padding: 1.75rem 1.5rem 1.25rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.logo-wordmark { font-size: 1.25rem; font-weight: 700; color: var(--text) !important; letter-spacing: -0.3px; }
.logo-wordmark span { color: var(--accent2) !important; }
.logo-inst { font-size: 0.65rem; font-weight: 400; color: var(--muted) !important; letter-spacing: 1.2px; text-transform: uppercase; margin-top: 2px; }

.status-row { padding: 0 1.5rem; margin-bottom: 0.5rem; }
.status-dot { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.73rem; font-weight: 500; color: var(--muted) !important; }
.dot-on  { width: 7px; height: 7px; border-radius: 50%; background: var(--success); box-shadow: 0 0 6px var(--success); }
.dot-off { width: 7px; height: 7px; border-radius: 50%; background: var(--danger);  box-shadow: 0 0 6px var(--danger);  }

.pg-title { font-size: 1.6rem; font-weight: 700; color: var(--text); letter-spacing: -0.5px; margin-bottom: 0.2rem; }
.pg-sub   { font-size: 0.82rem; color: var(--muted); margin-bottom: 1.75rem; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem 1.5rem;
    margin-bottom: 1.1rem;
}
.card-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}
.upload-cap { font-size: 0.75rem; font-weight: 600; color: var(--muted); margin-bottom: 0.35rem; letter-spacing: 0.2px; }

.info-banner {
    display: flex; align-items: center; gap: 0.6rem;
    background: #0e1e35; border: 1px solid #1f6feb33;
    border-radius: 8px; padding: 0.65rem 1rem;
    font-size: 0.8rem; color: #7cb9f7; font-weight: 400;
    margin-bottom: 1.1rem;
}

[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent2) !important;
    background: #111926 !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p { color: var(--muted) !important; font-size: 0.8rem !important; }
[data-testid="stFileUploader"] button { display: none !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    height: 2.7rem !important;
    letter-spacing: 0.2px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: var(--accent2) !important; }

.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(31,111,235,0.25) !important;
}
label { color: var(--muted) !important; font-size: 0.8rem !important; font-weight: 500 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--accent2) !important;
}

.score-hero {
    background: linear-gradient(135deg, #0d1117, #0e2a4d 60%, #1a4a8a);
    border: 1px solid #1f6feb44;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1.25rem 0;
}
.score-num { font-size: 4rem; font-weight: 700; color: #fff; line-height: 1; letter-spacing: -2px; }
.score-denom { font-size: 1.6rem; opacity: 0.4; font-weight: 300; }
.score-pct  { font-size: 0.9rem; color: #93c5fd; margin-top: 0.5rem; letter-spacing: 0.5px; }
.score-type { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #475569; margin-top: 0.35rem; letter-spacing: 2px; text-transform: uppercase; }

.feedback-box {
    background: #161b22;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.86rem;
    line-height: 1.7;
    color: #c9d1d9;
    margin: 0.9rem 0;
}

.pill-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.6rem 0; }
.pill { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.73rem; font-weight: 500; }
.pill-g { background: #0d2a1a; color: #3fb950; border: 1px solid #238636; }
.pill-r { background: #2a0e0d; color: #f85149; border: 1px solid #6e1a18; }

.mtile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1rem; text-align: center; }
.mtile-val   { font-size: 1.75rem; font-weight: 700; color: var(--text); line-height: 1; }
.mtile-label { font-size: 0.67rem; color: var(--muted); margin-top: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; }

.step-msg {
    display: flex; align-items: center; gap: 0.6rem;
    background: #0e2235; border: 1px solid #1f6feb44;
    border-radius: 8px; padding: 0.7rem 1rem;
    font-size: 0.83rem; color: #93c5fd; font-weight: 500; margin: 0.4rem 0;
}
.poll-msg {
    display: flex; align-items: center; gap: 0.6rem;
    background: #1a1a0e; border: 1px solid #d2992233;
    border-radius: 8px; padding: 0.7rem 1rem;
    font-size: 0.83rem; color: #d29922; font-weight: 500; margin: 0.4rem 0;
}

.stCodeBlock { border-radius: 8px !important; }
.streamlit-expanderHeader { color: var(--muted) !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.streamlit-expanderContent { background: var(--bg) !important; border-color: var(--border) !important; }
[data-testid="stDataFrame"] { border-radius: 8px !important; border: 1px solid var(--border) !important; }
.stProgress > div > div { background: var(--accent2) !important; border-radius: 999px !important; }
.stProgress { background: var(--border) !important; border-radius: 999px !important; }
hr { border-color: var(--border) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
.stAlert { border-radius: 8px !important; font-size: 0.83rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <div class="logo-wordmark">Intelli<span>Grade</span>-H</div>
        <div class="logo-inst">Sathyabama Institute</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        ok = requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except Exception:
        ok = False

    dot = 'dot-on' if ok else 'dot-off'
    label = 'API Online' if ok else 'API Offline'
    st.markdown(f'<div class="status-row"><span class="status-dot"><span class="{dot}"></span>{label}</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="padding:0 1.5rem 1rem;">', unsafe_allow_html=True)
    page = st.radio("nav", ["Single Grade", "Batch Grade", "Analytics", "Settings"], label_visibility="collapsed")
    if not ok:
        st.caption("`uvicorn backend.api:app --reload`")
    st.markdown('</div>', unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def upload_file(f, code="TEMP"):
    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (f.name, f.getvalue(), f.type or "image/jpeg")},
        data={"student_code": code},
        timeout=UPLOAD_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def run_ocr(sid):
    r = requests.post(f"{API_BASE}/ocr/{sid}", timeout=OCR_TIMEOUT)
    r.raise_for_status()
    return r.json().get("extracted_text", "")


def fire_evaluate(sid, question, teacher_answer, max_marks=None):
    """
    Fire the /evaluate request. Returns the result dict if received within
    EVALUATE_TIMEOUT, otherwise raises ReadTimeout so the caller can poll.
    """
    payload = {
        "submission_id": sid,
        "question": question,
        "teacher_answer": teacher_answer,
        "question_type": "auto",
    }
    if max_marks is not None:
        payload["max_marks"] = max_marks

    r = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=EVALUATE_TIMEOUT)
    r.raise_for_status()
    return r.json()


def poll_result(sid, ph):
    """
    Poll /result/{sid} until ready or POLL_TIMEOUT exceeded.
    Shows a live countdown in the placeholder ph.
    """
    deadline = time.time() + POLL_TIMEOUT
    elapsed  = 0
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_BASE}/result/{sid}", timeout=10)
            if r.status_code == 200:
                ph.empty()
                return r.json()
        except Exception:
            pass
        remaining = int(deadline - time.time())
        ph.markdown(
            f'<div class="poll-msg">⏳ AI is evaluating… {elapsed}s elapsed '
            f'(max {POLL_TIMEOUT}s) — please wait</div>',
            unsafe_allow_html=True,
        )
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    ph.empty()
    return None


def extract_max_marks(question_text: str) -> float:
    """
    Try to extract max marks from question paper OCR text.
    Looks for patterns like "2 Marks Each", "12 Marks Each", "(7 x 2=14)" etc.
    Returns the per-question mark value, defaulting to 10 if not found.
    """
    import re
    text = question_text.lower()

    # Pattern: "part b (12 marks each)" or "12 marks each"
    m = re.search(r'(\d+)\s*marks?\s*each', text)
    if m:
        return float(m.group(1))

    # Pattern: "x 2 = 14" → 2 marks each
    m = re.search(r'x\s*(\d+)\s*=\s*\d+', text)
    if m:
        return float(m.group(1))

    # Pattern: "max.*marks.*50" or "max. marks: 50"
    m = re.search(r'max[\.\s]*marks[\s:]*(\d+)', text)
    if m:
        return float(m.group(1))

    return 10.0  # safe default


def show_error(e):
    if isinstance(e, requests.exceptions.ReadTimeout):
        st.error("⏱️ Request timed out. The backend may still be processing. Try clicking Grade again in ~30 seconds.")
    elif isinstance(e, requests.exceptions.ConnectionError):
        st.error("❌ Backend offline. Run: `uvicorn backend.api:app --reload`")
    elif isinstance(e, requests.exceptions.HTTPError):
        try:
            d = e.response.json().get("detail", str(e))
        except Exception:
            d = str(e)
        st.error(f"Backend error: {d}")
    else:
        st.error(f"Error: {e}")


# ── Result renderer ───────────────────────────────────────────────────────────

def show_result(res, label="Student"):
    score = res.get("final_score", 0)
    maxm  = res.get("max_marks", 10)
    pct   = score / maxm * 100 if maxm else 0
    qt    = res.get("question_type", "open_ended")
    fb    = res.get("feedback", "")

    st.markdown(f"""
    <div class="score-hero">
        <div class="score-num">{score:.1f}<span class="score-denom"> /{maxm:.0f}</span></div>
        <div class="score-pct">{pct:.0f}% &nbsp;·&nbsp; {label}</div>
        <div class="score-type">{qt.replace('_',' ')}</div>
    </div>""", unsafe_allow_html=True)

    if qt in ("mcq", "true_false"):
        detected = res.get("mcq_detected_answer", "?")
        correct  = res.get("mcq_correct_answer", "?")
        icon = "✅" if res.get("mcq_correct") else "❌"
        st.markdown(
            f'<div class="feedback-box">{icon} &nbsp; Detected: <strong>{detected}</strong>'
            f' &nbsp;·&nbsp; Correct: <strong>{correct}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="mtile"><div class="mtile-val">{res.get("llm_score",0):.1f}</div><div class="mtile-label">LLM Score</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="mtile"><div class="mtile-val">{res.get("similarity_score",0):.2f}</div><div class="mtile-label">Similarity</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="mtile"><div class="mtile-val">{res.get("ocr_confidence",0):.0%}</div><div class="mtile-label">OCR Conf.</div></div>', unsafe_allow_html=True)

    if fb:
        st.markdown(f'<div class="feedback-box">{fb}</div>', unsafe_allow_html=True)

    s = res.get("strengths", [])
    m = res.get("missing_concepts", [])
    if s:
        st.markdown('<div class="pill-row">' + ''.join(f'<span class="pill pill-g">✓ {x}</span>' for x in s) + '</div>', unsafe_allow_html=True)
    if m:
        st.markdown('<div class="pill-row">' + ''.join(f'<span class="pill pill-r">✗ {x}</span>' for x in m) + '</div>', unsafe_allow_html=True)

    st.caption(f"Confidence: {res.get('confidence',0):.0%}  ·  Time: {res.get('evaluation_time_sec',0):.1f}s")

    with st.expander("Extracted OCR text"):
        st.code(res.get("ocr_text", "(empty)"), language=None)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE GRADE
# ─────────────────────────────────────────────────────────────────────────────

if page == "Single Grade":
    st.markdown('<div class="pg-title">Single Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload three documents — question paper, answer key, and student sheet — then click Grade.</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-banner">🤖 &nbsp; Question type and marks are automatically detected from the question paper. No manual configuration needed.</div>', unsafe_allow_html=True)

    # ── Upload card ───────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">Documents</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="upload-cap">📋 Question Paper</div>', unsafe_allow_html=True)
        qp = st.file_uploader("qp", type=["jpg","jpeg","png","bmp","tiff","pdf"], key="qp", label_visibility="collapsed")
    with c2:
        st.markdown('<div class="upload-cap">📝 Model Answer Key</div>', unsafe_allow_html=True)
        ak = st.file_uploader("ak", type=["jpg","jpeg","png","bmp","tiff","pdf"], key="ak", label_visibility="collapsed")
    with c3:
        st.markdown('<div class="upload-cap">✍️ Student Answer Sheet</div>', unsafe_allow_html=True)
        sa = st.file_uploader("sa", type=["jpg","jpeg","png","bmp","tiff","pdf"], key="sa", label_visibility="collapsed")

    stu_code = st.text_input("Student Code", value="STU001")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Grade ─────────────────────────────────────────────────────────────────
    if st.button("⚡  Grade Answer Sheet", type="primary", use_container_width=True):
        if not ok:
            st.error("❌ Backend offline. Run: `uvicorn backend.api:app --reload`")
        elif not sa:
            st.warning("Please upload the student answer sheet.")
        else:
            ph = st.empty()
            try:
                # Step 1: OCR question paper
                q_text = ""
                detected_marks = None
                if qp:
                    ph.markdown('<div class="step-msg">⏳ Step 1 / 3 &nbsp;— Extracting question paper…</div>', unsafe_allow_html=True)
                    q_text = run_ocr(upload_file(qp)["submission_id"])
                    detected_marks = extract_max_marks(q_text)

                # Step 2: OCR answer key
                a_text = ""
                if ak:
                    ph.markdown('<div class="step-msg">⏳ Step 2 / 3 &nbsp;— Extracting answer key…</div>', unsafe_allow_html=True)
                    a_text = run_ocr(upload_file(ak)["submission_id"])

                # Step 3: Upload student sheet + fire evaluation
                ph.markdown('<div class="step-msg">⏳ Step 3 / 3 &nbsp;— Uploading student sheet…</div>', unsafe_allow_html=True)
                up = upload_file(sa, stu_code)
                student_sid = up["submission_id"]

                ph.markdown('<div class="step-msg">⏳ Running AI evaluation… this may take 1–3 minutes on first run</div>', unsafe_allow_html=True)

                # Try direct evaluate; if it times out, poll /result/{id}
                res = None
                try:
                    res = fire_evaluate(student_sid, q_text, a_text, detected_marks)
                    ph.empty()
                except requests.exceptions.ReadTimeout:
                    # Backend is still processing — poll for result
                    res = poll_result(student_sid, ph)
                    if res is None:
                        st.error("⏱️ Evaluation is taking unusually long. Check backend logs. The result may still arrive — try refreshing.")
                        st.stop()
                except requests.exceptions.HTTPError as http_err:
                    # 5xx from backend while processing — still try polling
                    if http_err.response is not None and http_err.response.status_code >= 500:
                        res = poll_result(student_sid, ph)
                    if res is None:
                        raise

                st.session_state["s_result"]  = res
                st.session_state["s_student"] = stu_code
                if detected_marks:
                    st.session_state["s_marks"] = detected_marks

                st.success(f"✅ Evaluation complete — max marks auto-detected: {detected_marks or 'N/A'}")

            except Exception as e:
                ph.empty()
                show_error(e)

    if st.session_state.get("s_result"):
        st.divider()
        show_result(st.session_state["s_result"], label=st.session_state.get("s_student", "Student"))
        st.download_button(
            "⬇️  Download JSON",
            data=json.dumps(st.session_state["s_result"], indent=2),
            file_name="result.json",
            mime="application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH GRADE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Batch Grade":
    st.markdown('<div class="pg-title">Batch Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Grade a whole class at once. Name student files with an ID prefix: <code>S001_ans.jpg</code></div>', unsafe_allow_html=True)

    st.markdown('<div class="info-banner">🤖 &nbsp; Question type and marks are automatically detected from the question paper. No manual configuration needed.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Shared Documents</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="upload-cap">📋 Question Paper</div>', unsafe_allow_html=True)
        b_qp = st.file_uploader("bqp", type=["jpg","jpeg","png","bmp","tiff","pdf"], key="b_qp", label_visibility="collapsed")
    with b2:
        st.markdown('<div class="upload-cap">📝 Model Answer Key</div>', unsafe_allow_html=True)
        b_ak = st.file_uploader("bak", type=["jpg","jpeg","png","bmp","tiff","pdf"], key="b_ak", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Student Answer Sheets</div>', unsafe_allow_html=True)
    b_sheets = st.file_uploader(
        "Drop all student answer sheets here",
        type=["jpg","jpeg","png","bmp","tiff","pdf"],
        accept_multiple_files=True,
        key="b_sheets",
        label_visibility="collapsed",
    )
    if b_sheets:
        st.caption(f"✅ {len(b_sheets)} file(s) ready")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡  Grade All Students", type="primary", use_container_width=True, key="b_go"):
        if not ok:
            st.error("❌ Backend offline.")
        elif not b_sheets:
            st.warning("Upload student answer sheets first.")
        else:
            prog = st.progress(0, text="Starting…")
            results = []
            q_text = a_text = ""
            detected_marks = None

            # OCR shared docs
            try:
                if b_qp:
                    prog.progress(0, text="OCR: question paper…")
                    q_text = run_ocr(upload_file(b_qp)["submission_id"])
                    detected_marks = extract_max_marks(q_text)
                if b_ak:
                    prog.progress(0, text="OCR: answer key…")
                    a_text = run_ocr(upload_file(b_ak)["submission_id"])
            except Exception as e:
                prog.empty()
                show_error(e)
                st.stop()

            # Grade each student
            status_ph = st.empty()
            for i, f in enumerate(b_sheets):
                sid_label = f.name.split("_")[0] if "_" in f.name else f.name.rsplit(".", 1)[0]
                prog.progress((i + 1) / len(b_sheets), text=f"Grading {sid_label}  ({i+1}/{len(b_sheets)})…")
                try:
                    up = upload_file(f, sid_label)
                    student_sid = up["submission_id"]

                    # Fire evaluate
                    try:
                        r = fire_evaluate(student_sid, q_text, a_text, detected_marks)
                    except requests.exceptions.ReadTimeout:
                        # Poll for result
                        r = poll_result(student_sid, status_ph)
                    except requests.exceptions.HTTPError as http_err:
                        if http_err.response is not None and http_err.response.status_code >= 500:
                            r = poll_result(student_sid, status_ph)
                        else:
                            raise

                    if r:
                        results.append({"id": sid_label, "result": r})
                    else:
                        results.append({"id": sid_label, "result": None, "error": "Timed out"})
                except Exception as e:
                    emsg = str(e)
                    results.append({"id": sid_label, "result": None, "error": emsg})

            prog.empty()
            status_ph.empty()
            n_ok = sum(1 for r in results if r["result"])
            st.success(f"✅ Done — {n_ok} / {len(results)} graded successfully")
            if detected_marks:
                st.caption(f"Auto-detected max marks per question: {detected_marks}")

            rows = []
            for item in results:
                r = item.get("result")
                maxm = r.get("max_marks", detected_marks or "?") if r else "—"
                rows.append({
                    "Student": item["id"],
                    "Score": f"{r.get('final_score',0):.1f}" if r else "ERR",
                    "Max": maxm,
                    "%": f"{r.get('final_score',0)/r.get('max_marks',1)*100:.0f}%" if r and r.get('max_marks') else "—",
                    "Type": r.get("question_type", "—") if r else "—",
                    "Confidence": f"{r.get('confidence',0):.0%}" if r else "—",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇️  Download CSV", data=df.to_csv(index=False), file_name="batch_results.csv", mime="text/csv")

            st.divider()
            for item in results:
                r = item.get("result")
                lbl = f"{r.get('final_score',0):.1f} / {r.get('max_marks','?')}" if r else "ERROR"
                with st.expander(f"📄  {item['id']}  —  {lbl}"):
                    if r:
                        show_result(r, label=item["id"])
                    else:
                        st.error(item.get("error", "Unknown error"))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Analytics":
    st.markdown('<div class="pg-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">System statistics and AI scoring accuracy.</div>', unsafe_allow_html=True)

    if not ok:
        st.warning("Backend is offline. Start it to view analytics.")
    else:
        if st.button("🔄 Refresh"):
            st.rerun()
        try:
            stats = requests.get(f"{API_BASE}/stats", timeout=10).json()
            t1, t2, t3, t4 = st.columns(4)
            t1.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("total_submissions",0)}</div><div class="mtile-label">Submissions</div></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("evaluated",0)}</div><div class="mtile-label">Evaluated</div></div>', unsafe_allow_html=True)
            t3.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("average_score",0):.1f}</div><div class="mtile-label">Avg Score</div></div>', unsafe_allow_html=True)
            t4.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("average_evaluation_time_sec",0):.1f}s</div><div class="mtile-label">Avg Time</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Stats error: {e}")

        st.divider()
        st.markdown('<div class="card-label" style="font-size:0.65rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#8b949e;">AI Accuracy Metrics</div>', unsafe_allow_html=True)
        try:
            mx = requests.get(f"{API_BASE}/metrics", timeout=10).json()
            st.caption(f"Updated: {mx.get('last_updated','—')}  ·  Total: {mx.get('total_evaluated',0)}")
            mcq_m = mx.get("mcq")
            oe_m  = mx.get("open_ended")
            if mcq_m:
                st.markdown("**MCQ**")
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="mtile"><div class="mtile-val">{mcq_m.get("accuracy_pct",0):.1f}%</div><div class="mtile-label">Accuracy</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="mtile"><div class="mtile-val">{mcq_m.get("n_correct",0)}</div><div class="mtile-label">Correct</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="mtile"><div class="mtile-val">{mcq_m.get("n_wrong",0)}</div><div class="mtile-label">Wrong</div></div>', unsafe_allow_html=True)
            if oe_m:
                st.markdown("**Open-Ended**")
                o1, o2, o3 = st.columns(3)
                o1.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("mae",0):.3f}</div><div class="mtile-label">MAE</div></div>', unsafe_allow_html=True)
                o2.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("pearson_r",0):.3f}</div><div class="mtile-label">Pearson r</div></div>', unsafe_allow_html=True)
                o3.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("cohen_kappa",0):.3f}</div><div class="mtile-label">Cohen κ</div></div>', unsafe_allow_html=True)
                o4, o5 = st.columns(2)
                o4.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("accuracy_within_1_mark",0)*100:.1f}%</div><div class="mtile-label">Acc ±1 mark</div></div>', unsafe_allow_html=True)
                o5.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("accuracy_within_0_5_mark",0)*100:.1f}%</div><div class="mtile-label">Acc ±0.5 marks</div></div>', unsafe_allow_html=True)
            if not mcq_m and not oe_m:
                st.info("No graded results yet.")
        except Exception as e:
            st.warning(f"Metrics error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Settings":
    st.markdown('<div class="pg-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Configure API keys and backend connection. Add to <code>.env</code> to persist.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">API Keys</div>', unsafe_allow_html=True)
    with st.form("cfg"):
        ak2 = st.text_input("Gemini",    type="password", value=os.getenv("GEMINI_API_KEY",""),    placeholder="AIza…")
        ak1 = st.text_input("Claude",    type="password", value=os.getenv("ANTHROPIC_API_KEY",""), placeholder="sk-ant-…")
        ak3 = st.text_input("Groq",      type="password", value=os.getenv("GROQ_API_KEY",""),      placeholder="gsk_…")
        st.markdown("**Ollama (offline)**")
        ol_url = st.text_input("URL",   value=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))
        ol_mod = st.text_input("Model", value=os.getenv("OLLAMA_MODEL","llama3"))
        if st.form_submit_button("💾  Save Keys", use_container_width=True):
            if ak1: os.environ["ANTHROPIC_API_KEY"] = ak1
            if ak2: os.environ["GEMINI_API_KEY"]    = ak2
            if ak3: os.environ["GROQ_API_KEY"]      = ak3
            os.environ["OLLAMA_BASE_URL"] = ol_url
            os.environ["OLLAMA_MODEL"]    = ol_mod
            st.success("✅ Saved for this session.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Provider Status</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(f'<div class="mtile"><div class="mtile-val">{"✅" if os.getenv("GEMINI_API_KEY") else "—"}</div><div class="mtile-label">Gemini</div></div>', unsafe_allow_html=True)
    p2.markdown(f'<div class="mtile"><div class="mtile-val">{"✅" if os.getenv("ANTHROPIC_API_KEY") else "—"}</div><div class="mtile-label">Claude</div></div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="mtile"><div class="mtile-val">{"✅" if os.getenv("GROQ_API_KEY") else "—"}</div><div class="mtile-label">Groq</div></div>', unsafe_allow_html=True)
    try:
        import urllib.request
        urllib.request.urlopen(os.getenv("OLLAMA_BASE_URL","http://localhost:11434")+"/api/tags", timeout=2)
        ols = "✅"
    except Exception:
        ols = "—"
    p4.markdown(f'<div class="mtile"><div class="mtile-val">{ols}</div><div class="mtile-label">Ollama</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Backend</div>', unsafe_allow_html=True)
    new_url = st.text_input("API Base URL", value=API_BASE)
    if st.button("Update URL"):
        os.environ["API_BASE_URL"] = new_url
        st.success("Updated — restart Streamlit to apply.")
    if ok:
        st.success(f"✅ Connected to `{API_BASE}`")
    else:
        st.error(f"❌ Cannot reach `{API_BASE}`")
        st.code("uvicorn backend.api:app --reload")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Timeouts</div>', unsafe_allow_html=True)
    st.caption(f"Upload: {UPLOAD_TIMEOUT}s · OCR: {OCR_TIMEOUT}s · Evaluate: {EVALUATE_TIMEOUT}s · Poll max: {POLL_TIMEOUT}s · Poll interval: {POLL_INTERVAL}s")
    st.info("Timeouts are configured in `dashboard.py`. Increase `OCR_TIMEOUT` or `POLL_TIMEOUT` if you have very large PDFs.")
    st.markdown('</div>', unsafe_allow_html=True)