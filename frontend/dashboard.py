"""
IntelliGrade-H — Teacher Dashboard (v7)
Full drag-and-drop. Professional design. REST API backend.
"""

import streamlit as st
import os, json, requests
import pandas as pd

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

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

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

section[data-testid="stSidebar"] * { color: var(--muted) !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.83rem !important; padding: 0.1rem 0; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 0.1rem !important; }

/* ── Main area ── */
.main > div { background: var(--bg) !important; }
.main .block-container { padding: 2.5rem 2.5rem 4rem !important; max-width: 1100px !important; }

/* ── Logo block ── */
.logo-block { padding: 1.75rem 1.5rem 1.25rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.logo-wordmark { font-size: 1.25rem; font-weight: 700; color: var(--text) !important; letter-spacing: -0.3px; }
.logo-wordmark span { color: var(--accent2) !important; }
.logo-inst { font-size: 0.65rem; font-weight: 400; color: var(--muted) !important; letter-spacing: 1.2px; text-transform: uppercase; margin-top: 2px; }

/* ── Status dot ── */
.status-row { padding: 0 1.5rem; margin-bottom: 0.5rem; }
.status-dot { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.73rem; font-weight: 500; color: var(--muted) !important; }
.dot-on  { width: 7px; height: 7px; border-radius: 50%; background: var(--success); box-shadow: 0 0 6px var(--success); }
.dot-off { width: 7px; height: 7px; border-radius: 50%; background: var(--danger);  box-shadow: 0 0 6px var(--danger);  }

/* ── Page heading ── */
.pg-title { font-size: 1.6rem; font-weight: 700; color: var(--text); letter-spacing: -0.5px; margin-bottom: 0.2rem; }
.pg-sub   { font-size: 0.82rem; color: var(--muted); margin-bottom: 1.75rem; }

/* ── Card ── */
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

/* ── File uploaders ── */
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

/* ── Primary button ── */
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

/* ── Inputs / selects ── */
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

/* ── Tabs ── */
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

/* ── Score hero ── */
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

/* ── Feedback ── */
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

/* ── Pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.6rem 0; }
.pill { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.73rem; font-weight: 500; }
.pill-g { background: #0d2a1a; color: #3fb950; border: 1px solid #238636; }
.pill-r { background: #2a0e0d; color: #f85149; border: 1px solid #6e1a18; }

/* ── Metric tiles ── */
.mtile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1rem; text-align: center; }
.mtile-val   { font-size: 1.75rem; font-weight: 700; color: var(--text); line-height: 1; }
.mtile-label { font-size: 0.67rem; color: var(--muted); margin-top: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Step indicator ── */
.step-msg {
    display: flex; align-items: center; gap: 0.6rem;
    background: #0e2235; border: 1px solid #1f6feb44;
    border-radius: 8px; padding: 0.7rem 1rem;
    font-size: 0.83rem; color: #93c5fd; font-weight: 500; margin: 0.4rem 0;
}

/* ── Code block ── */
.stCodeBlock { border-radius: 8px !important; }

/* ── Expander ── */
.streamlit-expanderHeader { color: var(--muted) !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.streamlit-expanderContent { background: var(--bg) !important; border-color: var(--border) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px !important; border: 1px solid var(--border) !important; }

/* ── Progress ── */
.stProgress > div > div { background: var(--accent2) !important; border-radius: 999px !important; }
.stProgress { background: var(--border) !important; border-radius: 999px !important; }

/* ── Misc ── */
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
        ok = requests.get(f"{API_BASE}/health", timeout=2).status_code == 200
    except Exception:
        ok = False

    dot = 'dot-on' if ok else 'dot-off'
    label = 'API Online' if ok else 'API Offline'
    st.markdown(f'<div class="status-row"><span class="status-dot"><span class="{dot}"></span>{label}</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="padding:0 1.5rem 1rem;">', unsafe_allow_html=True)
    page = st.radio("nav", ["Single Grade", "Batch Grade", "Analytics", "Settings"], label_visibility="collapsed")
    if not ok:
        st.caption(f"`uvicorn backend.api:app --reload`")
    st.markdown('</div>', unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def upload(f, code="TEMP"):
    r = requests.post(f"{API_BASE}/upload",
        files={"file": (f.name, f.getvalue(), f.type or "image/jpeg")},
        data={"student_code": code}, timeout=60)
    r.raise_for_status()
    return r.json()

def ocr(sid):
    r = requests.post(f"{API_BASE}/ocr/{sid}", timeout=300)
    r.raise_for_status()
    return r.json().get("extracted_text", "")

def evaluate(sid, question, teacher_answer, max_marks, question_type,
             correct_option=None, correct_answer=None, mcq_options=None, rubric_criteria=None):
    payload = {"submission_id": sid, "question": question, "question_type": question_type,
               "teacher_answer": teacher_answer, "max_marks": max_marks}
    if correct_option:  payload["correct_option"]  = correct_option
    if correct_answer:  payload["correct_answer"]   = correct_answer
    if mcq_options:     payload["mcq_options"]      = mcq_options
    if rubric_criteria: payload["rubric_criteria"]  = rubric_criteria
    r = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()

def err(e):
    if isinstance(e, requests.exceptions.ReadTimeout):
        st.error("⏱️ Timed out — TrOCR is still loading model weights (normal on first run, ~2 min). Click **Grade** again shortly.")
    elif isinstance(e, requests.exceptions.ConnectionError):
        st.error("❌ Backend offline. Run: `uvicorn backend.api:app --reload`")
    elif isinstance(e, requests.exceptions.HTTPError):
        try: d = e.response.json().get("detail", str(e))
        except: d = str(e)
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

    if qt in ("mcq","true_false"):
        detected = res.get("mcq_detected_answer","?")
        correct  = res.get("mcq_correct_answer","?")
        icon = "✅" if res.get("mcq_correct") else "❌"
        st.markdown(f'<div class="feedback-box">{icon} &nbsp; Detected: <strong>{detected}</strong> &nbsp;·&nbsp; Correct: <strong>{correct}</strong></div>', unsafe_allow_html=True)
    else:
        m1,m2,m3 = st.columns(3)
        m1.markdown(f'<div class="mtile"><div class="mtile-val">{res.get("llm_score",0):.1f}</div><div class="mtile-label">LLM Score</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="mtile"><div class="mtile-val">{res.get("similarity_score",0):.2f}</div><div class="mtile-label">Similarity</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="mtile"><div class="mtile-val">{res.get("ocr_confidence",0):.0%}</div><div class="mtile-label">OCR Conf.</div></div>', unsafe_allow_html=True)

    if fb:
        st.markdown(f'<div class="feedback-box">{fb}</div>', unsafe_allow_html=True)

    s = res.get("strengths",[])
    m = res.get("missing_concepts",[])
    if s: st.markdown('<div class="pill-row">'+''.join(f'<span class="pill pill-g">✓ {x}</span>' for x in s)+'</div>', unsafe_allow_html=True)
    if m: st.markdown('<div class="pill-row">'+''.join(f'<span class="pill pill-r">✗ {x}</span>' for x in m)+'</div>', unsafe_allow_html=True)

    st.caption(f"Confidence: {res.get('confidence',0):.0%}  ·  Time: {res.get('evaluation_time_sec',0):.1f}s")

    with st.expander("Extracted OCR text"):
        st.code(res.get("ocr_text","(empty)"), language=None)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE GRADE
# ─────────────────────────────────────────────────────────────────────────────

if page == "Single Grade":
    st.markdown('<div class="pg-title">Single Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload three documents — question paper, answer key, and student sheet — then click Grade.</div>', unsafe_allow_html=True)

    # ── File upload card ──────────────────────────────────────────────────────
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
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Settings card ─────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">Evaluation Settings</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns([1,2,1])
    with s1: max_marks = st.number_input("Max Marks", 1.0, 100.0, 10.0, 1.0)
    with s2:
        qtype = st.selectbox("Question Type", ["auto","open_ended","short_answer","mcq","true_false","fill_blank","numerical","diagram"],
            format_func=lambda x:{"auto":"Auto-Detect","open_ended":"Open-Ended / Essay","short_answer":"Short Answer",
                "mcq":"Multiple Choice","true_false":"True / False","fill_blank":"Fill in the Blank",
                "numerical":"Numerical","diagram":"Diagram"}.get(x,x))
    with s3: stu_code = st.text_input("Student Code", value="STU001")

    copt, cans, mcq_opts, rubric = None, None, None, None
    if qtype == "mcq":
        copt = st.text_input("Correct option letter (A / B / C / D)", max_chars=1).upper().strip() or None
    elif qtype == "true_false":
        cans = st.selectbox("Correct answer", ["True","False"])
    if qtype in ("open_ended","short_answer","diagram"):
        with st.expander("Rubric criteria (optional)"):
            raw = st.text_area("One per line — criterion | marks", placeholder="Definition of the concept | 3\nReal-world example | 3\nAnalysis | 4", height=80)
            if raw.strip():
                rubric = []
                for ln in raw.strip().splitlines():
                    p = ln.split("|")
                    if len(p)==2:
                        try: rubric.append({"criterion":p[0].strip(),"marks":float(p[1].strip())})
                        except: pass
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
                q_text = ""
                if qp:
                    ph.markdown('<div class="step-msg">⏳ Step 1 / 3 &nbsp;— Extracting question paper via OCR…</div>', unsafe_allow_html=True)
                    q_text = ocr(upload(qp)["submission_id"])

                a_text = ""
                if ak:
                    ph.markdown('<div class="step-msg">⏳ Step 2 / 3 &nbsp;— Extracting answer key via OCR…</div>', unsafe_allow_html=True)
                    a_text = ocr(upload(ak)["submission_id"])

                ph.markdown('<div class="step-msg">⏳ Step 3 / 3 &nbsp;— Running AI evaluation on student answer…</div>', unsafe_allow_html=True)
                up = upload(sa, stu_code)
                res = evaluate(up["submission_id"], q_text, a_text, max_marks, qtype, copt, cans, mcq_opts or None, rubric)
                ph.empty()
                st.session_state["s_result"]  = res
                st.session_state["s_student"] = stu_code
                st.success("✅ Evaluation complete")
            except Exception as e:
                ph.empty()
                err(e)

    if st.session_state.get("s_result"):
        st.divider()
        show_result(st.session_state["s_result"], label=st.session_state.get("s_student","Student"))
        st.download_button("⬇️  Download JSON", data=json.dumps(st.session_state["s_result"], indent=2),
                           file_name="result.json", mime="application/json")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH GRADE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Batch Grade":
    st.markdown('<div class="pg-title">Batch Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Grade a whole class at once. Name student files with an ID prefix: <code>S001_ans.jpg</code></div>', unsafe_allow_html=True)

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
    b_sheets = st.file_uploader("Drop all student answer sheets here", type=["jpg","jpeg","png","bmp","tiff","pdf"],
                                accept_multiple_files=True, key="b_sheets", label_visibility="collapsed")
    if b_sheets: st.caption(f"✅ {len(b_sheets)} file(s) ready")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Evaluation Settings</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns([1,2])
    with bc1: b_max = st.number_input("Max Marks", 1.0, 100.0, 10.0, 1.0, key="bmax")
    with bc2:
        b_qt = st.selectbox("Question Type", ["auto","open_ended","short_answer","mcq","true_false","fill_blank","numerical","diagram"],
            format_func=lambda x:{"auto":"Auto-Detect","open_ended":"Open-Ended / Essay","short_answer":"Short Answer",
                "mcq":"Multiple Choice","true_false":"True / False","fill_blank":"Fill in the Blank",
                "numerical":"Numerical","diagram":"Diagram"}.get(x,x), key="bqt")
    b_copt = b_cans = None
    if b_qt == "mcq":   b_copt = st.text_input("Correct option", max_chars=1, key="bco").upper().strip() or None
    if b_qt == "true_false": b_cans = st.selectbox("Correct answer", ["True","False"], key="bca")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡  Grade All Students", type="primary", use_container_width=True, key="b_go"):
        if not ok:   st.error("❌ Backend offline.")
        elif not b_sheets: st.warning("Upload student answer sheets first.")
        else:
            prog = st.progress(0, text="Starting…")
            results = []
            q_text = a_text = ""
            try:
                if b_qp:
                    prog.progress(0, text="OCR: question paper…")
                    q_text = ocr(upload(b_qp)["submission_id"])
                if b_ak:
                    prog.progress(0, text="OCR: answer key…")
                    a_text = ocr(upload(b_ak)["submission_id"])
            except Exception as e:
                prog.empty(); err(e); st.stop()

            for i, f in enumerate(b_sheets):
                sid = f.name.split("_")[0] if "_" in f.name else f.name.rsplit(".",1)[0]
                prog.progress((i+1)/len(b_sheets), text=f"Grading {sid}  ({i+1}/{len(b_sheets)})…")
                try:
                    up = upload(f, sid)
                    r = evaluate(up["submission_id"], q_text, a_text, b_max, b_qt, b_copt, b_cans)
                    results.append({"id":sid,"result":r})
                except Exception as e:
                    emsg = "Timed out — retry" if isinstance(e, requests.exceptions.ReadTimeout) else str(e)
                    results.append({"id":sid,"result":None,"error":emsg})

            prog.empty()
            n_ok = sum(1 for r in results if r["result"])
            st.success(f"✅ Done — {n_ok} / {len(results)} graded successfully")

            rows = []
            for item in results:
                r = item.get("result")
                rows.append({
                    "Student": item["id"],
                    "Score": f"{r.get('final_score',0):.1f}" if r else "ERR",
                    "Max": r.get("max_marks",b_max) if r else b_max,
                    "%": f"{r.get('final_score',0)/r.get('max_marks',b_max)*100:.0f}%" if r else "—",
                    "Type": r.get("question_type","—") if r else "—",
                    "Confidence": f"{r.get('confidence',0):.0%}" if r else "—",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇️  Download CSV", data=df.to_csv(index=False), file_name="batch_results.csv", mime="text/csv")

            st.divider()
            for item in results:
                r = item.get("result")
                lbl = f"{r.get('final_score',0):.1f} / {r.get('max_marks',b_max)}" if r else "ERROR"
                with st.expander(f"📄  {item['id']}  —  {lbl}"):
                    if r: show_result(r, label=item["id"])
                    else: st.error(item.get("error","Unknown error"))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Analytics":
    st.markdown('<div class="pg-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">System statistics and AI scoring accuracy.</div>', unsafe_allow_html=True)

    if not ok:
        st.warning("Backend is offline. Start it to view analytics.")
    else:
        if st.button("🔄 Refresh"): st.rerun()
        try:
            stats = requests.get(f"{API_BASE}/stats", timeout=10).json()
            t1,t2,t3,t4 = st.columns(4)
            t1.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("total_submissions",0)}</div><div class="mtile-label">Submissions</div></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("evaluated",0)}</div><div class="mtile-label">Evaluated</div></div>', unsafe_allow_html=True)
            t3.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("average_score",0):.1f}</div><div class="mtile-label">Avg Score</div></div>', unsafe_allow_html=True)
            t4.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("average_evaluation_time_sec",0):.1f}s</div><div class="mtile-label">Avg Time</div></div>', unsafe_allow_html=True)
        except Exception as e: st.warning(f"Stats error: {e}")

        st.divider()
        st.markdown('<div class="card-label" style="font-size:0.65rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#8b949e;">AI Accuracy Metrics</div>', unsafe_allow_html=True)
        try:
            mx = requests.get(f"{API_BASE}/metrics", timeout=10).json()
            st.caption(f"Updated: {mx.get('last_updated','—')}  ·  Total: {mx.get('total_evaluated',0)}")
            mcq_m = mx.get("mcq"); oe_m = mx.get("open_ended")
            if mcq_m:
                st.markdown("**MCQ**")
                m1,m2,m3 = st.columns(3)
                m1.markdown(f'<div class="mtile"><div class="mtile-val">{mcq_m.get("accuracy_pct",0):.1f}%</div><div class="mtile-label">Accuracy</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="mtile"><div class="mtile-val">{mcq_m.get("n_correct",0)}</div><div class="mtile-label">Correct</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="mtile"><div class="mtile-val">{mcq_m.get("n_wrong",0)}</div><div class="mtile-label">Wrong</div></div>', unsafe_allow_html=True)
            if oe_m:
                st.markdown("**Open-Ended**")
                o1,o2,o3 = st.columns(3)
                o1.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("mae",0):.3f}</div><div class="mtile-label">MAE</div></div>', unsafe_allow_html=True)
                o2.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("pearson_r",0):.3f}</div><div class="mtile-label">Pearson r</div></div>', unsafe_allow_html=True)
                o3.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("cohen_kappa",0):.3f}</div><div class="mtile-label">Cohen κ</div></div>', unsafe_allow_html=True)
                o4,o5 = st.columns(2)
                o4.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("accuracy_within_1_mark",0)*100:.1f}%</div><div class="mtile-label">Acc ±1 mark</div></div>', unsafe_allow_html=True)
                o5.markdown(f'<div class="mtile"><div class="mtile-val">{oe_m.get("accuracy_within_0_5_mark",0)*100:.1f}%</div><div class="mtile-label">Acc ±0.5 marks</div></div>', unsafe_allow_html=True)
            if not mcq_m and not oe_m: st.info("No graded results yet.")
        except Exception as e: st.warning(f"Metrics error: {e}")


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
    p1,p2,p3,p4 = st.columns(4)
    p1.markdown(f'<div class="mtile"><div class="mtile-val">{"✅" if os.getenv("GEMINI_API_KEY") else "—"}</div><div class="mtile-label">Gemini</div></div>', unsafe_allow_html=True)
    p2.markdown(f'<div class="mtile"><div class="mtile-val">{"✅" if os.getenv("ANTHROPIC_API_KEY") else "—"}</div><div class="mtile-label">Claude</div></div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="mtile"><div class="mtile-val">{"✅" if os.getenv("GROQ_API_KEY") else "—"}</div><div class="mtile-label">Groq</div></div>', unsafe_allow_html=True)
    try:
        import urllib.request
        urllib.request.urlopen(os.getenv("OLLAMA_BASE_URL","http://localhost:11434")+"/api/tags", timeout=2)
        ols = "✅"
    except: ols = "—"
    p4.markdown(f'<div class="mtile"><div class="mtile-val">{ols}</div><div class="mtile-label">Ollama</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Backend</div>', unsafe_allow_html=True)
    new_url = st.text_input("API Base URL", value=API_BASE)
    if st.button("Update URL"):
        os.environ["API_BASE_URL"] = new_url
        st.success("Updated — restart Streamlit to apply.")
    if ok: st.success(f"✅ Connected to `{API_BASE}`")
    else:
        st.error(f"❌ Cannot reach `{API_BASE}`")
        st.code("uvicorn backend.api:app --reload")
    st.markdown('</div>', unsafe_allow_html=True)