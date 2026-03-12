"""
IntelliGrade-H — Teacher Dashboard (v5.0)
==========================================
Pages:
  📄  Paper Manager    — upload & view question papers
  🔑  Answer Keys      — upload answer key PDFs, manual edits
  📚  Student Booklets — upload individual booklets, evaluate
  📦  Bulk Upload      — upload multiple booklets for a subject at once
  📊  Analytics        — system stats, AI accuracy, paper insights
  ⚙️   Settings         — API keys, backend config
"""

import streamlit as st
import streamlit.components.v1 as st_components
import os, json, time, re, requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

UPLOAD_TIMEOUT   = 60
OCR_TIMEOUT      = 300
EVALUATE_TIMEOUT = 300
POLL_TIMEOUT     = 600
POLL_INTERVAL    = 4

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
    --surface2:  #1c2128;
    --border:    #21262d;
    --accent:    #1f6feb;
    --accent2:   #388bfd;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --success:   #3fb950;
    --danger:    #f85149;
    --warn:      #d29922;
    --purple:    #bc8cff;
    --radius:    10px;
}
html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
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
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem 1.5rem; margin-bottom: 1.1rem; }
.card-label { font-size: 0.62rem; font-weight: 600; letter-spacing: 1.6px; text-transform: uppercase; color: var(--muted); margin-bottom: 1rem; }
.info-banner  { display:flex;align-items:center;gap:.6rem;background:#0e1e35;border:1px solid #1f6feb33;border-radius:8px;padding:.65rem 1rem;font-size:.8rem;color:#7cb9f7;font-weight:400;margin-bottom:1.1rem; }
.success-banner { display:flex;align-items:center;gap:.6rem;background:#0a1f10;border:1px solid #2ea04333;border-radius:8px;padding:.65rem 1rem;font-size:.8rem;color:#3fb950;font-weight:400;margin-bottom:1.1rem; }
.warn-banner  { display:flex;align-items:center;gap:.6rem;background:#1a1600;border:1px solid #d2992233;border-radius:8px;padding:.65rem 1rem;font-size:.8rem;color:#d29922;font-weight:400;margin-bottom:1.1rem; }
[data-testid="stFileUploader"] { background:#0d1117 !important;border:1.5px dashed #30363d !important;border-radius:8px !important; }
[data-testid="stFileUploader"]:hover { border-color:var(--accent2) !important; }
[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] p { color:var(--muted) !important;font-size:.8rem !important; }
[data-testid="stFileUploader"] button { display:none !important; }
.stButton > button { background:var(--accent) !important;color:#fff !important;border:none !important;border-radius:8px !important;font-family:'Sora',sans-serif !important;font-weight:600 !important;font-size:.88rem !important;height:2.7rem !important;letter-spacing:.2px !important; }
.stButton > button:hover { background:var(--accent2) !important; }
.stButton > button[kind="secondary"] { background:var(--surface2) !important;color:var(--muted) !important;border:1px solid var(--border) !important; }
.stTextInput input,.stNumberInput input,.stTextArea textarea,.stSelectbox > div > div { background:var(--bg) !important;border:1px solid var(--border) !important;border-radius:8px !important;color:var(--text) !important;font-family:'Sora',sans-serif !important;font-size:.85rem !important; }
label { color:var(--muted) !important;font-size:.8rem !important;font-weight:500 !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important;border-bottom:1px solid var(--border) !important;gap:0 !important;margin-bottom:1.5rem; }
.stTabs [data-baseweb="tab"] { background:transparent !important;border:none !important;border-bottom:2px solid transparent !important;color:var(--muted) !important;font-weight:500 !important;font-size:.85rem !important;padding:.6rem 1.2rem !important;border-radius:0 !important; }
.stTabs [aria-selected="true"] { color:var(--text) !important;border-bottom-color:var(--accent2) !important; }
.score-hero { background:linear-gradient(135deg,#0d1117,#0e2a4d 60%,#1a4a8a);border:1px solid #1f6feb44;border-radius:14px;padding:2rem;text-align:center;margin:1.25rem 0; }
.score-num { font-size:4rem;font-weight:700;color:#fff;line-height:1;letter-spacing:-2px; }
.score-denom { font-size:1.6rem;opacity:.4;font-weight:300; }
.score-pct  { font-size:.9rem;color:#93c5fd;margin-top:.5rem;letter-spacing:.5px; }
.score-type { font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#475569;margin-top:.35rem;letter-spacing:2px;text-transform:uppercase; }
.feedback-box { background:#161b22;border:1px solid var(--border);border-left:3px solid var(--accent2);border-radius:0 8px 8px 0;padding:.9rem 1.1rem;font-size:.86rem;line-height:1.7;color:#c9d1d9;margin:.9rem 0; }
.pill-row { display:flex;flex-wrap:wrap;gap:.35rem;margin:.6rem 0; }
.pill { display:inline-flex;align-items:center;gap:.25rem;padding:.2rem .65rem;border-radius:999px;font-size:.73rem;font-weight:500; }
.pill-g { background:#0d2a1a;color:#3fb950;border:1px solid #238636; }
.pill-r { background:#2a0e0d;color:#f85149;border:1px solid #6e1a18; }
.pill-b { background:#0e1f35;color:#388bfd;border:1px solid #1f6feb44; }
.pill-p { background:#1a0e2a;color:#bc8cff;border:1px solid #6e3fa0; }
.mtile { background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.9rem 1rem;text-align:center; }
.mtile-val   { font-size:1.75rem;font-weight:700;color:var(--text);line-height:1; }
.mtile-label { font-size:.67rem;color:var(--muted);margin-top:.3rem;font-weight:500;text-transform:uppercase;letter-spacing:.8px; }
.q-table { width:100%;border-collapse:collapse;font-size:.82rem; }
.q-table th { color:var(--muted);font-weight:600;font-size:.68rem;letter-spacing:.8px;text-transform:uppercase;border-bottom:1px solid var(--border);padding:.4rem .6rem;text-align:left; }
.q-table td { padding:.45rem .6rem;border-bottom:1px solid #21262d55;color:var(--text);vertical-align:top; }
.q-table tr:hover td { background:#1c2128; }
.q-badge { display:inline-block;padding:.1rem .45rem;border-radius:4px;font-size:.68rem;font-weight:600;font-family:'JetBrains Mono',monospace; }
.badge-a { background:#0e2235;color:#388bfd; }
.badge-b { background:#1a0e2a;color:#bc8cff; }
.badge-c { background:#0a1f10;color:#3fb950; }
.badge-marks { background:#1a1600;color:#d29922; }
.paper-header { background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem; }
.paper-title { font-size:1rem;font-weight:600;color:var(--text); }
.paper-meta  { font-size:.75rem;color:var(--muted);margin-top:.2rem; }
.bulk-row { background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem;margin-bottom:.5rem;display:flex;align-items:center;gap:.75rem; }
hr { border-color:var(--border) !important; }
#MainMenu, footer, header { visibility:hidden !important; }
[data-testid="stDecoration"] { display:none !important; }
.stAlert { border-radius:8px !important;font-size:.83rem !important; }
[data-testid="stDataFrame"] { border-radius:8px !important;border:1px solid var(--border) !important; }
.stProgress > div > div { background:var(--accent2) !important;border-radius:999px !important; }
.stProgress { background:var(--border) !important;border-radius:999px !important; }
.streamlit-expanderHeader { color:var(--muted) !important;font-size:.82rem !important;font-weight:500 !important; }
.streamlit-expanderContent { background:var(--bg) !important;border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
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

    dot   = 'dot-on'  if ok else 'dot-off'
    label = 'API Online' if ok else 'API Offline'
    st.markdown(f'<div class="status-row"><span class="status-dot"><span class="{dot}"></span>{label}</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="padding:0 1.5rem 1rem;">', unsafe_allow_html=True)
    page = st.radio("nav", [
        "📄  Paper Manager",
        "🔑  Answer Keys",
        "📚  Student Booklets",
        "📦  Bulk Upload",
        "📊  Analytics",
        "⚙️  Settings",
    ], label_visibility="collapsed")
    if not ok:
        st.caption("`uvicorn backend.api:app --reload`")
    st.markdown('</div>', unsafe_allow_html=True)


# ── API helpers ────────────────────────────────────────────────────────────────

def get_papers():
    try:
        return requests.get(f"{API_BASE}/papers", timeout=10).json()
    except Exception:
        return []


def get_paper(paper_id):
    try:
        return requests.get(f"{API_BASE}/paper/{paper_id}", timeout=10).json()
    except Exception:
        return None


def get_booklets():
    try:
        return requests.get(f"{API_BASE}/booklets", timeout=10).json()
    except Exception:
        return []


def show_error(e):
    msg = str(e)
    try:
        detail = e.response.json().get("detail", msg) if hasattr(e, "response") and e.response else msg
    except Exception:
        detail = msg
    st.error(f"❌ {detail}")


# ── Correct total marks calculation ──────────────────────────────────────────

def compute_correct_total_marks(questions: list, parts: list) -> float:
    """
    Compute the correct total marks for an exam paper.
    Rule: For OR pairs (e.g. Q8 or Q9), only count ONE question's marks, not both.
    Total = sum of all non-OR questions + sum of one from each OR pair.
    """
    if not questions:
        return 0.0

    # Group by (part, or_pair_index) to identify OR pairs
    # OR pairs: consecutive questions where 2nd has is_or_option=True
    total = 0.0
    i = 0
    while i < len(questions):
        q = questions[i]
        marks = float(q.get("marks", 0))
        is_or = q.get("is_or_option", False)

        if not is_or:
            # Check if NEXT question is an OR alternative of this one
            if i + 1 < len(questions) and questions[i + 1].get("is_or_option", False):
                # This is an OR pair — count only this question's marks (the primary)
                total += marks
                i += 2  # skip the OR alternative
            else:
                total += marks
                i += 1
        else:
            # This is an OR alternative — already counted its pair, skip
            i += 1

    return total


# ── Paper structure renderer ───────────────────────────────────────────────────

def show_paper_structure(paper, show_answer_status=False):
    questions = paper.get("questions", [])
    parts     = paper.get("parts", [])

    # Recompute correct total marks
    correct_total = compute_correct_total_marks(questions, parts)

    meta = f"{paper.get('course_code','—')}  ·  {paper.get('exam_name','—')}"
    if paper.get("set_name"):
        meta += f"  ·  {paper['set_name']}"
    meta += f"  ·  Total: {correct_total:.0f} marks"

    st.markdown(f"""
    <div class="paper-header">
        <div class="paper-title">📋 {paper.get('course_name') or paper.get('course_code','Unknown Paper')}</div>
        <div class="paper-meta">{meta}</div>
    </div>
    """, unsafe_allow_html=True)

    # Parts summary — show correct question counts and marks
    if parts:
        # Count actual questions per part (excluding OR duplicates from count)
        part_q_counts = {}
        for q in questions:
            pn = q.get("part_name", "")
            if pn not in part_q_counts:
                part_q_counts[pn] = {"total": 0, "required": 0}
            part_q_counts[pn]["total"] += 1

        cols = st.columns(len(parts))
        badge_classes = ["badge-a", "badge-b", "badge-c"]
        for i, part in enumerate(parts):
            with cols[i]:
                pname   = part.get("part_name", "")
                mpq     = part.get("marks_per_question", 0)
                instr   = part.get("instructions", "Answer ALL questions")
                # Count primary (non-OR) questions in this part
                primary_count = sum(
                    1 for q in questions
                    if q.get("part_name") == pname and not q.get("is_or_option", False)
                )
                bc = badge_classes[i % len(badge_classes)]
                st.markdown(
                    f'<div class="mtile">'
                    f'<div class="mtile-val">{primary_count}</div>'
                    f'<div class="mtile-label">{pname} · {mpq}m each</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(instr or "")

    # Question table
    if questions:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        rows_html = ""
        badge_map = {}
        bc_list = ["badge-a", "badge-b", "badge-c"]
        for i, part in enumerate(parts):
            badge_map[part.get("part_name", "")] = bc_list[i % len(bc_list)]

        for q in questions:
            part_name = q.get("part_name", "")
            bc        = badge_map.get(part_name, "badge-a")
            q_type    = (q.get("question_type") or "—").replace("_", " ")
            or_flag   = ' <span style="color:#d29922;font-size:0.7rem;font-weight:600;">OR</span>' if q.get("is_or_option") else ""
            q_text    = q.get("question_text") or "—"
            q_preview = q_text[:140] + ("…" if len(q_text) > 140 else "")

            answered_col = ""
            if show_answer_status:
                answered = "✅" if q.get("teacher_answer") or q.get("has_answer_key") else "⚠️"
                answered_col = f'<td style="text-align:center;">{answered}</td>'

            rows_html += f"""
            <tr>
                <td><span class="q-badge {bc}">Q{q['question_number']}</span>{or_flag}</td>
                <td style="max-width:500px;word-break:break-word;white-space:normal;" title="{q_text}">{q_preview}</td>
                <td><span class="q-badge badge-marks">{q['marks']}m</span></td>
                <td style="color:var(--muted);font-size:0.78rem;">{q_type}</td>
                <td style="color:var(--muted);font-size:0.75rem;">{part_name}</td>
                {answered_col}
            </tr>"""

        ans_header = "<th>Key</th>" if show_answer_status else ""
        # Use components.html so Streamlit doesn't strip <table> tags (Streamlit 1.31+)
        table_html = f"""
        <style>
        :root {{
            --bg:#0d1117; --surface:#161b22; --surface2:#1c2128;
            --border:#21262d; --text:#e6edf3; --muted:#8b949e;
            --accent2:#388bfd; --warn:#d29922; --success:#3fb950;
        }}
        body {{ margin:0; padding:0; background:var(--bg); font-family:'Sora',sans-serif; }}
        table {{ width:100%;border-collapse:collapse;font-size:.82rem;background:var(--bg); }}
        th {{ color:var(--muted);font-weight:600;font-size:.68rem;letter-spacing:.8px;
              text-transform:uppercase;border-bottom:1px solid var(--border);
              padding:.45rem .6rem;text-align:left; }}
        td {{ padding:.45rem .6rem;border-bottom:1px solid #21262d55;color:var(--text);
              vertical-align:top;word-break:break-word; }}
        tr:hover td {{ background:#1c2128; }}
        .q-badge {{ display:inline-block;padding:.1rem .45rem;border-radius:4px;
                    font-size:.68rem;font-weight:600;font-family:monospace; }}
        .badge-a {{ background:#0e2235;color:#388bfd; }}
        .badge-b {{ background:#1a0e2a;color:#bc8cff; }}
        .badge-c {{ background:#0a1f10;color:#3fb950; }}
        .badge-marks {{ background:#1a1600;color:#d29922; }}
        </style>
        <div style="overflow-x:auto;border-radius:8px;border:1px solid #21262d;padding:0;">
        <table>
            <thead><tr><th>No.</th><th>Question</th><th>Marks</th><th>Type</th><th>Part</th>{ans_header}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>"""
        row_count = len(questions)
        st_components.html(table_html, height=max(250, row_count * 58 + 80), scrolling=True)
    else:
        st.warning("⚠️ No questions were extracted from this paper. Check the PDF format.")


def show_answer_key_structure(res):
    """Render parsed answer key with expandable model answers."""
    questions = res.get("questions", [])
    parts     = res.get("parts", [])
    linked    = res.get("answers_linked", res.get("questions_updated", 0))
    total_q   = len(questions)

    correct_total = compute_correct_total_marks(questions, parts)
    meta = f"{res.get('course_code','—')}  ·  {res.get('exam_name','—')}"
    if res.get('set_name'):
        meta += f"  ·  {res['set_name']}"

    st.markdown(f"""
    <div class="paper-header">
        <div class="paper-title">🔑 {res.get('course_name') or res.get('course_code','Answer Key')}</div>
        <div class="paper-meta">{meta} &nbsp;·&nbsp;
        <span style="color:var(--success);">{linked}/{total_q} answers linked</span></div>
    </div>
    """, unsafe_allow_html=True)

    if not questions:
        st.info("No questions found. Upload a question paper first.")
        return

    badge_map = {}
    bc_list   = ["badge-a", "badge-b", "badge-c"]
    for i, part in enumerate(parts):
        badge_map[part.get("part_name", "")] = bc_list[i % len(bc_list)]

    # Group by part for organised display
    part_names = list(dict.fromkeys(q.get("part_name", "") for q in questions))

    for pname in part_names:
        part_qs = [q for q in questions if q.get("part_name") == pname]
        if pname:
            st.markdown(f'<div class="card-label" style="margin-top:1rem;">{pname}</div>', unsafe_allow_html=True)

        for q in part_qs:
            has_ans  = bool(q.get("has_answer") or q.get("teacher_answer"))
            ans_text = q.get("teacher_answer") or q.get("has_answer") or ""
            icon     = "✅" if has_ans else "⚠️"
            or_sfx   = " (OR)" if q.get("is_or_option") else ""
            preview  = (q.get("question_text") or "")[:120]

            with st.expander(f"{icon}  Q{q['question_number']}{or_sfx}  ·  {q.get('marks',0)}m  —  {preview}"):
                c1, c2 = st.columns([1, 3])
                with c1:
                    bc = badge_map.get(pname, "badge-a")
                    st.markdown(f'<div class="mtile"><div class="mtile-val">{q.get("marks",0)}</div><div class="mtile-label">Marks</div></div>', unsafe_allow_html=True)
                    qt = (q.get("question_type") or "—").replace("_", " ")
                    st.markdown(f'<div class="mtile" style="margin-top:0.5rem;"><div class="mtile-val" style="font-size:0.85rem;">{qt}</div><div class="mtile-label">Type</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="card-label">Question</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:0.83rem;color:var(--text);margin-bottom:0.8rem;">{q.get("question_text") or "—"}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card-label">Model Answer</div>', unsafe_allow_html=True)
                    if has_ans and ans_text:
                        st.markdown(f'<div class="feedback-box" style="white-space:pre-wrap;">{ans_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warn-banner">⚠️ &nbsp; No model answer extracted yet.</div>', unsafe_allow_html=True)


def show_booklet_structure(res):
    """Render parsed student booklet with per-question answers."""
    roll         = res.get("roll_number") or res.get("register_number") or "Unknown"
    total_pages  = res.get("total_pages", "?")
    answers_found = res.get("answers_found", len(res.get("questions", [])))

    st.markdown(f"""
    <div class="paper-header">
        <div class="paper-title">📖 Student Booklet — {roll}</div>
        <div class="paper-meta">{res.get('course_code') or 'Booklet'} &nbsp;·&nbsp; {total_pages} pages &nbsp;·&nbsp;
        <span style="color:var(--success);">{answers_found} answers extracted</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Metadata tiles
    meta_tiles = []
    if res.get("roll_number"):     meta_tiles.append(("Roll No.",  res["roll_number"]))
    if res.get("register_number"): meta_tiles.append(("Reg. No.",  res["register_number"]))
    if res.get("set_name"):        meta_tiles.append(("Set",       res["set_name"]))
    if res.get("course_code"):     meta_tiles.append(("Course",    res["course_code"]))

    if meta_tiles:
        cols = st.columns(min(len(meta_tiles), 4))
        for i, (lbl, val) in enumerate(meta_tiles):
            with cols[i]:
                st.markdown(f'<div class="mtile"><div class="mtile-val" style="font-size:1.1rem;">{val}</div><div class="mtile-label">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("&nbsp;", unsafe_allow_html=True)

    questions = res.get("questions", [])
    if not questions:
        st.warning("⚠️ No answers found in booklet. OCR may need improvement or booklet is blank.")
        if res.get("raw_ocr_preview"):
            with st.expander("Raw OCR preview (debug)"):
                st.code(res["raw_ocr_preview"], language=None)
        return

    linked_count = sum(1 for q in questions if q.get("linked_to_eq"))
    banner_class = "success-banner" if linked_count == len(questions) else "info-banner"
    st.markdown(
        f'<div class="{banner_class}">🔗 &nbsp; {linked_count}/{len(questions)} answers linked to exam paper questions.</div>',
        unsafe_allow_html=True
    )

    for q in questions:
        q_num     = q.get("question_number", "?")
        part      = q.get("part_name", "")
        is_or     = q.get("is_or_option", False)
        linked    = q.get("linked_to_eq", False)
        ans_text  = q.get("answer_text") or q.get("preview") or ""
        preview   = ans_text[:100] if ans_text else "(empty)"

        link_icon = "🔗" if linked else "⚠️"
        or_sfx    = " (OR)" if is_or else ""
        part_tag  = f"  ·  {part}" if part else ""

        with st.expander(f"{link_icon}  Q{q_num}{or_sfx}{part_tag}  —  {preview}"):
            st.markdown('<div class="card-label">Extracted Student Answer</div>', unsafe_allow_html=True)
            full = q.get("answer_text") or q.get("preview") or "*(empty)*"
            st.markdown(f'<div class="feedback-box" style="white-space:pre-wrap;">{full}</div>', unsafe_allow_html=True)
            if not linked:
                st.markdown('<div class="warn-banner" style="margin-top:0.5rem;">⚠️ &nbsp; Could not link to exam paper question — check question numbering.</div>', unsafe_allow_html=True)


def show_evaluation_results(eval_res):
    """Render booklet evaluation results as a structured scorecard."""
    questions  = eval_res.get("questions", [])
    total_obt  = eval_res.get("total_obtained", 0)
    total_poss = eval_res.get("total_possible", 0)
    pct        = eval_res.get("percentage", 0)

    # Hero score card
    grade_color = "#3fb950" if pct >= 60 else ("#d29922" if pct >= 40 else "#f85149")
    st.markdown(f"""
    <div class="score-hero">
        <div class="score-num">{total_obt:.1f}<span class="score-denom"> /{total_poss:.0f}</span></div>
        <div class="score-pct" style="color:{grade_color};">{pct:.1f}% &nbsp;·&nbsp; {eval_res.get('roll_number') or 'Student'}</div>
        <div class="score-type">{eval_res.get('paper_id','—')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Question breakdown table
    if questions:
        rows = []
        for q in questions:
            status = q.get("status", "")
            if status == "evaluated":
                score   = q.get("score", 0)
                max_m   = q.get("max_marks", 0)
                q_pct   = f"{100*score/max_m:.0f}%" if max_m else "—"
                fb      = (q.get("feedback") or "")[:120]
                rows.append({
                    "Q#":    f"Q{q.get('question_number','?')}",
                    "Part":  q.get("part_name", "—"),
                    "Score": f"{score:.1f} / {max_m}",
                    "%":     q_pct,
                    "Feedback": fb,
                })
            else:
                rows.append({
                    "Q#":    f"Q{q.get('question_number','?')}",
                    "Part":  q.get("part_name", "—"),
                    "Score": "—",
                    "%":     "—",
                    "Feedback": q.get("reason", status),
                })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Per-question expandable feedback
        st.markdown("&nbsp;", unsafe_allow_html=True)
        for q in questions:
            if q.get("status") != "evaluated":
                continue
            score  = q.get("score", 0)
            max_m  = q.get("max_marks", 0)
            q_pct  = 100 * score / max_m if max_m else 0
            with st.expander(f"Q{q.get('question_number')}  ·  {q.get('part_name','')}  —  {score:.1f}/{max_m}  ({q_pct:.0f}%)"):
                fb = q.get("feedback", "")
                if fb:
                    st.markdown(f'<div class="feedback-box">{fb}</div>', unsafe_allow_html=True)
                strengths = q.get("strengths") or []
                missing   = q.get("missing_concepts") or []
                if strengths:
                    st.markdown("**✅ Strengths:**")
                    for s in strengths:
                        st.markdown(f"- {s}")
                if missing:
                    st.markdown("**⚠️ Missing concepts:**")
                    for m in missing:
                        st.markdown(f"- {m}")
                comp = q.get("component_scores", {})
                if comp:
                    cols = st.columns(5)
                    labels = [("LLM", "llm"), ("Similarity", "similarity"),
                              ("Rubric", "rubric"), ("Keyword", "keyword"), ("Length", "length")]
                    for col, (lbl, key) in zip(cols, labels):
                        col.metric(lbl, f"{comp.get(key, 0):.2f}")

    # Download CSV
    if questions:
        df_dl = pd.DataFrame([{
            "Q#":        f"Q{q.get('question_number','?')}",
            "Part":      q.get("part_name", ""),
            "Score":     q.get("score", ""),
            "Max Marks": q.get("max_marks", ""),
            "Feedback":  q.get("feedback", ""),
        } for q in questions if q.get("status") == "evaluated"])
        if not df_dl.empty:
            st.download_button(
                "⬇️  Download Results CSV",
                data=df_dl.to_csv(index=False),
                file_name=f"results_{eval_res.get('roll_number','student')}.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Paper Manager
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📄  Paper Manager":
    st.markdown('<div class="pg-title">Paper Manager</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload a question paper PDF to auto-extract its structure, parts, marks, and question types.</div>', unsafe_allow_html=True)

    tab_upload, tab_library = st.tabs(["Upload New Paper", "Paper Library"])

    with tab_upload:
        st.markdown('<div class="info-banner">🤖 &nbsp; The AI reads your PDF and extracts every question, marks per question, part structure — automatically. Part-A (2m each) and Part-B (OR questions, 12m each) are handled correctly.</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Question Paper PDF</div>', unsafe_allow_html=True)
        paper_pdf = st.file_uploader("Upload question paper", type=["pdf"], key="pm_pdf", label_visibility="collapsed")
        set_name  = st.text_input("Set Name (optional)", placeholder="e.g. Set-A", key="pm_setname")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔍  Parse Paper", type="primary", use_container_width=True, key="pm_parse"):
            if not ok:
                st.error("❌ Backend offline.")
            elif not paper_pdf:
                st.warning("Please upload a question paper PDF.")
            else:
                with st.spinner("Uploading and parsing question paper — this may take 30–60s…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/paper/upload",
                            files={"file": (paper_pdf.name, paper_pdf.getvalue(), "application/pdf")},
                            data={"set_name": set_name.strip()},
                            timeout=120,
                        )
                        r.raise_for_status()
                        paper = r.json()
                        st.session_state["pm_last_paper"] = paper

                        n_q     = len(paper.get("questions", []))
                        correct = compute_correct_total_marks(paper.get("questions", []), paper.get("parts", []))
                        emoji   = "✅" if n_q > 0 else "⚠️"

                        if n_q == 0:
                            st.warning(
                                f"{emoji} Paper uploaded as **{paper.get('paper_id')}** but **0 questions extracted**. "
                                "This usually means the PDF has complex formatting or is scanned. "
                                "Try re-uploading a cleaner PDF, or manually add answers in the Answer Keys tab."
                            )
                        else:
                            st.success(
                                f"{emoji} Paper parsed: **{paper.get('paper_id')}** — "
                                f"**{n_q} questions** extracted | Total marks: **{correct:.0f}**"
                            )
                    except Exception as e:
                        show_error(e)

        if "pm_last_paper" in st.session_state:
            st.divider()
            show_paper_structure(st.session_state["pm_last_paper"])

    with tab_library:
        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("🔄 Refresh", key="lib_refresh"):
                st.rerun()

        papers = get_papers()
        if not papers:
            st.markdown('<div class="warn-banner">📭 &nbsp; No papers uploaded yet.</div>', unsafe_allow_html=True)
        else:
            st.caption(f"{len(papers)} paper(s) in database")
            for p in papers:
                num_q    = p.get("num_questions", 0)
                answered = 0
                full     = None

                with st.expander(f"📋 {p.get('paper_id')}  —  {p.get('course_name') or p.get('course_code','?')}  ·  {num_q} questions"):
                    full = get_paper(p["paper_id"])
                    if full:
                        answered = sum(1 for q in full.get("questions", []) if q.get("has_answer_key"))
                        total_q  = len(full.get("questions", []))
                        correct  = compute_correct_total_marks(full.get("questions", []), full.get("parts", []))
                        st.caption(f"Total marks: {correct:.0f}  ·  {answered}/{total_q} answers loaded")
                        show_paper_structure(full, show_answer_status=True)
                    else:
                        st.warning("Could not load paper details.")

                    col_del, _ = st.columns([1, 3])
                    if col_del.button("🗑 Delete Paper", key=f"del_{p['paper_id']}"):
                        try:
                            rd = requests.delete(f"{API_BASE}/paper/{p['paper_id']}", timeout=10)
                            if rd.status_code in (200, 204, 404):
                                st.success("Deleted.")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {rd.text}")
                        except Exception as e:
                            show_error(e)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Answer Keys
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔑  Answer Keys":
    st.markdown('<div class="pg-title">Answer Keys</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload an answer key PDF to link model answers to each question. Supports Part-A and Part-B.</div>', unsafe_allow_html=True)

    papers = get_papers()
    if not papers:
        st.markdown('<div class="warn-banner">⚠️ &nbsp; No papers found. Upload a question paper first from Paper Manager.</div>', unsafe_allow_html=True)
        st.stop()

    paper_options = {f"{p['paper_id']}  ({p.get('course_name','?')})": p["paper_id"] for p in papers}

    st.markdown('<div class="card"><div class="card-label">Select Exam Paper</div>', unsafe_allow_html=True)
    selected_label    = st.selectbox("Select paper", list(paper_options.keys()), key="ak_paper_sel", label_visibility="collapsed")
    selected_paper_id = paper_options[selected_label]
    st.markdown('</div>', unsafe_allow_html=True)

    # Always show current paper structure with answer status
    full_paper = get_paper(selected_paper_id)
    if full_paper:
        questions = full_paper.get("questions", [])
        answered  = sum(1 for q in questions if q.get("has_answer_key") or q.get("teacher_answer"))
        total_q   = len(questions)
        correct   = compute_correct_total_marks(questions, full_paper.get("parts", []))

        if answered == total_q and total_q > 0:
            st.markdown(f'<div class="success-banner">✅ &nbsp; Answer key complete — {answered}/{total_q} questions have model answers. Total marks: {correct:.0f}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-banner">📝 &nbsp; {answered}/{total_q} questions have model answers. Upload an answer key PDF below.</div>', unsafe_allow_html=True)

    tab_upload, tab_manual = st.tabs(["Upload Answer Key PDF", "Manual Entry"])

    with tab_upload:
        st.markdown('<div class="info-banner">🤖 &nbsp; The AI extracts model answers from your PDF and links them to questions by number. Works with Part-A (short) and Part-B (long) answers.</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Answer Key PDF</div>', unsafe_allow_html=True)
        ak_pdf = st.file_uploader("Upload answer key", type=["pdf"], key="ak_pdf_upload", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("📤  Upload & Extract Answers", type="primary", use_container_width=True, key="ak_upload_btn"):
            if not ok:
                st.error("❌ Backend offline.")
            elif not ak_pdf:
                st.warning("Please upload the answer key PDF.")
            else:
                with st.spinner("Extracting model answers from PDF — this may take 30–60s…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/answer-key/upload",
                            files={"file": (ak_pdf.name, ak_pdf.getvalue(), "application/pdf")},
                            data={"paper_id": selected_paper_id},
                            timeout=180,
                        )
                        r.raise_for_status()
                        res = r.json()
                        linked = res.get("answers_linked", res.get("questions_updated", 0))
                        total_extracted = res.get("total_answers_extracted", linked)

                        if linked == 0:
                            st.warning(
                                f"⚠️ 0 model answers linked to **{selected_paper_id}**. "
                                f"The parser extracted {total_extracted} answers but none matched question numbers. "
                                "Try the Manual Entry tab to add answers directly."
                            )
                        else:
                            st.success(f"✅ {linked} model answers linked to **{selected_paper_id}**")

                        st.session_state["ak_last_result"] = res
                        # Reload full paper to get updated answers
                        st.session_state["ak_refreshed_paper"] = get_paper(selected_paper_id)

                    except Exception as e:
                        show_error(e)

        # Show answer key structure after upload
        if st.session_state.get("ak_refreshed_paper"):
            rp = st.session_state["ak_refreshed_paper"]
            if rp.get("paper_id") == selected_paper_id or True:
                st.divider()
                # Build show_answer_key_structure compatible dict
                ak_display = dict(rp)
                ak_display["answers_linked"] = sum(
                    1 for q in rp.get("questions", []) if q.get("has_answer_key") or q.get("teacher_answer")
                )
                show_answer_key_structure(ak_display)
        elif full_paper and any(q.get("has_answer_key") or q.get("teacher_answer") for q in full_paper.get("questions", [])):
            st.divider()
            ak_display = dict(full_paper)
            ak_display["answers_linked"] = sum(
                1 for q in full_paper.get("questions", []) if q.get("has_answer_key") or q.get("teacher_answer")
            )
            show_answer_key_structure(ak_display)

    with tab_manual:
        st.caption("Enter or edit model answers for individual questions manually.")
        if full_paper:
            _pid_safe = re.sub(r"[^a-zA-Z0-9]", "_", selected_paper_id)
            questions = full_paper.get("questions", [])

            # Group by part
            parts_list = list(dict.fromkeys(q.get("part_name", "") for q in questions))
            for pname in parts_list:
                part_qs = [q for q in questions if q.get("part_name") == pname]
                if pname:
                    st.markdown(f'<div class="card-label" style="margin-top:1rem;">{pname}</div>', unsafe_allow_html=True)

                for q in part_qs:
                    _or_sfx = "_or" if q.get("is_or_option") else ""
                    _qkey   = f"{q['question_number']}{_or_sfx}"
                    or_lbl  = " (OR)" if q.get("is_or_option") else ""
                    q_has   = q.get("teacher_answer") or q.get("has_answer_key")
                    icon    = "✅" if q_has else "📝"

                    with st.expander(f"{icon}  Q{q['question_number']}{or_lbl}  ({q['marks']}m)  —  {(q.get('question_text') or '')[:100]}"):
                        current = q.get("teacher_answer") or ""
                        new_ans = st.text_area(
                            "Model Answer",
                            value=current,
                            key=f"ans_{_pid_safe}_q{_qkey}",
                            height=120,
                            placeholder="Type the model answer here…",
                        )
                        if st.button("💾 Save Answer", key=f"save_{_pid_safe}_q{_qkey}"):
                            try:
                                r = requests.patch(
                                    f"{API_BASE}/paper/{selected_paper_id}/question/{q['question_number']}",
                                    json={"teacher_answer": new_ans},
                                    timeout=10,
                                )
                                r.raise_for_status()
                                st.success("✅ Saved")
                                st.rerun()
                            except Exception as e:
                                show_error(e)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Student Booklets
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📚  Student Booklets":
    st.markdown('<div class="pg-title">Student Booklets</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload a student handwritten answer booklet PDF — OCR extracts every answer and links it to the exam paper.</div>', unsafe_allow_html=True)

    papers = get_papers()
    if not papers:
        st.markdown('<div class="warn-banner">⚠️ &nbsp; No papers found. Upload a question paper first.</div>', unsafe_allow_html=True)
        st.stop()

    paper_options = {f"{p['paper_id']}  ({p.get('course_name','?')})": p["paper_id"] for p in papers}

    tab_upload, tab_library = st.tabs(["Upload & Evaluate", "Booklet Library"])

    with tab_upload:
        st.markdown('<div class="info-banner">🤖 &nbsp; The AI runs TrOCR on every handwritten page, extracts roll number from the cover page, then segments and evaluates each answer against the model answer.</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Exam Paper</div>', unsafe_allow_html=True)
        sel_label    = st.selectbox("Select exam paper", list(paper_options.keys()), key="bl_paper_sel", label_visibility="collapsed")
        sel_paper_id = paper_options[sel_label]
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Student Answer Booklet PDF</div>', unsafe_allow_html=True)
        bl_pdf = st.file_uploader("Upload student booklet", type=["pdf"], key="bl_pdf_upload", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        c_parse, c_eval = st.columns(2)

        if c_parse.button("🔍  Parse Booklet", type="primary", use_container_width=True, key="bl_parse"):
            if not ok:
                st.error("❌ Backend offline.")
            elif not bl_pdf:
                st.warning("Please upload a booklet PDF.")
            else:
                with st.spinner("Running OCR + LLM segmentation… this may take 1–3 minutes for a full booklet"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/booklet/upload",
                            files={"file": (bl_pdf.name, bl_pdf.getvalue(), "application/pdf")},
                            data={"paper_id": sel_paper_id},
                            timeout=OCR_TIMEOUT,
                        )
                        r.raise_for_status()
                        res = r.json()
                        st.session_state["bl_last_result"] = res
                        st.session_state["bl_last_paper"]  = sel_paper_id
                        st.session_state.pop("bl_eval_result", None)

                        n_ans = res.get("answers_found", len(res.get("questions", [])))
                        roll  = res.get("roll_number") or res.get("register_number") or "unknown"

                        if n_ans == 0:
                            st.warning(
                                f"⚠️ Booklet parsed but **0 answers extracted**. "
                                f"Roll: {roll}. OCR may have failed on handwriting. "
                                "Check that the PDF is a clear scan."
                            )
                        else:
                            st.success(f"✅ Booklet parsed — **{n_ans} answers** | Roll: {roll}")
                    except Exception as e:
                        show_error(e)

        if st.session_state.get("bl_last_result") and st.session_state.get("bl_last_paper") == sel_paper_id:
            bl_res = st.session_state["bl_last_result"]
            bid    = bl_res.get("booklet_id")

            if c_eval.button("⚡  Evaluate All", type="primary", use_container_width=True, key="bl_eval"):
                if not bid:
                    st.error("No booklet ID found. Re-parse the booklet.")
                else:
                    with st.spinner("Evaluating answers… this may take 2–5 minutes"):
                        try:
                            r = requests.post(
                                f"{API_BASE}/booklet/{bid}/evaluate",
                                json={"paper_id": sel_paper_id},
                                timeout=EVALUATE_TIMEOUT,
                            )
                            r.raise_for_status()
                            ev = r.json()
                            st.session_state["bl_eval_result"] = ev
                            total_obt  = ev.get("total_obtained", 0)
                            total_poss = ev.get("total_possible", 0)
                            pct        = ev.get("percentage", 0)
                            st.success(f"✅ Evaluation complete — {total_obt:.1f} / {total_poss} ({pct:.1f}%)")
                        except Exception as e:
                            show_error(e)

            st.divider()
            show_booklet_structure(bl_res)

            if st.session_state.get("bl_eval_result"):
                st.divider()
                st.markdown('<div class="card-label">Evaluation Results</div>', unsafe_allow_html=True)
                show_evaluation_results(st.session_state["bl_eval_result"])

    with tab_library:
        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("🔄 Refresh", key="bl_lib_refresh"):
                st.rerun()

        booklets = get_booklets()
        if not booklets:
            st.markdown('<div class="warn-banner">📭 &nbsp; No booklets uploaded yet.</div>', unsafe_allow_html=True)
        else:
            st.caption(f"{len(booklets)} booklet(s) in database")
            for b in booklets:
                roll  = b.get("roll_number") or b.get("register_number") or "Unknown"
                bid   = b.get("id") or b.get("booklet_id")
                pid   = b.get("paper_id") or "—"
                n_ans = b.get("num_answers", b.get("answers_found", "?"))
                with st.expander(f"📖  {roll}  ·  {pid}  ·  {n_ans} answers"):
                    st.json(b)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Bulk Upload
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📦  Bulk Upload":
    st.markdown('<div class="pg-title">Bulk Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload multiple student answer booklets at once for a subject. Each booklet register number is automatically detected from the Sathyabama cover page.</div>', unsafe_allow_html=True)

    papers = get_papers()
    if not papers:
        st.markdown('<div class="warn-banner">⚠️ &nbsp; No papers found. Upload a question paper first.</div>', unsafe_allow_html=True)
        st.stop()

    paper_options = {f"{p['paper_id']}  ({p.get('course_name','?')})": p["paper_id"] for p in papers}

    st.markdown('<div class="card"><div class="card-label">Select Exam Paper</div>', unsafe_allow_html=True)
    bulk_label    = st.selectbox("Exam paper for this batch", list(paper_options.keys()), key="bulk_paper_sel", label_visibility="collapsed")
    bulk_paper_id = paper_options[bulk_label]
    st.markdown('</div>', unsafe_allow_html=True)

    # Show paper summary
    bulk_paper = get_paper(bulk_paper_id)
    if bulk_paper:
        questions    = bulk_paper.get("questions", [])
        answered     = sum(1 for q in questions if q.get("has_answer_key") or q.get("teacher_answer"))
        correct_tot  = compute_correct_total_marks(questions, bulk_paper.get("parts", []))
        if answered == 0:
            st.markdown('<div class="warn-banner">⚠️ &nbsp; No model answers loaded for this paper. Upload an answer key before evaluating.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-banner">✅ &nbsp; {answered}/{len(questions)} model answers ready · Total marks: {correct_tot:.0f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Student Booklet PDFs (select multiple)</div>', unsafe_allow_html=True)
    bulk_files = st.file_uploader(
        "Upload multiple booklets",
        type=["pdf"],
        key="bulk_pdfs",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    auto_evaluate = st.checkbox("Auto-evaluate after parsing each booklet", value=True, key="bulk_auto_eval")

    if bulk_files:
        st.info(f"📂 {len(bulk_files)} file(s) selected")

    if st.button("🚀  Parse & Evaluate All Booklets", type="primary", use_container_width=True, key="bulk_run"):
        if not ok:
            st.error("❌ Backend offline.")
        elif not bulk_files:
            st.warning("Please upload at least one booklet PDF.")
        else:
            results_list = []
            progress_bar = st.progress(0, text="Starting…")
            status_area  = st.empty()

            for idx, f in enumerate(bulk_files):
                pct_done = idx / len(bulk_files)
                progress_bar.progress(pct_done, text=f"Processing {f.name} ({idx+1}/{len(bulk_files)})…")
                status_area.markdown(f'<div class="info-banner">⏳ &nbsp; Parsing {f.name}…</div>', unsafe_allow_html=True)

                row = {"file": f.name, "roll": "—", "reg": "—", "answers": 0,
                       "score": "—", "max": "—", "pct": "—", "status": "pending"}

                # Step 1: Parse booklet
                try:
                    r = requests.post(
                        f"{API_BASE}/booklet/upload",
                        files={"file": (f.name, f.getvalue(), "application/pdf")},
                        data={"paper_id": bulk_paper_id},
                        timeout=OCR_TIMEOUT,
                    )
                    r.raise_for_status()
                    parsed = r.json()
                    bid    = parsed.get("booklet_id")
                    row["roll"]    = parsed.get("roll_number", "—")
                    row["reg"]     = parsed.get("register_number", "—")
                    row["answers"] = parsed.get("answers_found", len(parsed.get("questions", [])))
                    row["status"]  = "parsed"

                except Exception as e:
                    row["status"] = f"parse_error: {e}"
                    results_list.append(row)
                    continue

                # Step 2: Evaluate
                if auto_evaluate and bid and row["answers"] > 0:
                    status_area.markdown(f'<div class="info-banner">⚡ &nbsp; Evaluating {f.name} (roll: {row["roll"]})…</div>', unsafe_allow_html=True)
                    try:
                        re2 = requests.post(
                            f"{API_BASE}/booklet/{bid}/evaluate",
                            json={"paper_id": bulk_paper_id},
                            timeout=EVALUATE_TIMEOUT,
                        )
                        re2.raise_for_status()
                        ev = re2.json()
                        row["score"]  = f"{ev.get('total_obtained',0):.1f}"
                        row["max"]    = f"{ev.get('total_possible',0):.0f}"
                        row["pct"]    = f"{ev.get('percentage',0):.1f}%"
                        row["status"] = "evaluated"
                    except Exception as e:
                        row["status"] = f"eval_error: {e}"

                results_list.append(row)

            progress_bar.progress(1.0, text="Done ✅")
            status_area.empty()

            # Summary table
            st.markdown("---")
            st.markdown('<div class="card-label">Batch Results</div>', unsafe_allow_html=True)

            n_ok  = sum(1 for r in results_list if r["status"] == "evaluated")
            n_err = sum(1 for r in results_list if "error" in r["status"])
            st.caption(f"✅ {n_ok} evaluated  ·  ⚠️ {n_err} errors  ·  Total: {len(results_list)}")

            df = pd.DataFrame([{
                "File":        r["file"],
                "Roll No.":    r["roll"],
                "Reg. No.":    r["reg"],
                "Answers":     r["answers"],
                "Score":       r["score"],
                "Max Marks":   r["max"],
                "%":           r["pct"],
                "Status":      r["status"],
            } for r in results_list])

            st.dataframe(df, use_container_width=True, hide_index=True)
            st.session_state["bulk_results_df"] = df

            # Download
            st.download_button(
                "⬇️  Download Batch Results CSV",
                data=df.to_csv(index=False),
                file_name=f"batch_{bulk_paper_id}.csv",
                mime="text/csv",
                key="bulk_dl",
            )

            # Error details
            errors = [r for r in results_list if "error" in r["status"]]
            if errors:
                st.markdown('<div class="warn-banner">⚠️ &nbsp; Some booklets had errors. See details below.</div>', unsafe_allow_html=True)
                for r in errors:
                    with st.expander(f"❌ {r['file']}"):
                        st.error(r["status"])

    # Show previous batch results if available
    elif "bulk_results_df" in st.session_state:
        st.markdown("---")
        st.caption("Previous batch results (still in session):")
        st.dataframe(st.session_state["bulk_results_df"], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Analytics
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊  Analytics":
    st.markdown('<div class="pg-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">System statistics, AI scoring accuracy, and paper-level insights.</div>', unsafe_allow_html=True)

    if not ok:
        st.warning("Backend is offline. Start it to view analytics.")
    else:
        c_r, _ = st.columns([1, 5])
        with c_r:
            if st.button("🔄 Refresh"):
                st.rerun()

        try:
            stats = requests.get(f"{API_BASE}/stats", timeout=10).json()
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("total_submissions",0)}</div><div class="mtile-label">Submissions</div></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("evaluated",0)}</div><div class="mtile-label">Evaluated</div></div>', unsafe_allow_html=True)
            t3.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("average_score",0):.1f}</div><div class="mtile-label">Avg Score</div></div>', unsafe_allow_html=True)
            t4.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("average_evaluation_time_sec",0):.1f}s</div><div class="mtile-label">Avg Time</div></div>', unsafe_allow_html=True)
            t5.markdown(f'<div class="mtile"><div class="mtile-val">{stats.get("total_exam_papers",0)}</div><div class="mtile-label">Papers</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Stats error: {e}")

        st.divider()

        # Paper breakdown with correct marks
        papers = get_papers()
        if papers:
            st.markdown('<div class="card-label">Papers on File</div>', unsafe_allow_html=True)
            rows = []
            for p in papers:
                full = get_paper(p["paper_id"])
                if full:
                    correct = compute_correct_total_marks(full.get("questions", []), full.get("parts", []))
                    answered = sum(1 for q in full.get("questions", []) if q.get("has_answer_key") or q.get("teacher_answer"))
                else:
                    correct  = p.get("total_marks", 0)
                    answered = 0

                rows.append({
                    "Paper ID":     p.get("paper_id", "—"),
                    "Course":       p.get("course_name") or p.get("course_code", "—"),
                    "Exam":         p.get("exam_name", "—"),
                    "Set":          p.get("set_name", "—"),
                    "Total Marks":  correct,
                    "Questions":    p.get("num_questions", 0),
                    "Answers Ready": answered,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown('<div class="card-label">AI Accuracy Metrics</div>', unsafe_allow_html=True)
        try:
            mx   = requests.get(f"{API_BASE}/metrics", timeout=10).json()
            st.caption(f"Updated: {mx.get('last_updated','—')}  ·  Total evaluated: {mx.get('total_evaluated',0)}")
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
                st.info("No graded results yet. Evaluate some booklets to see accuracy metrics.")
        except Exception as e:
            st.warning(f"Metrics error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️  Settings":
    st.markdown('<div class="pg-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Configure API keys and backend connection. Add to <code>.env</code> to persist.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">API Keys</div>', unsafe_allow_html=True)
    with st.form("cfg"):
        ak3    = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY",""), placeholder="gsk_…")
        st.markdown("**Groq Model**")
        groq_m = st.text_input("Model", value=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"))
        if st.form_submit_button("💾  Save", use_container_width=True):
            if ak3: os.environ["GROQ_API_KEY"] = ak3
            os.environ["GROQ_MODEL"] = groq_m
            st.success("✅ Saved for this session. Add to .env to persist.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Backend Connection</div>', unsafe_allow_html=True)
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
    st.caption(f"Upload: {UPLOAD_TIMEOUT}s  ·  OCR: {OCR_TIMEOUT}s  ·  Evaluate: {EVALUATE_TIMEOUT}s  ·  Poll max: {POLL_TIMEOUT}s")
    st.info("Increase OCR_TIMEOUT / EVALUATE_TIMEOUT in dashboard.py for very large booklets (e.g. 20+ pages).")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Groq Provider Status</div>', unsafe_allow_html=True)
    gk = os.getenv("GROQ_API_KEY","")
    st.markdown(f'<div class="mtile" style="display:inline-block;min-width:120px;"><div class="mtile-val">{"✅" if gk else "—"}</div><div class="mtile-label">Groq Key</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)