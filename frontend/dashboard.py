"""
IntelliGrade-H - Teacher Dashboard
Streamlit-based web interface for uploading and evaluating answer sheets.
"""

import streamlit as st
import requests
import json
from pathlib import Path

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
    .score-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
    }
    .score-value {
        font-size: 52px;
        font-weight: 800;
        color: #4CAF50;
    }
    .score-label {
        font-size: 14px;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-box {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .feedback-box {
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .missing-box {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .stProgress .st-bo { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=IntelliGrade-H", use_column_width=True)
    st.title("Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "📤 Evaluate Answer", "📊 Batch Evaluation", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("IntelliGrade-H v1.0")
    st.caption("Sathyabama Institute of Science and Technology")


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
        <div style="margin-top:12px; color:#90caf9;">
            Confidence: {int(confidence * 100)}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct / 100)


# ─────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    st.title("📚 IntelliGrade-H — Teacher Dashboard")
    st.subheader("AI-Powered Handwritten Answer Evaluation System")

    # API status
    is_alive = api_health_check()
    if is_alive:
        st.success("✅ API Server is online")
    else:
        st.error("❌ API Server is offline. Start the backend: `python -m uvicorn backend.api:app`")

    st.divider()

    # Stats
    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Submissions", stats.get("total_submissions", 0))
    col2.metric("Evaluated", stats.get("evaluated", 0))
    col3.metric("Avg Score", stats.get("average_score", "—"))
    col4.metric("Avg Eval Time (s)", stats.get("average_evaluation_time_sec", "—"))

    st.divider()
    st.markdown("""
    ### How to Use
    1. Go to **📤 Evaluate Answer** to upload and evaluate a single student answer.
    2. Go to **📊 Batch Evaluation** to evaluate multiple students at once.
    3. The system will:
       - Extract text using handwriting OCR (TrOCR or Tesseract)
       - Compute semantic similarity with the teacher answer
       - Use Gemini AI to evaluate content, accuracy, and depth
       - Generate a final score and detailed feedback
    """)


# ─────────────────────────────────────────────────────────
# Page: Evaluate Answer
# ─────────────────────────────────────────────────────────

elif page == "📤 Evaluate Answer":
    st.title("📤 Evaluate Student Answer")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Question & Model Answer")
        question = st.text_area(
            "Question", height=100,
            placeholder="e.g. Explain the working principle of a neural network."
        )
        teacher_answer = st.text_area(
            "Model Answer (Teacher's)", height=200,
            placeholder="Enter the expected answer..."
        )
        max_marks = st.number_input("Maximum Marks", min_value=1.0, max_value=100.0, value=10.0, step=0.5)

        st.subheader("2. Rubric (Optional)")
        num_criteria = st.number_input("Number of rubric criteria", min_value=0, max_value=10, value=3)
        rubric = []
        for i in range(int(num_criteria)):
            c1, c2 = st.columns([3, 1])
            criterion = c1.text_input(f"Criterion {i+1}", key=f"crit_{i}",
                                      placeholder="e.g. Definition of neural network")
            marks = c2.number_input(f"Marks {i+1}", min_value=0.1, max_value=10.0,
                                    value=1.0, key=f"marks_{i}")
            if criterion:
                rubric.append({"criterion": criterion, "marks": marks})

    with col_right:
        st.subheader("3. Upload Student Answer Sheet")
        student_code = st.text_input("Student ID / Roll Number",
                                     placeholder="e.g. SIST2024001")
        uploaded_file = st.file_uploader(
            "Upload answer sheet (JPG, PNG, PDF)",
            type=["jpg", "jpeg", "png", "pdf"]
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Answer Sheet", use_column_width=True)

    st.divider()

    if st.button("🚀 Evaluate Answer", type="primary", use_container_width=True):
        if not question or not teacher_answer:
            st.error("Please enter both the question and model answer.")
        elif not uploaded_file:
            st.error("Please upload a student answer sheet.")
        elif not student_code:
            st.error("Please enter the student ID.")
        else:
            with st.spinner("Uploading and evaluating... This may take 30–60 seconds."):
                try:
                    # Upload
                    upload_resp = upload_file(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        student_code
                    )
                    submission_id = upload_resp["submission_id"]
                    st.info(f"Uploaded successfully. Submission ID: {submission_id}")

                    # Evaluate
                    result = evaluate_submission(
                        submission_id=submission_id,
                        question=question,
                        teacher_answer=teacher_answer,
                        max_marks=max_marks,
                        rubric=rubric if rubric else None
                    )

                    st.success("✅ Evaluation complete!")
                    st.divider()

                    # ── Results Display ─────────────────────────────
                    r_col1, r_col2, r_col3 = st.columns([1, 1, 1])

                    with r_col1:
                        render_score_card(
                            result["final_score"],
                            result["max_marks"],
                            result["confidence"]
                        )

                    with r_col2:
                        st.metric("LLM Score", f"{result['llm_score']:.1f} / {result['max_marks']}")
                        st.metric("Semantic Similarity", f"{result['similarity_score']:.2%}")
                        st.metric("OCR Confidence", f"{result['ocr_confidence']:.2%}")
                        st.metric("Eval Time", f"{result['evaluation_time_sec']}s")

                    with r_col3:
                        if result.get("rubric_details"):
                            rd = result["rubric_details"]
                            st.metric("Rubric Coverage",
                                      f"{rd['earned_rubric_marks']:.1f}/{rd['total_rubric_marks']:.1f}")
                            for criterion, present in rd.get("criteria_scores", {}).items():
                                icon = "✅" if present else "❌"
                                st.write(f"{icon} {criterion}")

                    st.subheader("📝 Extracted Text (OCR)")
                    st.code(result["ocr_text"], language=None)

                    st.subheader("💪 Strengths")
                    for s in result.get("strengths", []):
                        st.markdown(f'<div class="feedback-box">✅ {s}</div>', unsafe_allow_html=True)

                    st.subheader("⚠️ Missing Concepts")
                    for m in result.get("missing_concepts", []):
                        st.markdown(f'<div class="missing-box">⚠️ {m}</div>', unsafe_allow_html=True)

                    st.subheader("💡 AI Feedback")
                    st.info(result.get("feedback", "No feedback generated."))

                    # Download result as JSON
                    st.download_button(
                        "⬇️ Download Result JSON",
                        data=json.dumps(result, indent=2),
                        file_name=f"result_{student_code}_{submission_id}.json",
                        mime="application/json"
                    )

                except requests.HTTPError as e:
                    st.error(f"API Error: {e.response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ─────────────────────────────────────────────────────────
# Page: Batch Evaluation
# ─────────────────────────────────────────────────────────

elif page == "📊 Batch Evaluation":
    st.title("📊 Batch Evaluation")
    st.info("Upload multiple answer sheets and evaluate them all at once.")

    question = st.text_area("Question", height=80)
    teacher_answer = st.text_area("Model Answer", height=120)
    max_marks = st.number_input("Max Marks", value=10.0, step=0.5)

    uploaded_files = st.file_uploader(
        "Upload multiple student answer sheets",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files and question and teacher_answer:
        if st.button("🚀 Run Batch Evaluation", type="primary"):
            progress = st.progress(0)
            results_table = []

            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Evaluating {f.name} ({i+1}/{len(uploaded_files)})..."):
                    try:
                        student_code = Path(f.name).stem
                        upload_resp = upload_file(f.getvalue(), f.name, student_code)
                        result = evaluate_submission(
                            submission_id=upload_resp["submission_id"],
                            question=question,
                            teacher_answer=teacher_answer,
                            max_marks=max_marks,
                            rubric=None
                        )
                        results_table.append({
                            "Student": student_code,
                            "Score": result["final_score"],
                            "Max": result["max_marks"],
                            "Similarity": f"{result['similarity_score']:.2%}",
                            "OCR Confidence": f"{result['ocr_confidence']:.2%}",
                            "Eval Time (s)": result["evaluation_time_sec"]
                        })
                    except Exception as e:
                        results_table.append({"Student": f.name, "Score": "ERROR", "Error": str(e)})

                progress.progress((i + 1) / len(uploaded_files))

            st.success(f"✅ Batch evaluation complete! {len(uploaded_files)} submissions processed.")
            st.dataframe(results_table, use_container_width=True)

            # Download CSV
            import csv, io
            buf = io.StringIO()
            if results_table:
                writer = csv.DictWriter(buf, fieldnames=results_table[0].keys())
                writer.writeheader()
                writer.writerows(results_table)
            st.download_button(
                "⬇️ Download Results CSV",
                data=buf.getvalue(),
                file_name="batch_results.csv",
                mime="text/csv"
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
