#!/usr/bin/env python3
"""
IntelliGrade-H - Quick start script
Runs both the API server and Streamlit dashboard in a single command.

Usage:
    python run.py          # start both API + dashboard
    python run.py api      # start API only
    python run.py ui       # start dashboard only
    python run.py init     # initialise database only
"""

import sys
import subprocess
import threading
import os

API_CMD  = ["uvicorn", "backend.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
UI_CMD   = ["streamlit", "run", "frontend/dashboard.py"]


def run_api():
    subprocess.run(API_CMD)


def run_ui():
    subprocess.run(UI_CMD)


def init_db():
    from backend.database import init_db as _init
    _init()
    print("✅ Database initialised.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "init":
        init_db()
        return

    if mode == "api":
        run_api()
    elif mode == "ui":
        run_ui()
    else:
        print("🚀 Starting IntelliGrade-H...")
        print("   API  → http://localhost:8000")
        print("   Docs → http://localhost:8000/docs")
        print("   UI   → http://localhost:8501")
        print("   Press Ctrl+C to stop\n")

        t = threading.Thread(target=run_api, daemon=True)
        t.start()
        run_ui()


if __name__ == "__main__":
    main()