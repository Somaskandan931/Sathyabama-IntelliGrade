"""
scripts/create_db.py
Standalone script to initialize the database (SQLite or PostgreSQL).

Usage:
    python scripts/create_db.py           # creates tables using DATABASE_URL from .env
    python scripts/create_db.py --drop    # drops all tables first (destructive!)
"""

import argparse
import os
import sys
from pathlib import Path

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, Base, engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="Drop all tables before creating")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL", "sqlite:///./intelligrade.db")
    print(f"Database: {db_url}")

    if args.drop:
        confirm = input("⚠️  This will DELETE all data. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        Base.metadata.drop_all(bind=engine)
        print("✅ All tables dropped.")

    init_db()
    print("✅ Database tables created successfully.")
    print("   Tables:", [t for t in Base.metadata.tables.keys()])


if __name__ == "__main__":
    main()
