"""Settings shared by both tools: API key, time zone, data folders."""

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EASTERN = ZoneInfo("America/New_York")
DATA_DIR = Path(os.getenv("LAB_DATA_DIR", ROOT / "data"))        # override to test on sample data
REPORTS_DIR = Path(os.getenv("LAB_REPORTS_DIR", ROOT / "reports"))


def api_key() -> str:
    key = os.getenv("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is missing. Copy .env.example to .env and add your key.")
    return key


def ns_to_eastern(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=EASTERN)
