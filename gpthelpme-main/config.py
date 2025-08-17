# src/config.py
from __future__ import annotations
import os
from pathlib import Path

# --- Project roots (robust to how Streamlit launches) ---
ROOT_DIR   = Path(__file__).resolve().parents[1]   # <- project root (folder that contains /src)
DATA_DIR   = Path(os.getenv("DATA_DIR", ROOT_DIR / "data")).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models")).resolve()

# Common subfolders
LINEUPS_DIR = DATA_DIR / "lineups"
OUTPUT_DIR  = DATA_DIR / "outputs"

# --- Core knobs ---
N_SIMS_DEFAULT: int = int(os.getenv("N_SIMS_DEFAULT", "3000"))

# Optional: The Odds API key (set this in Replit Secrets → ODDS_API_KEY)
ODDS_API_KEY: str | None = os.getenv("ODDS_API_KEY")

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

# Create dirs at import time
ensure_dirs(DATA_DIR, MODELS_DIR, LINEUPS_DIR, OUTPUT_DIR)

def get_odds_api_key() -> str | None:
    return ODDS_API_KEY
