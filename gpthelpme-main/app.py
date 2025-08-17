# app.py
# Streamlit UI for generating MLB strikeout picks from a lineup CSV
# Supports two paths:
#   1) Heuristic path (make_picks)
#   2) Smart MC path (make_picks_smart_mc) using model-driven simulations
# Includes optional edge ranking and CSV downloads.

from __future__ import annotations

import io
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# ------------------- Safe optional imports with graceful fallbacks -------------------
# These modules live in your repo. If they aren't importable yet, the UI still loads
# and shows a friendly message when you try to run the missing path.

# pick_selector: edge ranking helpers
try:
    from src.pick_selector import rank_strikeout_edges, augment_with_edges  # type: ignore
except Exception:
    def augment_with_edges(df: pd.DataFrame) -> pd.DataFrame:
        st.warning("augment_with_edges not available (src/pick_selector.py missing or import failed). Returning input.")
        return df

    def rank_strikeout_edges(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        st.warning("rank_strikeout_edges not available (src/pick_selector.py missing or import failed). Returning head().")
        return df.head(top_n)

# picks: both execution paths
_HEURISTIC_AVAILABLE = True
_SMART_MC_AVAILABLE = True

try:
    from src.picks import make_picks, make_picks_smart_mc  # type: ignore
except Exception as e:
    _HEURISTIC_AVAILABLE = False
    _SMART_MC_AVAILABLE = False

    def make_picks(*args, **kwargs):  # type: ignore
        raise ImportError("make_picks unavailable: src/picks.py not found or import failed.")

    def make_picks_smart_mc(*args, **kwargs):  # type: ignore
        raise ImportError("make_picks_smart_mc unavailable: src/picks.py not found or import failed.")

# smart_csv_processor: maps the uploaded daily lineup CSV to matchup tasks
try:
    from src.smart_csv_processor import map_daily_csv_to_tasks  # type: ignore
except Exception:
    def map_daily_csv_to_tasks(df: pd.DataFrame) -> pd.DataFrame:
        # Fall back to identity mapping; downstream code should still accept df
        st.info("map_daily_csv_to_tasks not available (src/smart_csv_processor.py). Using CSV as-is.")
        return df

# Optional: Smart MC models (for capability checks / version display)
_SMART_MC_META: Dict[str, Any] = {}
try:
    from src.smart_mc_models import SmartMonteCarloPredictor  # type: ignore
    _SMART_MC_META["smart_mc_loaded"] = True
except Exception:
    _SMART_MC_META["smart_mc_loaded"] = False

# Acknowledge your new modules for downstream runs (not required to import here)
# - src/build_team_batting_profiles.py (builder script)
# - src/fallbacks.py (fallback logic used across the project)

# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="MLBSKYNET • Strikeout Picks",
    page_icon="⚾",
    layout="wide",
)

st.title("MLBSKYNET • Strikeout Picks")
st.caption("Smart MC simulations + edges. Upload your daily lineup CSV → run picks → download results.")

# ------------------- Sidebar Controls -------------------
with st.sidebar:
    st.header("Run Settings")

    run_mode = st.radio(
        "Select run mode",
        options=["Smart MC (Models)", "Heuristic (Fallback)"],
        index=0,
        help=(
            "Smart MC uses model-driven simulations with guardrails (timeout, PA caps) and parallelism. "
            "Heuristic path is a quicker fallback using handcrafted logic."
        ),
    )

    st.subheader("Simulation Controls")
    sims_per_pitcher = st.number_input(
        "Simulations per pitcher",
        min_value=500,
        max_value=100_000,
        value=5_000,
        step=500,
        help=(
            "Total Monte Carlo draws for each pitcher vs. lineup task. "
            "Higher = slower but tighter confidence intervals."
        ),
    )

    n_jobs = st.number_input(
        "Parallel jobs (n_jobs)",
        min_value=-1,
        max_value=64,
        value=-1,
        step=1,
        help=(
            "Joblib-style parallelism. Use -1 for all cores, or a positive integer to limit. "
            "Streamlit is single-process; parallelism happens inside the task execution."
        ),
    )

    per_game_timeout_s = st.number_input(
        "Per-game timeout (seconds)",
        min_value=10,
        max_value=3600,
        value=120,
        step=10,
        help=(
            "Safety guard: any single game taking longer than this is aborted and skipped to keep the run responsive."
        ),
    )

    max_pas_per_game = st.number_input(
        "Max PAs per game (cap)",
        min_value=30,
        max_value=120,
        value=78,
        step=2,
        help=(
            "Guardrail: caps plate appearances sampled per game context to avoid pathological runtimes."
        ),
    )

    st.subheader("Edges & Output")
    add_edges = st.checkbox(
        "Augment with edge metrics",
        value=True,
        help="Compute derived edges (e.g., strikeout value vs. lines) when helpers are available.",
    )
    top_n_edges = st.slider(
        "Top N edges to display",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

# ------------------- File Upload -------------------
st.markdown("### 1) Upload daily lineup CSV")

uploaded = st.file_uploader(
    "CSV with pitcher vs. lineup tasks (one or many games).",
    type=["csv"],
    accept_multiple_files=False,
    help=(
        "Typical columns include date, pitcher_id/name, opponent hitters, handedness mix, etc. "
        "Your src/smart_csv_processor.py will map this CSV to the internal tasks format."
    ),
)

if uploaded is not None:
    try:
        raw_bytes = uploaded.read()
        df_input = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
    except Exception:
        uploaded.seek(0)
        df_input = pd.read_csv(uploaded)

    st.success(f"Loaded CSV with {len(df_input):,} rows and {len(df_input.columns)} columns.")
    with st.expander("Preview uploaded CSV", expanded=False):
        st.dataframe(df_input.head(200), use_container_width=True)
else:
    df_input = None

# ------------------- Run Button -------------------
st.markdown("### 2) Configure & Run")
run_clicked = st.button("▶️ Run Picks", type="primary", use_container_width=True)

# ------------------- Utility: download helper -------------------
def _download_link_for_df(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ------------------- Main Execution -------------------
if run_clicked:
    if df_input is None:
        st.error("Please upload a CSV first.")
        st.stop()

    # Map CSV → internal tasks
    with st.status("Mapping CSV to tasks…", expanded=False) as s:
        try:
            task_df = map_daily_csv_to_tasks(df_input)
            if not isinstance(task_df, pd.DataFrame) or task_df.empty:
                st.warning("Task mapping returned empty or non-DataFrame. Falling back to raw CSV.")
                task_df = df_input
            s.update(label=f"Mapped {len(task_df):,} tasks.")
        except Exception as e:
            st.exception(e)
            st.stop()

    st.markdown("### 3) Results")

    try:
        if run_mode.startswith("Smart MC"):
            if not _SMART_MC_AVAILABLE:
                st.error("Smart MC path unavailable: src/picks.py import failed. Check repo state.")
                st.stop()

            with st.status("Running Smart MC simulations…", expanded=False) as s:
                try:
                    results_df: pd.DataFrame = make_picks_smart_mc(
                        task_df,
                        sims_per_pitcher=int(sims_per_pitcher),
                        n_jobs=int(n_jobs),
                        per_game_timeout_s=int(per_game_timeout_s),
                        max_pas_per_game=int(max_pas_per_game),
                    )
                    s.update(label="Smart MC complete.")
                except Exception as e:
                    st.exception(e)
                    st.stop()
        else:
            if not _HEURISTIC_AVAILABLE:
                st.error("Heuristic path unavailable: src/picks.py import failed. Check repo state.")
                st.stop()

            with st.status("Running heuristic picks…", expanded=False) as s:
                try:
                    results_df: pd.DataFrame = make_picks(task_df)
                    s.update(label="Heuristic run complete.")
                except Exception as e:
                    st.exception(e)
                    st.stop()

        if results_df.empty:
            st.warning("No results returned. Verify inputs and try again.")
            st.stop()

        # Optional edges
        if add_edges:
            with st.status("Computing edges…", expanded=False):
                try:
                    results_df = augment_with_edges(results_df)
                except Exception as e:
                    st.warning(f"augment_with_edges failed: {e}")

        # Show main table
        st.subheader("All Results")
        st.dataframe(results_df, use_container_width=True)
        _download_link_for_df(results_df, "results_all.csv", "💾 Download results (CSV)")

        # Top edges summary
        if add_edges:
            try:
                top_edges = rank_strikeout_edges(results_df, top_n=int(top_n_edges))
                st.subheader(f"Top {int(top_n_edges)} Edges")
                st.dataframe(top_edges, use_container_width=True)
                _download_link_for_df(top_edges, "results_top_edges.csv", "💾 Download top edges (CSV)")
            except Exception as e:
                st.warning(f"Edge ranking failed: {e}")

    except Exception as e:
        st.error("Run failed due to an unexpected error. See details below.")
        st.code("\n" + traceback.format_exc())

# ------------------- Footer / Diagnostics -------------------
with st.expander("Diagnostics & Versions", expanded=False):
    meta = {
        "smart_mc_available": _SMART_MC_AVAILABLE,
        "heuristic_available": _HEURISTIC_AVAILABLE,
        "smart_mc_models_loaded": _SMART_MC_META.get("smart_mc_loaded", False),
        "working_dir": str(Path.cwd()),
    }
    st.json(meta)
    st.markdown(
        "If imports are failing, verify your repo layout and that Streamlit is launched from the project root: `streamlit run app.py`."
    )
