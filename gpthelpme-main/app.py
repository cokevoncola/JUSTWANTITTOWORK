# app.py
# Streamlit UI for generating MLB picks from a lineup CSV
# Uses team-aware batter fallbacks (player split -> player season -> team split -> team season).

import io
import traceback
from pathlib import Path
from typing import List, Dict, Any, Union

import pandas as pd
import streamlit as st

# ------------------- Optional edges (app stays resilient if absent) -------------------
try:
    from src.pick_selector import rank_strikeout_edges, augment_with_edges
except Exception:
    def augment_with_edges(df): return df
    def rank_strikeout_edges(df, top_n=10): return (pd.DataFrame(), pd.DataFrame())

# ------------------- Pipeline cores -------------------
from src.picks import make_picks, make_picks_smart_mc
from src.smart_csv_processor import process_smart_csv

# ------------------- Team fallbacks (keeps your CSVs as-is) -------------------
from src.fallbacks import load_team_tables, resolve_batter_rate_with_fallback

# ------------------- Page config -------------------
st.set_page_config(page_title="MLBSKYNET – Coladyne Systems", page_icon="⚾", layout="wide")
st.title("MLBSKYNET – Coladyne Systems")
st.header("Generate Picks from Lineup CSV")
st.markdown(
    """
Upload your lineup CSV (columns like:  
`team code`, `game_date`, `game_number`, `mlb id`, `player name`, `batting order`, `home/away`).  
We auto-detect SPs by **game_number**, build 9-man opponent lineups, enrich with **team-aware fallbacks**, and produce picks.
"""
)

# ------------------- Constants -------------------
CAPMODE_MAP = {"auto": "hybrid", "soft": "bf", "hard": "pitch"}
SEASON_YEAR_DEFAULT = 2025

# ------------------- Load team tables once -------------------
tables_status = st.empty()
try:
    TEAM_TABLES = load_team_tables()   # season backbone + LHP/RHP Fangraphs splits (from models/)
    tables_status.info("✅ Team tables loaded (season + LHP/RHP splits).")
except Exception as e:
    TEAM_TABLES = None
    tables_status.warning(f"⚠️ Team tables unavailable: {type(e).__name__}: {e}")

# ------------------- Adapter: smart_csv_processor -> simulator tasks -------------------
def _first_letter(s: Union[str, None]) -> str:
    try:
        return (s or "R").strip().upper()[0]
    except Exception:
        return "R"

def build_sim_tasks_from_smart_csv(
    file_path_or_df: Union[str, Path, pd.DataFrame],
    *,
    cap_mode_ui: str = "auto",
    default_pitch_cap: int = 95,
    hook_aggr: float = 1.0,
    use_heads: bool = True,
    strict_model_only: bool = True,
    season_year: int = SEASON_YEAR_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Adapter: smart_csv_processor -> Smart MC simulator tasks.
    Adds per-batter k_rate / bb_rate via player-first fallbacks:
      player split -> player season -> team split -> team season
    """
    # Allow passing a path or a preloaded DataFrame
    if isinstance(file_path_or_df, (str, Path)):
        raw_tasks = process_smart_csv(str(file_path_or_df))
    else:
        # write temp CSV for existing processor
        tmp_path = "lineups_runtime.csv"
        cast_df = pd.DataFrame(file_path_or_df)
        cast_df.to_csv(tmp_path, index=False)
        raw_tasks = process_smart_csv(tmp_path)

    tasks: List[Dict[str, Any]] = []

    for t in raw_tasks:
        opp_lineup = []
        seen = set()

        pitcher_hand = _first_letter(t.get("pitcher_throws"))  # if present
        opponent_team_code = t.get("opponent")

        for row in (t.get("lineup") or []):
            key = (row.get("mlbid"), row.get("name"))
            if key in seen:
                continue
            seen.add(key)

            b_hand = _first_letter(row.get("hand"))

            # Batter dict can include own split/season stats if already merged upstream
            batter_dict = {
                "k_rate": row.get("k_rate"),
                "bb_rate": row.get("bb_rate"),
                "k_rate_vs_LHP": row.get("k_rate_vs_LHP"),
                "k_rate_vs_RHP": row.get("k_rate_vs_RHP"),
                "bb_rate_vs_LHP": row.get("bb_rate_vs_LHP"),
                "bb_rate_vs_RHP": row.get("bb_rate_vs_RHP"),
                "team": row.get("team") or opponent_team_code,  # batter's MLB team if present
            }

            k_rate = None
            bb_rate = None
            if TEAM_TABLES is not None:
                team_for_fallback = batter_dict.get("team") or opponent_team_code
                k_rate = resolve_batter_rate_with_fallback(
                    batter=batter_dict, tables=TEAM_TABLES,
                    team=team_for_fallback, year=season_year,
                    stat="k_rate", pitcher_hand=pitcher_hand
                )
                bb_rate = resolve_batter_rate_with_fallback(
                    batter=batter_dict, tables=TEAM_TABLES,
                    team=team_for_fallback, year=season_year,
                    stat="bb_rate", pitcher_hand=pitcher_hand
                )

            opp_lineup.append({
                "batter_id": row.get("mlbid"),
                "name": row.get("name"),
                "hand": b_hand,
                # Resolved rates so the simulator can use model-only inputs
                "k_rate": k_rate,
                "bb_rate": bb_rate,
            })

        if len(opp_lineup) < 7:
            # Skip malformed games
            continue

        tasks.append({
            "pitcher": {"id": t.get("pitcher_mlbid"), "name": t.get("pitcher_name")},
            "opponent_lineup": opp_lineup[:9],
            "expected_bf": 24,
            "pitch_cap": int(default_pitch_cap),
            "pitches_per_pa": 3.9,
            "cap_mode": CAPMODE_MAP.get(cap_mode_ui.lower(), "hybrid"),
            "hook_aggr": float(hook_aggr),
            "use_heads": bool(use_heads),
            "strict_model_only": bool(strict_model_only),  # harmless if simulator ignores
            # meta
            "game_id": t.get("game_id"),
            "team": t.get("team"),
            "opponent": t.get("opponent"),
            "home_away": t.get("home_away"),
            "venue": t.get("venue"),
            "pitcher_throws": t.get("pitcher_throws"),
        })

    # UI debug
    st.session_state["_resolved_inputs_preview"] = [{
        "pitcher_name": x["pitcher"].get("name"),
        "pitch_cap": x.get("pitch_cap"),
        "cap_mode": x.get("cap_mode"),
        "lineup_size": len(x.get("opponent_lineup", [])),
        "k_rates_seen": sum(1 for b in x.get("opponent_lineup", []) if b.get("k_rate") is not None),
        "bb_rates_seen": sum(1 for b in x.get("opponent_lineup", []) if b.get("bb_rate") is not None),
    } for x in tasks]

    return tasks

# ------------------- Sidebar controls -------------------
with st.sidebar:
    st.subheader("Simulation Settings")
    num_sims = st.number_input("Simulations per pitcher", min_value=100, max_value=100_000, value=3000, step=100, key="num_sims")
    st.markdown("---")
    st.subheader("Smart MC (Models) Options")
    cap_mode = st.selectbox(
        "Pitch cap mode",
        options=["auto", "soft", "hard"],
        index=0,
        help=("auto → hybrid (tightest cap)\nsoft → BF envelope only\nhard → pitch-cap only"),
        key="cap_mode",
    )
    pitch_cap = st.number_input("Default pitch cap (fallback)", min_value=50, max_value=130, value=95, step=1, key="pitch_cap")
    hook_aggr = st.slider("Hook aggressiveness", min_value=0.5, max_value=1.5, value=1.0, step=0.05, key="hook_aggr")
    use_heads = st.checkbox("Use trained heads (if available)", value=True, key="use_heads")
    strict_model_only = st.checkbox("Strict model-only (no heuristic overrides)", value=True, key="strict_model_only")

st.divider()

# ------------------- File uploader -------------------
uploaded_lineups = st.file_uploader("Upload lineup CSV", type=["csv"], key="lineup_csv_main")

tmp_path = None
preview_df = None
if uploaded_lineups:
    try:
        content = uploaded_lineups.getvalue().decode("utf-8", errors="ignore")
        tmp_path = "lineups_runtime.csv"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)

        preview_df = pd.read_csv(io.StringIO(content), nrows=15, encoding="utf-8")
        with st.expander("Preview uploaded CSV (first 15 rows)", expanded=False):
            st.dataframe(preview_df, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Failed to read lineup CSV: {type(e).__name__}: {e}")
        tmp_path = None

# ------------------- Environment diagnostics -------------------
with st.expander("Environment diagnostics", expanded=False):
    needed = [
        "models/comprehensive_trained_models.joblib",
        "models/batter_heads.joblib",
        "models/tendency_tables.json",
        "models/pitcher_profiles.parquet",
        "models/batter_profiles.parquet",
        "models/2025_team_batter_stats.csv",
        "models/2025_team_batting_vs_LHP.csv",
        "models/2025_team_batting_vs_RHP.csv",
    ]
    rows = []
    for p in needed:
        exists = Path(p).exists()
        size_mb = round(Path(p).stat().st_size/1e6, 2) if exists else 0.0
        rows.append({"path": p, "exists": exists, "size_mb": size_mb})
    st.dataframe(pd.DataFrame(rows))

# ------------------- Action buttons -------------------
col1, col2 = st.columns(2)
with col1:
    run_baseline = st.button("Run ML Picks", type="primary", key="btn_baseline")
with col2:
    run_models = st.button("Run Smart MC (Models)", key="btn_models")

# ------------------- Heuristic path -------------------
if run_baseline:
    if not tmp_path:
        st.warning("Please upload a lineup CSV first.")
    else:
        with st.spinner("Running ML Picks (heuristic path)…"):
            try:
                tasks = build_sim_tasks_from_smart_csv(
                    tmp_path,
                    cap_mode_ui=cap_mode,
                    default_pitch_cap=int(pitch_cap),
                    hook_aggr=float(hook_aggr),
                    use_heads=bool(use_heads),
                    strict_model_only=bool(strict_model_only),
                )
                results_df = make_picks(tasks)
                if isinstance(results_df, pd.DataFrame) and len(results_df) > 0:
                    st.success(f"Generated {len(results_df)} pitcher rows")
                    show_cols = [c for c in results_df.columns if c != "meta"]
                    st.dataframe(results_df[show_cols], use_container_width=True, height=520)
                    st.download_button(
                        "Download CSV",
                        data=results_df.to_csv(index=False).encode(),
                        file_name="mlbskynet_picks.csv",
                        mime="text/csv",
                        key="dl_baseline",
                    )
                else:
                    st.warning("No results returned (check that SPs were detected).")
            except Exception as e:
                st.error(f"Failed: {type(e).__name__}: {e}")
                st.code("".join(traceback.format_exc()), language="text")

# ------------------- Smart MC (model-driven) -------------------
if run_models:
    if not tmp_path:
        st.warning("Please upload a lineup CSV first.")
    else:
        with st.spinner("Running Smart Monte Carlo (models)…"):
            try:
                tasks = build_sim_tasks_from_smart_csv(
                    tmp_path,
                    cap_mode_ui=cap_mode,
                    default_pitch_cap=int(pitch_cap),
                    hook_aggr=float(hook_aggr),
                    use_heads=bool(use_heads),
                    strict_model_only=bool(strict_model_only),
                )

                with st.expander("Resolved per-pitcher inputs", expanded=False):
                    st.dataframe(pd.DataFrame(st.session_state.get("_resolved_inputs_preview", [])),
                                 use_container_width=True, height=260)

                results_df = make_picks_smart_mc(
                    tasks=tasks,
                    n_sims=int(num_sims),
                    n_jobs=4,
                    rng_seed=42,
                    max_pas_per_game=50,
                    per_game_timeout_s=2.5,
                    default_pitch_cap=int(pitch_cap),  # fallback only if a task lacks pitch_cap
                    cap_mode=CAPMODE_MAP[cap_mode],
                )

                if not isinstance(results_df, pd.DataFrame) or len(results_df) == 0:
                    st.warning("No results returned.")
                else:
                    aug = augment_with_edges(results_df)
                    st.success(f"Generated {len(aug)} pitcher rows")
                    show_cols = [c for c in aug.columns if c != "meta"]
                    st.dataframe(aug[show_cols], use_container_width=True, height=520)

                    st.download_button(
                        "Download CSV (with edges)",
                        data=aug.to_csv(index=False).encode(),
                        file_name="mlbskynet_smart_mc_with_edges.csv",
                        mime="text/csv",
                        key="dl_models_edges",
                    )

                    overs, unders = rank_strikeout_edges(aug, top_n=10)
                    tab1, tab2 = st.tabs(["Top Over candidates", "Top Under candidates"])
                    with tab1:
                        if len(overs) > 0:
                            st.dataframe(overs, use_container_width=True, height=360)
                        else:
                            st.info("No strong Over edges.")
                    with tab2:
                        if len(unders) > 0:
                            st.dataframe(unders, use_container_width=True, height=360)
                        else:
                            st.info("No strong Under edges.")
            except Exception as e:
                st.error(f"Failed: {type(e).__name__}: {e}")
                st.code("".join(traceback.format_exc()), language="text")

# ------------------- Footnotes -------------------
st.caption(
    "Place team CSVs in models/: "
    "2025_team_batter_stats.csv, 2025_team_batting_vs_LHP.csv, 2025_team_batting_vs_RHP.csv. "
    "The app resolves batter rates via player-first fallbacks, then team splits/season."
)
