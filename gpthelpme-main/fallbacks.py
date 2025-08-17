# src/fallbacks.py
"""
Team-aware fallback helpers for batter rates.

We keep your CSVs exactly as-is in `models/`:
  - models/2025_team_batter_stats.csv          (season backbone; e.g., pa, bb, so, woba, etc.)
  - models/2025_team_batting_vs_LHP.csv        (Fangraphs vs LHP splits; includes K%, BB% columns)
  - models/2025_team_batting_vs_RHP.csv        (Fangraphs vs RHP splits; includes K%, BB% columns)

Functions:
  - load_team_tables()                    -> dict with DataFrames (season/lhp/rhp), indexed by (team, year)
  - resolve_batter_rate_with_fallback()   -> per-batter rate resolution using player-first -> team split -> team season
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

# Keep your CSVs exactly as-is; no renaming required.
SEASON_CSV = Path("models/2025_team_batter_stats.csv")
LHP_CSV    = Path("models/2025_team_batting_vs_LHP.csv")
RHP_CSV    = Path("models/2025_team_batting_vs_RHP.csv")

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize team/year columns if needed
    if "team" not in df.columns:
        for c in ("Team", "Tm", "club"):
            if c in df.columns:
                df = df.rename(columns={c: "team"})
                break
    if "year" not in df.columns:
        if "Season" in df.columns:
            df = df.rename(columns={"Season": "year"})
        else:
            df["year"] = 2025
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2025).astype(int)
    return df

def _pct_to_dec(df: pd.DataFrame, cols):
    """Convert % columns (e.g., Fangraphs 'K%') to decimals (0-1)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

def load_team_tables() -> Dict[str, pd.DataFrame]:
    """
    Returns dict with:
      - season: Statcast-style season backbone (indexed by team,year)
      - lhp: Fangraphs vs LHP splits (K%, BB% converted to decimals)
      - rhp: Fangraphs vs RHP splits (K%, BB% converted to decimals)
    """
    season = _load_csv(SEASON_CSV)
    lhp    = _load_csv(LHP_CSV)
    rhp    = _load_csv(RHP_CSV)

    # Convert Fangraphs % to decimals if present
    _pct_to_dec(lhp, ["BB%", "K%", "LD%", "GB%", "FB%", "Pull%", "Oppo%"])
    _pct_to_dec(rhp, ["BB%", "K%", "LD%", "GB%", "FB%", "Pull%", "Oppo%"])

    # Index for fast lookups
    season = season.set_index(["team","year"]).sort_index()
    lhp    = lhp.set_index(["team","year"]).sort_index()
    rhp    = rhp.set_index(["team","year"]).sort_index()

    return {"season": season, "lhp": lhp, "rhp": rhp}

def team_rate(
    tables: Dict[str,pd.DataFrame], team: str, year: int,
    stat: str, split: Optional[str]
) -> Optional[float]:
    """
    Compute team-level rate for stat in {'k_rate','bb_rate'}.
    split: None | 'vs_lhp' | 'vs_rhp'

    Uses Fangraphs split K%/BB% when available; otherwise falls back to counts/PA.
    Falls back to season backbone for no-split case (expects 'so','bb','pa' columns).
    """
    team = str(team).upper().strip()
    idx = (team, int(year))
    season = tables["season"]
    lhp, rhp = tables["lhp"], tables["rhp"]

    if split in ("vs_lhp", "vs_rhp"):
        df = lhp if split == "vs_lhp" else rhp
        pct_col = {"k_rate": "K%", "bb_rate": "BB%"}[stat]
        if pct_col in df.columns and idx in df.index:
            val = df.at[idx, pct_col]
            return None if pd.isna(val) else float(val)

        # fallback using counts
        num_col = {"k_rate": "SO", "bb_rate": "BB"}[stat]
        if {"PA", num_col}.issubset(df.columns) and idx in df.index:
            pa = float(df.at[idx, "PA"])
            if pa > 0:
                return float(df.at[idx, num_col]) / pa

    # Season backbone (Statcast-style)
    if stat == "k_rate" and {"so","pa"}.issubset(season.columns) and idx in season.index:
        pa = float(season.at[idx, "pa"])
        return float(season.at[idx, "so"]) / pa if pa > 0 else None
    if stat == "bb_rate" and {"bb","pa"}.issubset(season.columns) and idx in season.index:
        pa = float(season.at[idx, "pa"])
        return float(season.at[idx, "bb"]) / pa if pa > 0 else None
    return None

def resolve_batter_rate_with_fallback(
    batter: Dict[str, float],
    *, tables: Dict[str,pd.DataFrame],
    team: str, year: int, stat: str, pitcher_hand: str
) -> Optional[float]:
    """
    Player-first resolution:
      1) player split (e.g. batter['k_rate_vs_LHP'] if pitcher throws L)
      2) player season (e.g. batter['k_rate'])
      3) team split (preferred vs pitcher hand)
      4) team season
    """
    hand = (pitcher_hand or "R").upper()
    split = "vs_lhp" if hand == "L" else "vs_rhp"

    # 1) player split
    ps_key = f"{stat}_vs_LHP" if hand == "L" else f"{stat}_vs_RHP"
    v = batter.get(ps_key)
    if v is not None and not pd.isna(v):
        return float(v)

    # 2) player season
    v = batter.get(stat)
    if v is not None and not pd.isna(v):
        return float(v)

    # 3) team split
    v = team_rate(tables, team, year, stat=stat, split=split)
    if v is not None:
        return float(v)

    # 4) team season
    v = team_rate(tables, team, year, stat=stat, split=None)
    return None if v is None else float(v)

