# src/build_team_batting_profiles.py
"""
Utility to validate and prepare team batting reference tables.

- Reads three CSVs placed in models/: season backbone + vs LHP + vs RHP (as-is).
- Converts Fangraphs percentage columns to decimals where present.
- Writes a compact preview CSV for sanity checks (models/team_batting_preview.csv).
- Does NOT merge into a giant table; simulator uses runtime lookups via src.fallbacks.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

DEFAULT_MODELS_DIR = Path("models")
SEASON_CSV = DEFAULT_MODELS_DIR / "2025_team_batter_stats.csv"
LHP_CSV    = DEFAULT_MODELS_DIR / "2025_team_batting_vs_LHP.csv"
RHP_CSV    = DEFAULT_MODELS_DIR / "2025_team_batting_vs_RHP.csv"
PREVIEW_OUT = DEFAULT_MODELS_DIR / "team_batting_preview.csv"

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

def export_preview(season: pd.DataFrame, lhp: pd.DataFrame, rhp: pd.DataFrame, out_path: Path) -> None:
    base_keep = [c for c in ["team","year","pa","bb","so","hr","ba","obp","slg","woba"] if c in season.columns]
    base = season[base_keep] if base_keep else season[["team","year"]].copy()

    def pick(df, cols):
        cols = [c for c in cols if c in df.columns]
        return df[["team","year"] + cols] if cols else df[["team","year"]]

    l_cols = ["PA","BB","SO","HR","K%","BB%"]
    r_cols = ["PA","BB","SO","HR","K%","BB%"]

    lhp_small = pick(lhp, l_cols).rename(columns={
        "PA":"pa_vs_lhp","BB":"bb_vs_lhp","SO":"so_vs_lhp","HR":"hr_vs_lhp","K%":"k_rate_vs_lhp","BB%":"bb_rate_vs_lhp"
    })
    rhp_small = pick(rhp, r_cols).rename(columns={
        "PA":"pa_vs_rhp","BB":"bb_vs_rhp","SO":"so_vs_rhp","HR":"hr_vs_rhp","K%":"k_rate_vs_rhp","BB%":"bb_rate_vs_rhp"
    })

    preview = base.merge(lhp_small, on=["team","year"], how="left").merge(rhp_small, on=["team","year"], how="left")
    preview.to_csv(out_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Directory containing the team CSVs.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    season = _load_csv(models_dir / SEASON_CSV.name)
    lhp    = _load_csv(models_dir / LHP_CSV.name)
    rhp    = _load_csv(models_dir / RHP_CSV.name)

    # Convert Fangraphs percentage columns to decimals
    _pct_to_dec(lhp, ["BB%", "K%", "LD%", "GB%", "FB%", "Pull%", "Oppo%"])
    _pct_to_dec(rhp, ["BB%", "K%", "LD%", "GB%", "FB%", "Pull%", "Oppo%"])

    # Basic validations
    print(f"Loaded season rows: {len(season)} | cols: {len(season.columns)}")
    print(f"Loaded vs LHP rows: {len(lhp)} | cols: {len(lhp.columns)}")
    print(f"Loaded vs RHP rows: {len(rhp)} | cols: {len(rhp.columns)}")
    missing = []
    for team in season["team"].unique():
        if not ((lhp["team"] == team).any() and (rhp["team"] == team).any()):
            missing.append(team)
    if missing:
        print("⚠️ Teams missing in split files:", ", ".join(sorted(set(missing))))

    models_dir.mkdir(parents=True, exist_ok=True)
    export_preview(season, lhp, rhp, models_dir / PREVIEW_OUT.name)
    print(f"Preview saved to {models_dir / PREVIEW_OUT.name}")

if __name__ == "__main__":
    main()
