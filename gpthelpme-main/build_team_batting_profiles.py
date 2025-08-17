# src/build_team_batting_profiles.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

OUT_PATH = Path("models/team_batting_profiles.parquet")

# ---------- naming helpers ----------
REPLACERS = {
    "%": "pct",
    "/": "_",
    "-": "_",
    " ": "_",
}

CANON_RENAMES = {
    # core counting columns
    "so": "so", "bb": "bb", "pa": "pa", "ab": "ab", "h": "h",
    # slash line
    "ba": "ba", "obp": "obp", "slg": "slg",
    # quality of contact + plate discipline (we keep these)
    "woba": "woba", "wobacon": "wobacon",
    "xwoba": "xwoba", "xwobacon": "xwobacon",
    "xba": "xba", "xslg": "xslg",
    "zone_pct": "zone_pct",
    "zone_swing_pct": "zone_swing_pct",
    "zone_contact_pct": "zone_contact_pct",
    "chase_pct": "chase_pct",
    "chase_contact_pct": "chase_contact_pct",
    "whiff_pct": "whiff_pct",
    "swing_pct": "swing_pct",
    "meatball_pct": "meatball_pct",
    "meatball_swing_pct": "meatball_swing_pct",
    "gb_pct": "gb_pct", "fb_pct": "fb_pct", "ld_pct": "ld_pct", "pu_pct": "pu_pct",
    "pull_pct": "pull_pct", "straight_pct": "straight_pct", "oppo_pct": "oppo_pct",
    "weak_pct": "weak_pct", "topped_pct": "topped_pct", "under_pct": "under_pct",
    "solid_pct": "solid_pct",
    # barrels etc (note: some exports have "barrell_pct" misspelled)
    "barrels": "barrels",
    "barrel_pct": "barrel_pct",
    "barrell_pct": "barrel_pct",
    # misc EV / LA
    "exit_velocity": "exit_velocity",
    "launch_angle": "launch_angle",
    "launch_angle_sweet_spot_pct": "sweet_spot_pct",
    "flare_burner_pct": "flare_burner_pct",
    "pitches": "pitches",
    "batted_balls": "batted_balls",
    "bbe": "bbe",
    # meta
    "team": "team",
    "year": "year",
    "1st_pitch_swing_pct": "first_pitch_swing_pct",
}

def to_snake(c: str) -> str:
    c = c.strip()
    for k, v in REPLACERS.items():
        c = c.replace(k, v)
    c = re.sub(r"__+", "_", c)
    c = re.sub(r"[^0-9a-zA-Z_]+", "_", c)
    c = c.lower().strip("_")
    # special cases to harmonize
    c = c.replace("launch_angle_sweet_spot_pct", "launch_angle_sweet_spot_pct")
    c = c.replace("flare_burner_pct", "flare_burner_pct")
    return CANON_RENAMES.get(c, c)

# ---------- empirical bayes ----------
def eb_shrink(p_hat, n, prior, lam):
    p_hat = pd.Series(p_hat, dtype="float64")
    n = pd.Series(n, dtype="float64")
    prior = prior if isinstance(prior, pd.Series) else pd.Series(prior, index=p_hat.index, dtype="float64")
    w = n / (n + float(lam))
    return w * p_hat + (1.0 - w) * prior

def add_rate(df: pd.DataFrame, num: str, den: str, out: str) -> None:
    df[out] = np.where(df[den].fillna(0) > 0, df[num] / df[den], np.nan)

def build_team_batting(in_path: str, out_path: Path = OUT_PATH, use_eb: bool = True, lam: float = 600.0) -> pd.DataFrame:
    # read CSV or parquet
    p = Path(in_path)
    if not p.exists():
        raise FileNotFoundError(in_path)
    if p.suffix.lower() == ".csv":
        raw = pd.read_csv(p)
    else:
        raw = pd.read_parquet(p)

    # rename columns to snake_case and canonical names
    raw = raw.rename(columns={c: to_snake(c) for c in raw.columns})

    # make sure required columns exist (we’ll tolerate extra ones)
    required = ["team", "year", "pa", "bb", "so"]
    for r in required:
        if r not in raw.columns:
            raw[r] = np.nan

    df = raw.copy()

    # derive core rates
    add_rate(df, "so", "pa", "k_rate")
    add_rate(df, "bb", "pa", "bb_rate")

    # if you prefer everything as rate (0..1) not %, keep as-is
    # optional EB smoothing for k_rate/bb_rate by team using team PA as n
    if use_eb:
        # league priors (simple mean across teams for that year)
        dfg = df.groupby("year", dropna=False)
        league_k = dfg["k_rate"].transform("mean")
        league_bb = dfg["bb_rate"].transform("mean")

        df["k_rate_eb"] = eb_shrink(df["k_rate"], df["pa"], league_k, lam)
        df["bb_rate_eb"] = eb_shrink(df["bb_rate"], df["pa"], league_bb, lam)
    else:
        df["k_rate_eb"] = df["k_rate"]
        df["bb_rate_eb"] = df["bb_rate"]

    # keep a compact set your sim/fallback needs (you can keep more)
    keep_cols = [
        "team", "year",
        "pa", "bb", "so",
        "k_rate", "k_rate_eb", "bb_rate", "bb_rate_eb",
        "zone_pct", "zone_swing_pct", "zone_contact_pct",
        "chase_pct", "chase_contact_pct",
        "whiff_pct", "swing_pct",
        "woba", "xwoba", "xba", "xslg",
        "gb_pct", "fb_pct", "ld_pct", "pu_pct",
        "pull_pct", "straight_pct", "oppo_pct",
        "barrel_pct", "sweet_spot_pct", "solid_pct", "flare_burner_pct",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    # index by team for fast lookup later
    out = out.sort_values(["year", "team"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"✅ wrote {out_path} with {len(out)} rows and {len(out.columns)} cols")
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="CSV or parquet of team season batting")
    ap.add_argument("--out", dest="out_path", default=str(OUT_PATH))
    ap.add_argument("--no-eb", dest="use_eb", action="store_false", help="disable EB smoothing")
    ap.add_argument("--lam", dest="lam", type=float, default=600.0, help="EB prior strength (team PA pseudo-counts)")
    args = ap.parse_args()
    build_team_batting(args.in_path, Path(args.out_path), use_eb=args.use_eb, lam=args.lam)
