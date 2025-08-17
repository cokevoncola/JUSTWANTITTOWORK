# src/daily_update.py
from __future__ import annotations
import argparse, os
import pandas as pd
from typing import Optional
# reuse helpers from build_profiles
from .build_profiles import build_profiles, norm_cols

MASTER_DEFAULT = "data/statcast_master.csv.gz"

def _make_key(df: pd.DataFrame) -> pd.Series:
    """Best-effort unique pitch key: sv_id if present, else game_pk+pitch_number, else fallback."""
    if "sv_id" in df.columns:
        return df["sv_id"].astype(str)
    have_gp = "game_pk" in df.columns
    have_no = "pitch_number" in df.columns
    if have_gp and have_no:
        return df["game_pk"].astype(str) + "_" + df["pitch_number"].astype(str)
    # last resort (not perfect, but prevents most dupes)
    cols = [c for c in ["game_date","pitcher","batter","pitch_number"] if c in df.columns]
    if cols:
        return df[cols].astype(str).agg("_".join, axis=1)
    return pd.util.hash_pandas_object(df, index=False).astype(str)

def merge_master(new_csv: str, master_path: str = MASTER_DEFAULT) -> str:
    os.makedirs(os.path.dirname(master_path), exist_ok=True)

    # load new
    new = pd.read_csv(new_csv, low_memory=False)
    new = norm_cols(new)
    new["_key"] = _make_key(new)

    # load existing master (if any) and drop dupes
    if os.path.exists(master_path):
        master = pd.read_csv(master_path, compression="infer", low_memory=False)
        master = norm_cols(master)
        have = set(master.get("_key", pd.Series([], dtype=str)))
        new = new[~new["_key"].isin(have)]
        out = pd.concat([master, new], ignore_index=True, sort=False)
    else:
        out = new

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        out = out.sort_values("game_date")

    out.to_csv(master_path, index=False, compression="gzip")
    print(f"✅ Master updated: {master_path}  (+{len(new)} new rows)")
    return master_path

def main():
    ap = argparse.ArgumentParser(description="Append daily Statcast CSV and rebuild profiles.")
    ap.add_argument("--new", required=True, help="Path to NEW daily Statcast CSV")
    ap.add_argument("--master", default=MASTER_DEFAULT, help="Path to master CSV.GZ")
    ap.add_argument("--out", default="models/", help="Output dir for profiles")
    args = ap.parse_args()

    master_path = merge_master(args.new, args.master)
    # Rebuild all profiles off the master (streaming read; gz is fine)
    build_profiles(master_path, args.out)

if __name__ == "__main__":
    main()
