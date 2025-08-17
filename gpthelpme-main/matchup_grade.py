# src/matchup_grade.py
from __future__ import annotations
import os, re, argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------------------- config ---------------------------------
ROLL_PREF = [15, 30, 5]     # try 15g first, fallback to 30g, then 5g
LEAGUE_K = 0.22             # league-average K% per PA baseline for grading
MIN_LINEUP_SIZE = 7         # require at least this many batters to grade
# weights for a simple K% estimator (can tune later)
W_PIT_SWSTR = 0.60
W_BAT_CONTACT = 0.40

# ------------------------------ utils -----------------------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: re.sub(r"[^a-z0-9_]+","_", c.strip().lower()) for c in df.columns})

def read_profiles(models_dir: str):
    ppath = Path(models_dir) / "pitcher_profiles.parquet"
    bpath = Path(models_dir) / "batter_profiles.parquet"
    tpath = Path(models_dir) / "tendency_tables.json"
    nmap  = Path(models_dir) / "id_to_name.json"

    if not ppath.exists() or not bpath.exists():
        raise FileNotFoundError(f"Missing profiles under {models_dir}. Expected pitcher_profiles.parquet and batter_profiles.parquet")

    pit = pd.read_parquet(ppath)
    bat = pd.read_parquet(bpath)
    tend = json.load(open(tpath)) if tpath.exists() else {}
    id2name = json.load(open(nmap)) if nmap.exists() else {}

    # keep only needed id + metrics
    return pit, bat, tend, id2name

def _first_non_nan(values: List[Optional[float]]) -> Optional[float]:
    for v in values:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return None

def _col_if_exists(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    return df[col] if col in df.columns else None

def pick_metric_row(row: pd.Series, prefix: str, labels: List[str]) -> Optional[float]:
    """
    From a single row (pitcher or batter profile), try columns like:
        f"{prefix}_{label}" in order; fallback to plain f"{prefix}".
    Returns float or None.
    """
    for lab in labels:
        col = f"{prefix}_{lab}"
        if col in row and pd.notna(row[col]):
            return float(row[col])
    # fallback to non-split
    if prefix in row and pd.notna(row[prefix]):
        return float(row[prefix])
    return None

def choose_pitcher_metrics(p_row: pd.Series, vs_bat_hand: str, p_home_away: str) -> Dict[str, Optional[float]]:
    # labels we created in build_profiles: vsL / vsR / vsBOTH, home/away, vsL_home, vsR_away, etc.
    combos = [f"vs{vs_bat_hand}_{p_home_away}", f"vs{vs_bat_hand}", p_home_away]
    out = {
        "swstr": pick_metric_row(p_row, "swstr_rw15", combos) or \
                 pick_metric_row(p_row, "swstr_rw30", combos) or \
                 pick_metric_row(p_row, "swstr_rw5", combos) or \
                 pick_metric_row(p_row, "swstr_season", combos),

        "csw":   pick_metric_row(p_row, "csw_rw15", combos) or \
                 pick_metric_row(p_row, "csw_rw30", combos) or \
                 pick_metric_row(p_row, "csw_rw5", combos) or \
                 pick_metric_row(p_row, "csw_season", combos),

        "zone":  pick_metric_row(p_row, "zone_rw15", combos) or \
                 pick_metric_row(p_row, "zone_rw30", combos) or \
                 pick_metric_row(p_row, "zone_rw5", combos) or \
                 pick_metric_row(p_row, "zone_season", combos),
    }
    return out

def choose_batter_metrics(b_row: pd.Series, vs_pitch_hand: str, b_home_away: str) -> Dict[str, Optional[float]]:
    combos = [f"vs{vs_pitch_hand}_{b_home_away}", f"vs{vs_pitch_hand}", b_home_away]
    out = {
        "contact": pick_metric_row(b_row, "contact_rw15", combos) or \
                   pick_metric_row(b_row, "contact_rw30", combos) or \
                   pick_metric_row(b_row, "contact_rw5", combos) or \
                   pick_metric_row(b_row, "contact_season", combos),

        "swstr":   pick_metric_row(b_row, "swstr_rw15", combos) or \
                   pick_metric_row(b_row, "swstr_rw30", combos) or \
                   pick_metric_row(b_row, "swstr_rw5", combos) or \
                   pick_metric_row(b_row, "swstr_season", combos),
    }
    return out

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def estimate_k_per_pa(pit_swstr: Optional[float], bat_contact: Optional[float], pit_csw: Optional[float]) -> float:
    """
    Very fast, transparent K% estimator. Blend pitcher miss-induction with batter whiff/contact.
    If csw is available, it helps lift K% when called strikes are strong.
    """
    # sensible fallbacks
    ps = pit_swstr if pit_swstr is not None else 0.105   # league-ish swstr
    bc = bat_contact if bat_contact is not None else 0.76 # league-ish contact on swings
    csw = pit_csw if pit_csw is not None else 0.285      # league-ish CSW

    # base driver: pitcher whiffs vs batter (lack of) contact
    base = W_PIT_SWSTR * ps + W_BAT_CONTACT * (1.0 - bc)
    # add a small CSW bump (called strikes matter for Ks)
    adj  = base + 0.35 * (csw - 0.28)
    return float(np.clip(adj, 0.05, 0.50))  # sane bounds for per-PA K probability

def letter_grade(exp_k: float, league_k: float = LEAGUE_K) -> str:
    """
    Grade by delta over league K% per PA. Tunable thresholds.
    """
    delta = exp_k - league_k
    if   delta >= 0.08: return "A+"
    elif delta >= 0.06: return "A"
    elif delta >= 0.04: return "A-"
    elif delta >= 0.02: return "B"
    elif delta >= -0.02: return "C"
    elif delta >= -0.04: return "D"
    else: return "F"

# --------------------------- lineup parsing --------------------------------
def read_lineups_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = norm_cols(df)
    # normalize expected columns
    ren = {
        "team_code":"team",
        "mlb_id":"mlb_id",
        "player_name":"player_name",
        "batting_order":"batting_order",
        "home_away":"home_away",
        "position":"position",
        "game_date":"game_date",
        "game_number":"game_number",
        "weather":"weather",
    }
    # be robust to small spelling/casing diffs
    for k in list(ren.keys()):
        if k not in df.columns:
            # try to find similar
            for c in df.columns:
                if c.replace("_","") == k.replace("_",""):
                    ren[k] = c
                    break
    # rename only those that exist
    m = {v:k for k,v in ren.items() if v in df.columns}
    df = df.rename(columns=m)

    # coerce types
    if "mlb_id" in df.columns:
        df["mlb_id"] = pd.to_numeric(df["mlb_id"], errors="coerce").astype("Int64")
    if "batting_order" in df.columns:
        df["batting_order"] = df["batting_order"].astype(str)

    return df

def extract_matchups(df: pd.DataFrame) -> List[Dict[str,Any]]:
    """
    Return list of matchups:
      { game_key, pitcher_id, pitcher_side, pitcher_home_away, lineup_batters (list of ids),
        lineup_side ('home'/'away'), pitcher_name? }
    """
    out = []
    if df.empty: return out
    df = df.copy()

    # derive a game key
    for col in ("game_date","game_number"):
        if col not in df.columns:
            df[col] = ""  # keep simple keys if missing

    # identify SP rows (batting_order == 'SP' or position == 'SP')
    is_sp = (df.get("batting_order","").str.upper() == "SP") | (df.get("position","").str.upper() == "SP")
    sp_df = df[is_sp].copy()

    # lineup rows: batting_order 1..9
    is_bat = df.get("batting_order","").str.fullmatch(r"\d+")
    bats_df = df[is_bat].copy()

    # group by game + side
    for (gdate, gnum, side), g in df.groupby([df.get("game_date",""), df.get("game_number",""), df.get("home_away","")]):
        # pitcher for that side
        sp = sp_df[(sp_df.get("game_date","")==gdate) & (sp_df.get("game_number","")==gnum) & (sp_df.get("home_away","")==side)]
        if sp.empty: 
            continue
        pid = int(sp["mlb_id"].iloc[0])
        pname = sp.get("player_name", pd.Series([None])).iloc[0]

        # batter list = the opposite side for that same game
        opp = "home" if side == "away" else "away"
        opp_bats = bats_df[(bats_df.get("game_date","")==gdate) & (bats_df.get("game_number","")==gnum) & (bats_df.get("home_away","")==opp)]
        blist = [int(x) for x in opp_bats["mlb_id"].dropna().astype(int).tolist()]

        out.append({
            "game_key": f"{gdate}_{gnum}",
            "pitcher_id": pid,
            "pitcher_name": pname,
            "pitcher_home_away": side,
            "lineup_side": opp,
            "batters": blist
        })
    return out

# --------------------------- grading engine --------------------------------
def build_batter_lookup(bat_prof: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    # minimal info we need: last known stand (L/R/S), and all split columns for metrics
    keep_cols = ["batter", "stand"]
    keep_cols += [c for c in bat_prof.columns if c.startswith(("contact_rw","contact_season","swstr_rw","swstr_season"))]
    bp = bat_prof[keep_cols].copy()
    bp = bp.sort_values("batter").drop_duplicates("batter", keep="last")
    d = {}
    for _, r in bp.iterrows():
        b = int(r["batter"])
        d[b] = r.to_dict()
    return d

def build_pitcher_lookup(pit_prof: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    keep_cols = ["pitcher","p_throws"]
    keep_cols += [c for c in pit_prof.columns if c.startswith(("swstr_rw","swstr_season","csw_rw","csw_season","zone_rw","zone_season"))]
    pp = pit_prof[keep_cols].copy()
    pp = pp.sort_values("pitcher").drop_duplicates("pitcher", keep="last")
    d = {}
    for _, r in pp.iterrows():
        d[int(r["pitcher"])] = r.to_dict()
    return d

def grade_matchup(
    pitcher_id: int,
    batter_ids: List[int],
    pitcher_home_away: str,
    pitLUT: Dict[int, Dict[str,Any]],
    batLUT: Dict[int, Dict[str,Any]],
    id_to_name: Dict[str,str],
    league_k: float = LEAGUE_K
) -> Dict[str,Any]:
    p = pitLUT.get(pitcher_id)
    if not p or not batter_ids:
        return {"pitcher_id": pitcher_id, "grade": "NA", "exp_k_pa": np.nan, "n_batters": len(batter_ids)}

    p_hand = str(p.get("p_throws","R")).upper()[:1] or "R"

    # per-batter K% estimate
    ks = []
    details = []
    for bid in batter_ids:
        b = batLUT.get(bid)
        if not b: 
            continue
        # batter stance to choose pitcher's vs_bat_hand split
        b_stand = str(b.get("stand","R")).upper()[:1] or "R"
        vs_bat_hand = "BOTH" if b_stand == "S" else b_stand
        # batter split label depends on pitcher's hand, and batter home/away = opposite of pitcher's H/A
        b_home_away = "home" if pitcher_home_away == "away" else "away"

        pit_metrics = choose_pitcher_metrics(pd.Series(p), vs_bat_hand, pitcher_home_away)
        bat_metrics = choose_batter_metrics(pd.Series(b), p_hand, b_home_away)

        k_pa = estimate_k_per_pa(
            pit_swstr = pit_metrics["swstr"],
            bat_contact = bat_metrics["contact"],
            pit_csw = pit_metrics["csw"]
        )
        ks.append(k_pa)
        details.append((bid, k_pa, pit_metrics["swstr"], bat_metrics["contact"], pit_metrics["csw"]))

    if len(ks) < max(3, MIN_LINEUP_SIZE):
        return {"pitcher_id": pitcher_id, "grade": "NA", "exp_k_pa": np.nan, "n_batters": len(ks)}

    exp_k_pa = float(np.mean(ks))
    grade = letter_grade(exp_k_pa, league_k=league_k)

    # rough expected Ks (assume ~25 batters faced by starter; adjust later with your IP model)
    bf = 25
    exp_Ks = bf * exp_k_pa

    name = id_to_name.get(str(pitcher_id), None)
    # top 3 soft spots (highest K prob batters)
    details.sort(key=lambda x: x[1], reverse=True)
    top3 = [{"batter_id": d[0], "k_pa": round(float(d[1]),3)} for d in details[:3]]

    return {
        "pitcher_id": pitcher_id,
        "pitcher_name": name,
        "pitcher_hand": p_hand,
        "home_away": pitcher_home_away,
        "n_batters": len(ks),
        "exp_k_pa": round(exp_k_pa, 4),
        "exp_Ks": round(exp_Ks, 2),
        "grade": grade,
        "top3_targets": top3
    }

# ------------------------------ CLI ---------------------------------------
def run(lineups_csv: str, models_dir: str, out_csv: Optional[str], league_k: float = LEAGUE_K):
    pit_prof, bat_prof, tendencies, id2name = read_profiles(models_dir)
    pitLUT = build_pitcher_lookup(pit_prof)
    batLUT = build_batter_lookup(bat_prof)

    ldf = read_lineups_csv(lineups_csv)
    matchups = extract_matchups(ldf)

    rows = []
    for m in matchups:
        res = grade_matchup(
            m["pitcher_id"], m["batters"], m["pitcher_home_away"],
            pitLUT, batLUT, id2name, league_k=league_k
        )
        res["game_key"] = m["game_key"]
        rows.append(res)

    out = pd.DataFrame(rows).sort_values(["grade","exp_Ks"], ascending=[True, False])

    print("\n=== Matchup Grades ===")
    if not out.empty:
        print(out[["game_key","pitcher_name","pitcher_id","home_away","exp_k_pa","exp_Ks","grade","n_batters"]].to_string(index=False))
    else:
        print("No matchups found.")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineups", required=True, help="Lineup CSV (team code / mlb id / batting order / home/away …)")
    ap.add_argument("--models_dir", default="models/", help="Directory with pitcher_profiles.parquet, batter_profiles.parquet")
    ap.add_argument("--out", dest="out_csv", default=None, help="Optional: write grades CSV here")
    ap.add_argument("--league_k", type=float, default=LEAGUE_K, help="League K%% per PA baseline (default 0.22)")
    args = ap.parse_args()
    run(args.lineups, args.models_dir, args.out_csv, league_k=args.league_k)

if __name__ == "__main__":
    main()
