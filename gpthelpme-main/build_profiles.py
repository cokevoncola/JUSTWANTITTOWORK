# src/build_profiles.py
from __future__ import annotations
import os, json, re, argparse, warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------ config ------------------------------
ROLL_WINDOWS: List[int] = [5, 15, 30]      # rolling windows (games)
CHUNK = 500_000
OUT_DIR_DEFAULT = "models"
MIN_PITCHES_SEASON_P = 200                 # pitcher season threshold
MIN_PITCHES_SEASON_B = 150                 # batter season threshold

# Empirical-Bayes pseudo counts (shrink toward league priors)
PRIOR_S = {"rate_per_pitch": 200, "rate_per_swing": 80, "rate_per_bip": 80}

NEEDED = [
    "pitch_type","pitch_name","game_date","player_name",
    "batter","pitcher","events","description","zone",
    "stand","p_throws","type","bb_type","balls","strikes",
    "plate_x","plate_z","on_3b","on_2b","on_1b","outs_when_up",
    "inning","inning_topbot","pitch_number","n_thruorder_pitcher",
    "home_team","away_team","launch_speed","launch_angle",
]

# ------------------------------ small utils ------------------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: re.sub(r"[^a-z0-9_]+","_", c.strip().lower()) for c in df.columns}
    return df.rename(columns=m)

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def short_pitch(x) -> str:
    s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    s = s.upper()
    s = (s.replace("FOUR-SEAM","FF").replace("4-SEAM","FF").replace("FOUR SEAM","FF")
           .replace("SINKER","SI").replace("TWO-SEAM","SI").replace("2-SEAM","SI")
           .replace("CUTTER","FC").replace("CHANGEUP","CH").replace("CHANGE UP","CH")
           .replace("SLIDER","SL").replace("CURVEBALL","CU").replace("KNUCKLE CURVE","KC")
           .replace("SPLIT-FINGER","FS").replace("SPLITTER","FS"))
    s = re.sub(r"[^A-Z]","", s)
    if s in ("FF","SI","SL","CH","CU","FC","FS","KC"): return s
    return (s[:2] or "UN")

def count_bucket(b: int, s: int) -> str:
    try: return f"{int(b)}-{int(s)}"
    except: return "NA"

def base_state_series(df: pd.DataFrame) -> pd.Series:
    on1 = pd.to_numeric(df.get("on_1b", 0), errors="coerce").fillna(0)
    on2 = pd.to_numeric(df.get("on_2b", 0), errors="coerce").fillna(0)
    on3 = pd.to_numeric(df.get("on_3b", 0), errors="coerce").fillna(0)
    men = (on1.ne(0) | on2.ne(0) | on3.ne(0))
    return np.where(men, "men_on", "empty")

def is_swing(desc: str) -> int:
    d = (str(desc) or "").lower()
    return int(any(k in d for k in ("foul","swing","in_play")))

def is_whiff(desc: str) -> int:
    d = (str(desc) or "").lower()
    return int(any(k in d for k in ("swinging_strike","swinging","foul_tip","blocked")))

def is_called_strike(desc: str) -> int:
    d = (str(desc) or "").lower()
    return int("called_strike" in d or ("called" in d and "strike" in d))

def bip_shape(row) -> str:
    bt = str(row.get("bb_type") or "").lower()
    if bt in ("ground_ball","groundball","gb"): return "GB"
    if bt in ("fly_ball","flyball","fb"):       return "FB"
    if bt in ("line_drive","linedrive","ld"):   return "LD"
    if bt in ("popup","pop_up","pu"):           return "PU"
    la = row.get("launch_angle")
    try: la = float(la)
    except: return "NA"
    if np.isnan(la): return "NA"
    if la < 10:  return "GB"
    if la < 25:  return "LD"
    if la < 50:  return "FB"
    return "PU"

def in_zone(zone) -> int:
    try:
        z = int(zone)
        return int(1 <= z <= 9)
    except:
        return 0

def shrink_hat(k, n, prior_p, prior_n) -> float:
    n = float(max(0.0, n))
    prior_n = float(prior_n)
    if n <= 0: return float(prior_p)
    return float((k + prior_p * prior_n) / (n + prior_n))

def shrink_series(k, n, prior_p, prior_n):
    # vectorized EB shrink; k or n may be scalars or Series
    k = pd.to_numeric(k, errors="coerce")
    n = pd.to_numeric(n, errors="coerce")
    if isinstance(k, pd.Series): k = k.fillna(0.0)
    else: k = 0.0 if pd.isna(k) else float(k)
    if isinstance(n, pd.Series): n = n.fillna(0.0)
    else: n = 0.0 if pd.isna(n) else float(n)
    prior_p = float(prior_p); prior_n = float(prior_n)
    return (k + prior_p * prior_n) / (n + prior_n)

# ------------------------------ streaming aggregation ------------------------------
def stream_aggregate(stat_path: str):
    head = pd.read_csv(stat_path, nrows=0)
    usecols = [c for c in head.columns if re.sub(r"[^a-z0-9_]+","_", c.strip().lower()) in set(NEEDED)]
    chunks = pd.read_csv(stat_path, usecols=usecols, chunksize=CHUNK)

    pit_daily_parts, bat_daily_parts = [], []
    mix_parts, ptype_parts = [], []
    pit_split_daily_parts, bat_split_daily_parts = [], []
    pit_split_parts, bat_split_parts = [], []
    id_to_name: Dict[str,str] = {}

    for ch in chunks:
        df = norm_cols(ch)
        # coerce numerics
        for c in ("balls","strikes","pitch_number","n_thruorder_pitcher","outs_when_up"):
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        df["game_date"] = to_dt(df["game_date"])
        df = df.dropna(subset=["game_date","pitcher","batter"])

        # id -> name
        if "player_name" in df.columns:
            tmp = df[["pitcher","player_name"]].dropna().drop_duplicates()
            for _, r in tmp.iterrows():
                id_to_name[str(int(r["pitcher"]))] = str(r["player_name"])

        # derived
        pname = df["pitch_name"] if "pitch_name" in df.columns else df.get("pitch_type", pd.Series(index=df.index))
        df["pitch_lbl"] = pname.map(short_pitch)
        df["cnt"] = [count_bucket(b,s) for b,s in zip(df["balls"], df["strikes"])]
        df["bas"] = base_state_series(df)
        dsc = df["description"].astype(str)
        df["swing"]   = dsc.map(is_swing)
        df["whiff"]   = dsc.map(is_whiff)
        df["cstrike"] = dsc.map(is_called_strike)
        df["inzone"]  = df["zone"].map(in_zone)
        df["contact"] = df["swing"] - df["whiff"]
        df["shape"]   = df.apply(bip_shape, axis=1)
        df["bip"]     = df["shape"].isin(["GB","FB","LD","PU"]).astype(int)
        df["hardhit"] = (pd.to_numeric(df["launch_speed"], errors="coerce") >= 95).fillna(0).astype(int)

        # home/away from inning half
        itb = df["inning_topbot"].astype(str).str.upper()
        df["p_home_away"] = np.where(itb.eq("TOP"), "home", "away")
        df["b_home_away"] = np.where(itb.eq("TOP"), "away", "home")

        # handedness labels
        df["vs_bat_hand"]   = df["stand"].astype(str).str.upper().str[:1]
        df["vs_pitch_hand"] = df["p_throws"].astype(str).str.upper().str[:1]

        # pitcher daily
        pit = df.groupby(["pitcher","game_date"], as_index=False).agg(
            p_throws=("p_throws","last"),
            pitches=("pitcher","size"),
            swings=("swing","sum"),
            whiffs=("whiff","sum"),
            cstrikes=("cstrike","sum"),
            inzone=("inzone","sum"),
            contact=("contact","sum"),
            bip=("bip","sum"),
            gb=("shape", lambda s: (s=="GB").sum()),
            fb=("shape", lambda s: (s=="FB").sum()),
            ld=("shape", lambda s: (s=="LD").sum()),
            pu=("shape", lambda s: (s=="PU").sum()),
            hard=("hardhit","sum"),
        )
        pit_daily_parts.append(pit)

        # batter daily
        bat = df.groupby(["batter","game_date"], as_index=False).agg(
            stand=("stand","last"),
            pitches=("batter","size"),
            swings=("swing","sum"),
            whiffs=("whiff","sum"),
            contact=("contact","sum"),
            bip=("bip","sum"),
            gb=("shape", lambda s: (s=="GB").sum()),
            fb=("shape", lambda s: (s=="FB").sum()),
            ld=("shape", lambda s: (s=="LD").sum()),
            pu=("shape", lambda s: (s=="PU").sum()),
            hard=("hardhit","sum"),
        )
        bat_daily_parts.append(bat)

        # pitcher count mix (for sim pitch-selection)
        mix = df.groupby(["pitcher","cnt","bas","pitch_lbl"], as_index=False).size().rename(columns={"size":"n"})
        mix_parts.append(mix)

        # pitcher pitch outcomes (season level)
        ptype = df.groupby(["pitcher","pitch_lbl"], as_index=False).agg(
            swings=("swing","sum"),
            whiffs=("whiff","sum"),
            bip=("bip","sum"),
            gb=("shape", lambda s: (s=="GB").sum()),
            fb=("shape", lambda s: (s=="FB").sum()),
            ld=("shape", lambda s: (s=="LD").sum()),
            pu=("shape", lambda s: (s=="PU").sum()),
        )
        ptype_parts.append(ptype)

        # SPLITS season + daily
        pit_split_season = df.groupby(["pitcher","vs_bat_hand","p_home_away"], as_index=False).agg(
            pitches=("pitcher","size"),
            swings=("swing","sum"),
            whiffs=("whiff","sum"),
            cstrikes=("cstrike","sum"),
            inzone=("inzone","sum"),
            contact=("contact","sum"),
            bip=("bip","sum"),
            gb=("shape", lambda s: (s=="GB").sum()),
            hard=("hardhit","sum"),
        )
        pit_split_parts.append(pit_split_season)

        pit_split_daily = df.groupby(["pitcher","game_date","vs_bat_hand","p_home_away"], as_index=False).agg(
            pitches=("pitcher","size"),
            swings=("swing","sum"),
            whiffs=("whiff","sum"),
            cstrikes=("cstrike","sum"),
            inzone=("inzone","sum"),
            contact=("contact","sum"),
            bip=("bip","sum"),
            gb=("shape", lambda s: (s=="GB").sum()),
            hard=("hardhit","sum"),
        )
        pit_split_daily_parts.append(pit_split_daily)

        bat_split_season = df.groupby(["batter","vs_pitch_hand","b_home_away"], as_index=False).agg(
            pitches=("batter","size"),
            swings=("swing","sum"),
            whiffs=("whiff","sum"),
            contact=("contact","sum"),
            bip=("bip","sum"),
            gb=("shape", lambda s: (s=="GB").sum()),
            hard=("hardhit","sum"),
        )
        bat_split_parts.append(bat_split_season)

        bat_split_daily = df.groupby(["batter","game_date","vs_pitch_hand","b_home_away"], as_index=False).agg(
            pitches=("batter","size"),
            swings=("swing","sum"),
            whiffs=("whiff","sum"),
            contact=("contact","sum"),
            bip=("bip","sum"),
            gb=("shape", lambda s: (s=="GB").sum()),
            hard=("hardhit","sum"),
        )
        bat_split_daily_parts.append(bat_split_daily)

    # combine helpers
    def combine_sum(parts, keys, sum_cols=None):
        if not parts: return pd.DataFrame(columns=keys if sum_cols is None else keys+sum_cols)
        df = pd.concat(parts, ignore_index=True)
        if sum_cols is None:
            sum_cols = [c for c in df.columns if c not in keys]
        return df.groupby(keys, as_index=False)[sum_cols].sum()

    pitcher_daily = pd.concat(pit_daily_parts, ignore_index=True) if pit_daily_parts else pd.DataFrame()
    batter_daily  = pd.concat(bat_daily_parts, ignore_index=True) if bat_daily_parts else pd.DataFrame()

    mix_df   = combine_sum(mix_parts,   ["pitcher","cnt","bas","pitch_lbl"], ["n"]) if mix_parts else pd.DataFrame(columns=["pitcher","cnt","bas","pitch_lbl","n"])
    ptype_df = combine_sum(ptype_parts, ["pitcher","pitch_lbl"]) if ptype_parts else pd.DataFrame(columns=["pitcher","pitch_lbl","swings","whiffs","bip","gb","fb","ld","pu"])

    pit_split_df        = combine_sum(pit_split_parts,        ["pitcher","vs_bat_hand","p_home_away"])
    bat_split_df        = combine_sum(bat_split_parts,        ["batter","vs_pitch_hand","b_home_away"])
    pit_split_daily_all = pd.concat(pit_split_daily_parts, ignore_index=True) if pit_split_daily_parts else pd.DataFrame()
    bat_split_daily_all = pd.concat(bat_split_daily_parts, ignore_index=True) if bat_split_daily_parts else pd.DataFrame()

    return (pitcher_daily, batter_daily, mix_df, ptype_df,
            pit_split_df, bat_split_df, pit_split_daily_all, bat_split_daily_all, id_to_name)

# ------------------------------ rates & rollings ------------------------------
def _add_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    s = lambda a,b: np.where(b>0, a/b, np.nan)
    df["csw"]     = s(df.get("whiffs",0)+df.get("cstrikes",0), df.get("pitches",0)) if "cstrikes" in df else np.nan
    df["swstr"]   = s(df.get("whiffs",0), df.get("swings",0))
    df["zone"]    = s(df.get("inzone",0), df.get("pitches",0)) if "inzone" in df else np.nan
    df["contact"] = s(df.get("contact",0), df.get("swings",0))
    df["gb_rate"] = s(df.get("gb",0), df.get("bip",0))
    df["hardhit_rate"] = s(df.get("hard",0), df.get("bip",0))
    return df

def _add_rollings(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if df.empty: return df
    outs = []
    for _id, g in df.groupby(id_col, sort=False):
        g = g.sort_values("game_date").reset_index(drop=True)
        for w in ROLL_WINDOWS:
            g[f"rw{w}_pitches"] = g["pitches"].rolling(w, min_periods=1).sum()
            g[f"rw{w}_swings"]  = g["swings"].rolling(w, min_periods=1).sum()
            g[f"rw{w}_whiffs"]  = g["whiffs"].rolling(w, min_periods=1).sum()
            if "cstrikes" in g: g[f"rw{w}_cstrikes"]= g["cstrikes"].rolling(w, min_periods=1).sum()
            if "inzone" in g:   g[f"rw{w}_inzone"]  = g["inzone"].rolling(w, min_periods=1).sum()
            g[f"rw{w}_contact"] = g["contact"].rolling(w, min_periods=1).sum()
            g[f"rw{w}_bip"]     = g["bip"].rolling(w, min_periods=1).sum()
            g[f"rw{w}_gb"]      = g["gb"].rolling(w, min_periods=1).sum()
            g[f"rw{w}_hard"]    = g["hard"].rolling(w, min_periods=1).sum()

            def r(n,d): return np.where(d>0, n/d, np.nan)
            g[f"csw_rw{w}"]     = r(g.get(f"rw{w}_whiffs",0) + g.get(f"rw{w}_cstrikes",0), g.get(f"rw{w}_pitches",0)) if "cstrikes" in g else np.nan
            g[f"swstr_rw{w}"]   = r(g.get(f"rw{w}_whiffs",0), g.get(f"rw{w}_swings",0))
            g[f"zone_rw{w}"]    = r(g.get(f"rw{w}_inzone",0), g.get(f"rw{w}_pitches",0)) if "inzone" in g else np.nan
            g[f"contact_rw{w}"] = r(g.get(f"rw{w}_contact",0), g.get(f"rw{w}_swings",0))
            g[f"gb_rw{w}"]      = r(g.get(f"rw{w}_gb",0), g.get(f"rw{w}_bip",0))
            g[f"hard_rw{w}"]    = r(g.get(f"rw{w}_hard",0), g.get(f"rw{w}_bip",0))
        outs.append(g)
    return pd.concat(outs, ignore_index=True)

def _league_priors(pit_daily: pd.DataFrame) -> Dict[str,float]:
    def rate(num, den):
        n, d = pit_daily.get(num, pd.Series(dtype=float)).sum(), pit_daily.get(den, pd.Series(dtype=float)).sum()
        return float(n) / float(d) if d>0 else 0.0
    pri = {
        "csw":    rate("whiffs","pitches") + rate("cstrikes","pitches"),
        "swstr":  rate("whiffs","swings"),
        "zone":   rate("inzone","pitches"),
        "contact":1.0 - rate("whiffs","swings"),
        "gb":     (pit_daily.get("gb", pd.Series(dtype=float)).sum() / max(1, pit_daily.get("bip", pd.Series(dtype=float)).sum())),
        "hard":   (pit_daily.get("hard", pd.Series(dtype=float)).sum() / max(1, pit_daily.get("bip", pd.Series(dtype=float)).sum())),
    }
    for k,v in pri.items():
        pri[k] = float(np.clip(v, 0.05, 0.95))
    return pri

# ------------------------------ finalize profiles ------------------------------
def _finalize_profiles(df: pd.DataFrame, id_col: str, priors: Dict[str,float]) -> pd.DataFrame:
    if df.empty: return df
    last = df.sort_values(["game_date"]).groupby(id_col, as_index=False).tail(1).copy()

    if id_col == "pitcher":
        last["csw_season"]  = shrink_series(last.get("whiffs",0) + last.get("cstrikes",0), last.get("pitches",0), priors["csw"],  PRIOR_S["rate_per_pitch"])
        last["zone_season"] = shrink_series(last.get("inzone",0),                           last.get("pitches",0), priors["zone"], PRIOR_S["rate_per_pitch"])
    else:
        last["csw_season"]  = np.nan
        last["zone_season"] = np.nan

    last["swstr_season"]   = shrink_series(last.get("whiffs",0),  last.get("swings",0),  priors["swstr"],  PRIOR_S["rate_per_swing"])
    last["contact_season"] = shrink_series(last.get("contact",0), last.get("swings",0),  priors["contact"],PRIOR_S["rate_per_swing"])
    last["gb_season"]      = shrink_series(last.get("gb",0),      last.get("bip",0),     priors["gb"],     PRIOR_S["rate_per_bip"])
    last["hard_season"]    = shrink_series(last.get("hard",0),    last.get("bip",0),     priors["hard"],   PRIOR_S["rate_per_bip"])

    # keep "last game" counts explicitly
    rename_counts = {"pitches":"pitches_last","swings":"swings_last","bip":"bip_last"}
    for k,v in rename_counts.items():
        if k in last.columns: last.rename(columns={k:v}, inplace=True)

    keep = [id_col, "game_date", "pitches_last","swings_last","bip_last",
            "csw_season","swstr_season","zone_season","contact_season","gb_season","hard_season"]
    for w in ROLL_WINDOWS:
        for m in ("csw","swstr","zone","contact","gb","hard"):
            col = f"{m}_rw{w}"
            if col in last.columns:
                keep.append(col)

    hand_col = "p_throws" if id_col == "pitcher" else "stand"
    if hand_col in last.columns: keep.append(hand_col)

    return last[keep]

# ------------------------------ rolling split helpers ------------------------------
def _rolling_split_rows(split_daily: pd.DataFrame, id_col: str, label_cols: List[str], label_fmt, include_csw_zone: bool) -> pd.DataFrame:
    if split_daily.empty:
        return pd.DataFrame(columns=[id_col,"label","game_date"])
    rows = []
    for keys, g in split_daily.groupby([id_col] + label_cols, sort=False):
        if len(label_cols) == 1:
            _id, lab1 = keys; labels = {label_cols[0]: lab1}
        else:
            _id = keys[0]; labels = {label_cols[i]: keys[i+1] for i in range(len(label_cols))}
        g = g.sort_values("game_date").reset_index(drop=True)
        rec = {id_col: _id, "label": label_fmt(labels), "game_date": g.loc[len(g)-1, "game_date"]}
        for w in ROLL_WINDOWS:
            sums = {}
            for name in ("pitches","swings","whiffs","contact","bip","gb","hard"):
                sums[name] = g[name].rolling(w, min_periods=1).sum()
            if include_csw_zone:
                sums["cstrikes"] = g["cstrikes"].rolling(w, min_periods=1).sum()
                sums["inzone"]   = g["inzone"].rolling(w, min_periods=1).sum()
            i = len(g)-1
            rec[f"swstr_rw{w}"]   = shrink_hat(sums["whiffs"].iloc[i],  sums["swings"].iloc[i],  priors["swstr"],  PRIOR_S["rate_per_swing"])
            rec[f"contact_rw{w}"] = shrink_hat(sums["contact"].iloc[i], sums["swings"].iloc[i],  priors["contact"],PRIOR_S["rate_per_swing"])
            rec[f"gb_rw{w}"]      = shrink_hat(sums["gb"].iloc[i],      sums["bip"].iloc[i],     priors["gb"],     PRIOR_S["rate_per_bip"])
            rec[f"hard_rw{w}"]    = shrink_hat(sums["hard"].iloc[i],    sums["bip"].iloc[i],     priors["hard"],   PRIOR_S["rate_per_bip"])
            if include_csw_zone:
                csw_num = sums["whiffs"].iloc[i] + sums["cstrikes"].iloc[i]
                rec[f"csw_rw{w}"]  = shrink_hat(csw_num,                 sums["pitches"].iloc[i], priors["csw"],    PRIOR_S["rate_per_pitch"])
                rec[f"zone_rw{w}"] = shrink_hat(sums["inzone"].iloc[i],  sums["pitches"].iloc[i], priors["zone"],   PRIOR_S["rate_per_pitch"])
        rows.append(rec)
    return pd.DataFrame(rows)

def _rows_to_wide(rows: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if rows.empty: return pd.DataFrame({id_col: []})
    rows = rows.sort_values("game_date").groupby([id_col,"label"], as_index=False).tail(1)
    metric_cols = [c for c in rows.columns if c not in (id_col,"label","game_date")]
    w = rows.set_index([id_col,"label"])[metric_cols].unstack("label")
    w.columns = [f"{m}_{lab}" for (m,lab) in w.columns]
    return w.reset_index()

# ------------------------------ tendencies ------------------------------
def build_tendency_json(mix_df: pd.DataFrame, ptype_df: pd.DataFrame) -> Dict[str,Any]:
    out: Dict[str,Any] = {}
    for (pid, cnt, bas), g in mix_df.groupby(["pitcher","cnt","bas"]):
        d = out.setdefault(str(int(pid)), {})
        row = g.set_index("pitch_lbl")["n"]; tot = row.sum()
        if tot > 0:
            d[f"count:{cnt}:{bas}"] = (row / tot).round(6).to_dict()
    for pid, g in ptype_df.groupby("pitcher"):
        d = out.setdefault(str(int(pid)), {})
        for _, r in g.iterrows():
            p = r["pitch_lbl"]; swings = max(1.0, float(r["swings"]))
            d[f"whiff|{p}"] = float(np.clip(r["whiffs"]/swings, 0.0, 1.0))
            bip = max(1.0, float(r["bip"]))
            d[f"bip|{p}"] = {"GB": float(r["gb"]/bip), "FB": float(r["fb"]/bip),
                            "LD": float(r["ld"]/bip), "PU": float(r["pu"]/bip)}
    return out

# ------------------------------ main build ------------------------------
def build_profiles(stat_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print("📥 Streaming Statcast and aggregating…")
    (pit_daily, bat_daily, mix_df, ptype_df,
     pit_split_df, bat_split_df, pit_split_daily, bat_split_daily, id_to_name) = stream_aggregate(stat_path)

    print("📏 Computing league priors…")
    global priors
    pit_daily_rates = _add_rates(pit_daily)
    priors = _league_priors(pit_daily_rates)
    with open(os.path.join(out_dir,"global_priors.json"), "w") as f:
        json.dump(priors, f, indent=2)

    print("🔁 Adding overall rolling windows…")
    pit_roll = _add_rollings(pit_daily_rates, "pitcher")
    bat_roll = _add_rollings(_add_rates(bat_daily), "batter")

    print("🧮 Finalizing base profiles…")
    pit_prof = _finalize_profiles(pit_roll, "pitcher", priors)
    bat_prof = _finalize_profiles(bat_roll, "batter",  priors)

    # season totals for filtering
    pit_totals = pit_daily.groupby("pitcher", as_index=False)["pitches"].sum().rename(columns={"pitches":"pitches_total"})
    bat_totals = bat_daily.groupby("batter",  as_index=False)["pitches"].sum().rename(columns={"pitches":"pitches_total"})
    pit_prof = pit_prof.merge(pit_totals, on="pitcher", how="left")
    bat_prof = bat_prof.merge(bat_totals, on="batter",  how="left")

    # thresholds on season totals (NOT last game)
    pit_prof = pit_prof[pit_prof["pitches_total"].fillna(0) >= MIN_PITCHES_SEASON_P].copy()
    bat_prof = bat_prof[bat_prof["pitches_total"].fillna(0) >= MIN_PITCHES_SEASON_B].copy()

    # --------- SEASON-LEVEL SPLITS (shrunken) ----------
    print("🎯 Season-level L/R & Home/Away splits…")

    def season_splits_pitchers(ps: pd.DataFrame) -> pd.DataFrame:
        if ps.empty: return pd.DataFrame({"pitcher":[]})
        ps = ps.copy()
        ps["whiffs_cstrikes"] = ps["whiffs"] + ps["cstrikes"]
        ps["label_vs"]    = "vs" + ps["vs_bat_hand"].str.replace("S","BOTH", regex=False)
        ps["label_ha"]    = ps["p_home_away"]
        ps["label_vsha"]  = ps["label_vs"] + "_" + ps["label_ha"]

        def compute_rates(df):
            df = df.copy()
            def sh(num,den,key,scale): return shrink_series(df[num], df[den], priors[key], PRIOR_S[scale])
            out = pd.DataFrame({
                "pitcher": df["pitcher"],
                "label": df["label"],
                "csw_season":   sh("whiffs_cstrikes","pitches","csw","rate_per_pitch"),
                "swstr_season": sh("whiffs","swings","swstr","rate_per_swing"),
                "zone_season":  sh("inzone","pitches","zone","rate_per_pitch"),
                "contact_season": sh("contact","swings","contact","rate_per_swing"),
                "gb_season":    sh("gb","bip","gb","rate_per_bip"),
                "hard_season":  sh("hard","bip","hard","rate_per_bip"),
            })
            return out

        rows = []
        for sel in ("label_vs","label_ha","label_vsha"):
            tmp = ps.copy(); tmp["label"] = tmp[sel]
            rows.append(compute_rates(tmp))
        season_rows = pd.concat(rows, ignore_index=True)
        season_rows = season_rows.drop_duplicates(subset=["pitcher","label"])
        w = season_rows.pivot(index="pitcher", columns="label", values=["csw_season","swstr_season","zone_season","contact_season","gb_season","hard_season"])
        w.columns = [f"{m}_{lab}" for (m,lab) in w.columns]
        return w.reset_index()

    def season_splits_batters(bs: pd.DataFrame) -> pd.DataFrame:
        if bs.empty: return pd.DataFrame({"batter":[]})
        bs = bs.copy()
        bs["label_vs"]   = "vs" + bs["vs_pitch_hand"].str.replace("S","BOTH", regex=False)
        bs["label_ha"]   = bs["b_home_away"]
        bs["label_vsha"] = bs["label_vs"] + "_" + bs["label_ha"]

        def compute_rates(df):
            df = df.copy()
            def sh(num,den,key,scale): return shrink_series(df[num], df[den], priors[key], PRIOR_S[scale])
            out = pd.DataFrame({
                "batter": df["batter"],
                "label": df["label"],
                "swstr_season":   sh("whiffs","swings","swstr","rate_per_swing"),
                "contact_season": sh("contact","swings","contact","rate_per_swing"),
                "gb_season":      sh("gb","bip","gb","rate_per_bip"),
                "hard_season":    sh("hard","bip","hard","rate_per_bip"),
            })
            return out

        rows = []
        for sel in ("label_vs","label_ha","label_vsha"):
            tmp = bs.copy(); tmp["label"] = tmp[sel]
            rows.append(compute_rates(tmp))
        season_rows = pd.concat(rows, ignore_index=True)
        season_rows = season_rows.drop_duplicates(subset=["batter","label"])
        w = season_rows.pivot(index="batter", columns="label", values=["swstr_season","contact_season","gb_season","hard_season"])
        w.columns = [f"{m}_{lab}" for (m,lab) in w.columns]
        return w.reset_index()

    pit_season_wide = season_splits_pitchers(pit_split_df)
    bat_season_wide = season_splits_batters(bat_split_df)

    # --------- ROLLING SPLITS ----------
    print("🌀 Rolling L/R & Home/Away splits…")
    fmt_vs   = lambda d: "vs" + str(d["vs_bat_hand"]).replace("S","BOTH")
    fmt_ha   = lambda d: str(d["p_home_away"])
    fmt_vsha = lambda d: "vs" + str(d["vs_bat_hand"]).replace("S","BOTH") + "_" + str(d["p_home_away"])
    p_rows_vs   = _rolling_split_rows(pit_split_daily, "pitcher", ["vs_bat_hand"], fmt_vs,  include_csw_zone=True)
    p_rows_ha   = _rolling_split_rows(pit_split_daily, "pitcher", ["p_home_away"], fmt_ha, include_csw_zone=True)
    p_rows_vsha = _rolling_split_rows(pit_split_daily, "pitcher", ["vs_bat_hand","p_home_away"], fmt_vsha, include_csw_zone=True)
    p_wide_vs   = _rows_to_wide(p_rows_vs,   "pitcher")
    p_wide_ha   = _rows_to_wide(p_rows_ha,   "pitcher")
    p_wide_vsha = _rows_to_wide(p_rows_vsha, "pitcher")

    b_fmt_vs   = lambda d: "vs" + str(d["vs_pitch_hand"]).replace("S","BOTH")
    b_fmt_ha   = lambda d: str(d["b_home_away"])
    b_fmt_vsha = lambda d: "vs" + str(d["vs_pitch_hand"]).replace("S","BOTH") + "_" + str(d["b_home_away"])
    b_rows_vs   = _rolling_split_rows(bat_split_daily, "batter", ["vs_pitch_hand"], b_fmt_vs,  include_csw_zone=False)
    b_rows_ha   = _rolling_split_rows(bat_split_daily, "batter", ["b_home_away"],  b_fmt_ha,  include_csw_zone=False)
    b_rows_vsha = _rolling_split_rows(bat_split_daily, "batter", ["vs_pitch_hand","b_home_away"], b_fmt_vsha, include_csw_zone=False)
    b_wide_vs   = _rows_to_wide(b_rows_vs,   "batter")
    b_wide_ha   = _rows_to_wide(b_rows_ha,   "batter")
    b_wide_vsha = _rows_to_wide(b_rows_vsha, "batter")

    # --------- MERGE all into profiles ----------
    for w in (pit_season_wide, p_wide_vs, p_wide_ha, p_wide_vsha):
        if not w.empty: pit_prof = pit_prof.merge(w, on="pitcher", how="left")
    for w in (bat_season_wide, b_wide_vs, b_wide_ha, b_wide_vsha):
        if not w.empty: bat_prof = bat_prof.merge(w, on="batter",  how="left")

    # --------- SAVE ----------
    pit_path = os.path.join(out_dir, "pitcher_profiles.parquet")
    bat_path = os.path.join(out_dir, "batter_profiles.parquet")
    pit_prof.to_parquet(pit_path, index=False)
    bat_prof.to_parquet(bat_path, index=False)
    print(f"✅ Wrote {len(pit_prof):,} pitcher profiles → {pit_path}")
    print(f"✅ Wrote {len(bat_prof):,} batter profiles → {bat_path}")

    print("🎯 Building pitcher tendencies…")
    tend = build_tendency_json(mix_df, ptype_df)
    with open(os.path.join(out_dir, "tendency_tables.json"), "w") as f:
        json.dump(tend, f, indent=2)
    with open(os.path.join(out_dir,"id_to_name.json"), "w") as f:
        json.dump(id_to_name, f, indent=2)
    with open(os.path.join(out_dir,"global_priors.json"), "w") as f:
        json.dump(priors, f, indent=2)
    print("✅ Saved tendency_tables.json, global_priors.json, id_to_name.json")

# ------------------------------ cli ------------------------------
def main(master: str|None=None, out: str|None=None):
    master = master or "data/statcast_master.csv.gz"
    out    = out or OUT_DIR_DEFAULT
    build_profiles(master, out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", dest="master", default=None, help="Path to statcast_master.csv[.gz]")
    ap.add_argument("--out",    dest="out",    default=None, help="Output dir (default models/)")
    args = ap.parse_args()
    main(args.master, args.out)
