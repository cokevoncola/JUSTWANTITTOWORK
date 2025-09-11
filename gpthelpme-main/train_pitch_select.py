#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLB Skynet - PITCH SELECT (Multiclass) - Standalone

What it does
------------
- Loads statcast master CSV (gz ok)
- RAM-aware map-joins pitcher & batter profiles (parquet) in batches
- Builds leak-safe feature set for pitch selection
- Selects top-N pitcher/batter profile columns via a small sampling model (categorical-safe)
- Trains LightGBM multiclass with 5-fold CV, early stopping, n_jobs=7
- Saves:
  * models bundle (.joblib): models, feature_names, categorical_features, params, best_iterations, class_mapping
  * OOF CSV
  * reliability CSV + PNG (top-class)
  * metrics JSON (multi_logloss, macro_auc_ovr, ece_topclass, best_iterations)
- Prints fold metrics & summary to console

Notes
-----
- No custom wrappers; pure sklearn LightGBM API
- Categorical NA is handled by adding "__NA__" token (safe for LGBM)
- Numeric NA -> 0.0 (float32)
"""

import os
import gc
import json
import math
import time
import psutil
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.copy_on_write = True

# ---------------------------
# Leak guards / feature policy
# ---------------------------

BANNED_FEATURES = set([
    # Pitch Outcomes & Descriptions
    "events", "description", "des", "hit_location", "bb_type", "type",
    # Batted Ball Metrics (Leakage)
    "hit_distance_sc", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    "launch_speed_angle", "hc_x", "hc_y",
    # Bat Swing Metrics (Leakage before contact)
    "bat_speed", "swing_length", "hyper_speed",
    # Deprecated / Redundant IDs
    "spin_dir", "spin_rate_deprecated", "break_angle_deprecated", "break_length_deprecated",
    "tfs_deprecated", "tfs_zulu_deprecated", "game_pk", "sv_id", "at_bat_number", "pitch_number", "umpire",
    # Player & Team Identifiers (use profiles for these)
    "player_name", "batter", "pitcher", "fielder_2", "fielder_3", "fielder_4",
    "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "home_team", "away_team", "game_date", "game_type",
    # Post-Event Score (Leakage)
    "post_away_score", "post_home_score", "post_bat_score", "post_fld_score",
    # (pre-contact models should not use final scores either)
    "home_score", "away_score", "bat_score", "fld_score",
])

# Task-specific categorical features
TASK_CATS = [
    "stand", "p_throws", "if_fielding_alignment", "of_fielding_alignment", "inning_topbot"
]

# Base numeric features safe for pitch selection (pre-contact)
BASE_NUMS = [
    # pitch physics / location / count / game state
    "release_speed","release_pos_x","release_pos_y","release_pos_z",
    "release_spin_rate","release_extension","effective_speed","spin_axis","arm_angle",
    "pfx_x","pfx_z","plate_x","plate_z","vx0","vy0","vz0","ax","ay","az",
    "api_break_z_with_gravity","api_break_x_arm","api_break_x_batter_in",
    "zone","sz_top","sz_bot","balls","strikes","outs_when_up","inning",
    "on_1b","on_2b","on_3b",
    # score diffs allowed as context (NOT final outcomes)
    "home_score_diff","bat_score_diff",
    # ages / fatigue / order
    "age_pit","age_bat","n_thruorder_pitcher","n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game","batter_days_since_prev_game",
]

# ---------------------------
# Helpers
# ---------------------------

def human(n: float) -> str:
    for u in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000
    return f"{n:.1f}T"

def available_ram_gb() -> float:
    mem = psutil.virtual_memory()
    return (mem.available / (1024**3))

def read_master(path: str) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    print(f"[rows] {df.shape[0]:,} in {time.time()-t0:.1f}s")
    return df

def read_profiles(path: str, key: str) -> pd.DataFrame:
    # Keep numeric only; prefix columns so we never collide with base features
    prof = pd.read_parquet(path)
    if key not in prof.columns:
        raise ValueError(f"Profile parquet {path} missing ID column '{key}'")
    num_cols = [c for c in prof.select_dtypes(include=["number"]).columns if c != key]
    prof = prof[[key] + num_cols].copy()
    return prof

def map_join_profiles_in_batches(
    df: pd.DataFrame,
    pit: pd.DataFrame,
    bat: pd.DataFrame,
    pit_key: str = "pitcher",
    bat_key: str = "batter",
    ram_frac: float = 0.5,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    RAM-aware map-join: map each requested profile column to df by ID in batches,
    then single concat to avoid fragmentation.
    Returns: df_extended, pit_cols_added, bat_cols_added
    """
    ram = available_ram_gb()
    target_mem = max(2.0, ram * ram_frac)  # GB
    # heuristic: rows per batch to keep per-batch Series mapping comfortable
    rows = len(df)
    chunk_rows = max(50_000, min(rows, int(rows * min(0.75, (ram / 16.0)))))
    n_batches = math.ceil(rows / chunk_rows)

    # Build mapping dicts once (Series for each column)
    pit_map = {c: pit.set_index(pit_key)[c] for c in pit.columns if c != pit_key}
    bat_map = {c: bat.set_index(bat_key)[c] for c in bat.columns if c != bat_key}

    # We accumulate blocks then concat once
    blocks = []
    base_cols = list(df.columns)
    pit_cols_out = [f"pit_{c}" for c in pit.columns if c != pit_key]
    bat_cols_out = [f"bat_{c}" for c in bat.columns if c != bat_key]

    print(f"[join] RAM≈{ram:.1f} GB; chunk_rows≈{human(chunk_rows)}; batches={n_batches}")

    for bi in range(n_batches):
        s = bi * chunk_rows
        e = min(rows, s + chunk_rows)
        part = df.iloc[s:e, :][[pit_key, bat_key]].copy()

        # map all pitcher cols
        pit_block = {}
        pid = part[pit_key]
        for c, s_map in pit_map.items():
            pit_block[f"pit_{c}"] = pid.map(s_map)

        # map all batter cols
        bat_block = {}
        bid = part[bat_key]
        for c, s_map in bat_map.items():
            bat_block[f"bat_{c}"] = bid.map(s_map)

        # concatenate this batch's block once
        block = pd.concat(
            [df.iloc[s:e, :].reset_index(drop=True),
             pd.DataFrame(pit_block, index=part.index).reset_index(drop=True),
             pd.DataFrame(bat_block, index=part.index).reset_index(drop=True)],
            axis=1
        )
        blocks.append(block)
        if (bi+1) % 3 == 0 or (bi+1) == n_batches:
            print(f"[join] batch {bi+1}/{n_batches} rows={human(block.shape[0])}")

    out = pd.concat(blocks, axis=0, ignore_index=True)
    del blocks
    gc.collect()
    return out, pit_cols_out, bat_cols_out

def enforce_feature_policy(df: pd.DataFrame) -> None:
    # Drop banned columns if present (in-place)
    drop = [c for c in BANNED_FEATURES if c in df.columns]
    if drop:
        df.drop(columns=drop, inplace=True, errors="ignore")
        print(f"[prune] dropped={len(drop)} kept={df.shape[1]} (from {df.shape[1]+len(drop)})")

def cast_and_impute(df: pd.DataFrame, feats: List[str], cats: List[str]) -> pd.DataFrame:
    # Select and type-cast exactly once; avoid fragmented inserts
    X = df[feats].copy()

    cat_cols = [c for c in cats if c in X.columns]
    num_cols = [c for c in feats if c not in cat_cols]

    # categoricals
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X[c] = X[c].cat.add_categories(["__NA__"]).fillna("__NA__")

    # numerics
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    if num_cols:
        X[num_cols] = X[num_cols].fillna(0.0)

    return X

def compute_reliability_topclass(y_true: np.ndarray, P: np.ndarray, bins: int = 20) -> pd.DataFrame:
    """Top-class reliability: bin by max-proba, compute accuracy per bin."""
    max_p = P.max(axis=1)
    pred = P.argmax(axis=1)
    acc = (pred == y_true).astype(float)

    edges = np.linspace(0, 1, bins + 1)
    bins_idx = np.digitize(max_p, edges[1:-1], right=False)

    rows = []
    for b in range(bins):
        mask = (bins_idx == b)
        if mask.sum() == 0:
            rows.append((edges[b], edges[b+1], 0, np.nan, np.nan))
        else:
            conf = max_p[mask].mean()
            cal = acc[mask].mean()
            rows.append((edges[b], edges[b+1], int(mask.sum()), float(conf), float(cal)))
    rel = pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "count", "mean_conf", "mean_acc"])
    return rel

def expected_calibration_error(rel_df: pd.DataFrame) -> float:
    # ECE over top-class curve (weighted by bin counts)
    w = rel_df["count"].to_numpy()
    conf = rel_df["mean_conf"].to_numpy()
    acc = rel_df["mean_acc"].to_numpy()
    mask = np.isfinite(acc)
    if mask.sum() == 0:
        return float("nan")
    w = w[mask]; conf = conf[mask]; acc = acc[mask]
    if w.sum() == 0:
        return float("nan")
    return float(np.sum(w * np.abs(conf - acc)) / np.sum(w))

def macro_auc_ovr(y: np.ndarray, P: np.ndarray) -> float:
    aucs = []
    K = P.shape[1]
    for k in range(K):
        yk = (y == k).astype(int)
        try:
            aucs.append(roc_auc_score(yk, P[:, k]))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")

# --------------------------------------
# Profile feature selection (categorical-safe)
# --------------------------------------

def select_top_profile_features(
    df: pd.DataFrame,
    y: np.ndarray,
    base_feats: List[str],
    pit_feats: List[str],
    bat_feats: List[str],
    cats: List[str],
    max_pit: int,
    max_bat: int,
    sample_n: int,
    seed: int,
    n_jobs: int,
) -> Tuple[List[str], List[str]]:
    """Train a small LGBM on stratified sample; keep top-k pitcher/batter features."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if sample_n < n:
        idx = []
        cls_vals, cls_counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(cls_vals, cls_counts):
            cls_idx = np.flatnonzero(y == cls)
            take = min(len(cls_idx), max(500, int(sample_n * (cnt / n))))
            if take > 0:
                idx.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        idx = np.array(sorted(set(idx)))
    else:
        idx = np.arange(n)

    sample_cols = list(dict.fromkeys([*base_feats, *pit_feats, *bat_feats, *cats]))
    sample_cols = [c for c in sample_cols if c in df.columns]
    Xs = df.loc[idx, sample_cols].copy()

    cat_cols = [c for c in cats if c in Xs.columns]
    num_cols = [c for c in sample_cols if c not in cat_cols]

    for c in cat_cols:
        Xs[c] = Xs[c].astype("category")
        Xs[c] = Xs[c].cat.add_categories(["__NA__"]).fillna("__NA__")

    for c in num_cols:
        Xs[c] = pd.to_numeric(Xs[c], errors="coerce").astype("float32")
    if num_cols:
        Xs[num_cols] = Xs[num_cols].fillna(0.0)

    small = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y)),
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        max_bin=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        force_col_wise=True,
    )
    small.fit(Xs, y[idx], categorical_feature=cat_cols)

    imp = pd.Series(small.feature_importances_, index=sample_cols).sort_values(ascending=False)
    top_pit = [c for c in imp.index if c in pit_feats][:max_pit]
    top_bat = [c for c in imp.index if c in bat_feats][:max_bat]
    return top_pit, top_bat

# ---------------------------
# Training
# ---------------------------

def train(args: argparse.Namespace) -> None:
    print("====================================================================")
    print("[TASK] pitch_select")
    print(f"[file] {args.input}")

    df = read_master(args.input)

    # Build target mapping (pitch_type)
    pitch_codes = (
        df["pitch_type"].astype("string").str.upper().fillna("MISSING").unique().tolist()
    )
    pitch_codes = sorted(set(pitch_codes))
    cls_map = {c: i for i, c in enumerate(pitch_codes)}
    print("[info] pitch type mapping:", cls_map)

    y = df["pitch_type"].astype("string").str.upper().fillna("MISSING").map(cls_map).astype(int).to_numpy()

    # Enforce feature policy and add categories
    enforce_feature_policy(df)
    for c in TASK_CATS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # ---- Profiles (optional) ----
    pit_cols_out: List[str] = []
    bat_cols_out: List[str] = []
    if args.pitcher_profiles and os.path.exists(args.pitcher_profiles):
        pit = read_profiles(args.pitcher_profiles, key=args.pitcher_id_key)
    else:
        pit = pd.DataFrame({args.pitcher_id_key: []})
    if args.batter_profiles and os.path.exists(args.batter_profiles):
        bat = read_profiles(args.batter_profiles, key=args.batter_id_key)
    else:
        bat = pd.DataFrame({args.batter_id_key: []})

    if len(pit) and len(bat):
        df, pit_cols_out, bat_cols_out = map_join_profiles_in_batches(
            df, pit, bat,
            pit_key=args.pitcher_id_key,
            bat_key=args.batter_id_key,
            ram_frac=args.join_ram_frac
        )
        # keep list for later selection
    else:
        print("[profiles] missing or empty; skipping joins")

    # Feature lists
    base_feats = [c for c in BASE_NUMS if c in df.columns]
    cat_feats  = [c for c in TASK_CATS if c in df.columns]

    # Optionally select top profile features (categorical-safe)
    top_pit, top_bat = [], []
    if pit_cols_out or bat_cols_out:
        top_pit, top_bat = select_top_profile_features(
            df=df,
            y=y,
            base_feats=base_feats,
            pit_feats=pit_cols_out,
            bat_feats=bat_cols_out,
            cats=cat_feats,
            max_pit=args.max_pit_prof,
            max_bat=args.max_bat_prof,
            sample_n=args.fs_sample_n,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

    features = list(dict.fromkeys([*base_feats, *cat_feats, *top_pit, *top_bat]))
    cats = [c for c in cat_feats if c in features]

    # Prepare data
    X = cast_and_impute(df, features, cats)

    # CV setup
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    models = []
    best_iters = []
    K = len(cls_map)
    oof_P = np.zeros((len(X), K), dtype=np.float32)
    oof_idx = np.arange(len(X))

    params = dict(
        objective="multiclass",
        num_class=K,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        max_depth=-1,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        force_col_wise=True,
        max_bin=args.max_bin,
    )

    fold_no = 0
    for tr_idx, va_idx in skf.split(X, y):
        fold_no += 1
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=args.verbose)]
        )
        models.append(model)
        best_iters.append(int(model.best_iteration_ or params["n_estimators"]))
        P_va = model.predict_proba(X_va, num_iteration=model.best_iteration_)
        oof_P[va_idx] = P_va.astype(np.float32)

        # Fold metrics
        ll = log_loss(y_va, P_va, labels=list(range(K)))
        auc = macro_auc_ovr(y_va, P_va)
        print(f"[fold {fold_no}] logloss={ll:.5f}  macroAUC={auc:.4f}  best_iter={best_iters[-1]}")

    # OOF metrics
    oof_ll = log_loss(y, oof_P, labels=list(range(K)))
    oof_auc = macro_auc_ovr(y, oof_P)
    rel = compute_reliability_topclass(y, oof_P, bins=20)
    ece = expected_calibration_error(rel)

    print("------------------------------------------------------------------------")
    print("[SUMMARY] pitch_select")
    print(f"  oof_logloss     : {oof_ll:.6f}")
    print(f"  oof_macro_auc   : {oof_auc:.6f}")
    print(f"  ece_topclass    : {ece:.6f}")
    print(f"  best_iterations : {best_iters}")
    print("------------------------------------------------------------------------")

    # Save artifacts
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, "pitch_select")

    # 1) OOF CSV
    inv_map = {v: k for k, v in cls_map.items()}
    oof_df = pd.DataFrame({"y_true": y})
    # stable order by class index
    for i in range(K):
        colname = f"prob_{inv_map.get(i, str(i))}"
        oof_df[colname] = oof_P[:, i]
    oof_path = base + "_oof.csv"
    oof_df.to_csv(oof_path, index=False)

    # 2) Reliability CSV + PNG
    rel_path = base + "_reliability.csv"
    rel.to_csv(rel_path, index=False)
    png_path = base + "_reliability.png"
    plt.figure()
    plt.plot(rel["mean_conf"], rel["mean_acc"], marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Confidence (top-class)")
    plt.ylabel("Accuracy")
    plt.title("Pitch Select – Reliability (Top-Class)")
    plt.grid(True, alpha=0.3)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()

    # 3) Metrics JSON
    met = {
        "multi_logloss": float(oof_ll),
        "macro_auc_ovr": float(oof_auc),
        "ece_topclass": float(ece),
        "best_iterations": best_iters,
    }
    met_path = base + "_metrics.json"
    with open(met_path, "w") as f:
        json.dump(met, f, indent=2)

    # 4) Bundle .joblib
    bundle = {
        "models": models,
        "feature_names": features,
        "categorical_features": cats,
        "params": params,
        "best_iterations": best_iters,
        "class_mapping": cls_map,
    }
    model_path = base + ".joblib"
    joblib.dump(bundle, model_path)

    print("------------------------------------------------------------------------")
    print("[SKYNET] Saved artifacts:")
    print(f"  - model path    : {model_path}")
    print(f"  - oof csv       : {oof_path}")
    print(f"  - rel csv       : {rel_path}")
    print(f"  - rel png       : {png_path}")
    print(f"  - metrics json  : {met_path}")
    print(f"  - best_iterations: {best_iters}")
    print("------------------------------------------------------------------------")


# ---------------------------
# CLI
# ---------------------------

def make_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("train_pitch_select (standalone)")
    # IO
    ap.add_argument("--input", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "data", "raw", "statcast_master.csv.gz"))
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "models", "trained"))

    # Profiles
    ap.add_argument("--pitcher_profiles", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "models", "patched_floored", "pitcher_profiles.parquet"))
    ap.add_argument("--batter_profiles", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "models", "patched_floored", "batter_profiles.parquet"))
    ap.add_argument("--pitcher_id_key", type=str, default="pitcher")
    ap.add_argument("--batter_id_key", type=str, default="batter")
    ap.add_argument("--join_ram_frac", type=float, default=0.5)

    # Feature selection on profiles
    ap.add_argument("--max_pit_prof", type=int, default=80)
    ap.add_argument("--max_bat_prof", type=int, default=60)
    ap.add_argument("--fs_sample_n", type=int, default=250_000)

    # LightGBM / CV
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_estimators", type=int, default=3000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--max_bin", type=int, default=255)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n_jobs", type=int, default=7)
    ap.add_argument("--verbose", action="store_true", default=False)
    return ap


if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()
    train(args)
