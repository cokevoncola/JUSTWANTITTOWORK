# src/picks.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .smart_mc_models import (
    SmartMonteCarloPredictor,
    SimulationGuards,
    default_true_k_probability,
)

__all__ = ["make_picks", "make_picks_smart_mc"]


# ---------------- Utilities ----------------

def _validate_task(task: Dict) -> None:
    if "pitcher" not in task or not isinstance(task["pitcher"], dict):
        raise ValueError("Task missing 'pitcher' dict")
    if "opponent_lineup" not in task or not isinstance(task["opponent_lineup"], list):
        raise ValueError("Task missing 'opponent_lineup' list")
    for b in task.get("opponent_lineup", []):
        if "hand" not in b or not b["hand"]:
            b["hand"] = "R"


def _to_tasks_from_dataframe(df: pd.DataFrame) -> List[Dict]:
    tasks: List[Dict] = []
    def pick(colnames):
        for c in df.columns:
            if c.strip().lower() in colnames:
                return c
        return None

    pid_col = pick({"pitcher_id", "mlbid"})
    pname_col = pick({"pitcher_name", "sp", "pitcher", "starter", "name"})
    hand_cols = [c for c in df.columns if c.strip().lower() in {"hand", "bats", "bat_hand", "batter_hand"}]

    if not pid_col and not pname_col:
        return tasks

    group_cols = [c for c in [pid_col, pname_col] if c]
    for _, g in df.groupby(group_cols, dropna=False):
        pitcher = {
            "id": g.iloc[0].get(pid_col) if pid_col else None,
            "name": str(g.iloc[0].get(pname_col)) if pname_col else None,
            "k_rate": float(g.iloc[0].get("k_rate", np.nan)) if "k_rate" in g.columns else None,
            "k_rate_vs_LHB": float(g.iloc[0].get("k_rate_vs_LHB", np.nan)) if "k_rate_vs_LHB" in g.columns else None,
            "k_rate_vs_RHB": float(g.iloc[0].get("k_rate_vs_RHB", np.nan)) if "k_rate_vs_RHB" in g.columns else None,
        }
        lineup = []
        for _, row in g.iterrows():
            hand = None
            for hc in hand_cols:
                if pd.notna(row.get(hc)):
                    hand = str(row.get(hc)).strip()[:1].upper()
                    break
            lineup.append({
                "batter_id": row.get("batter_id") if "batter_id" in g.columns else None,
                "hand": hand or "R",
                "k_rate": float(row.get("batter_k_rate", np.nan)) if "batter_k_rate" in g.columns else None,
                "k_rate_vs_LHP": float(row.get("k_rate_vs_LHP", np.nan)) if "k_rate_vs_LHP" in g.columns else None,
                "k_rate_vs_RHP": float(row.get("k_rate_vs_RHP", np.nan)) if "k_rate_vs_RHP" in g.columns else None,
            })

        tasks.append({
            "pitcher": pitcher,
            "opponent_lineup": lineup,
            "expected_bf": int(g.iloc[0].get("expected_bf", 24)) if "expected_bf" in g.columns else 24,
            "pitch_cap": int(g.iloc[0].get("pitch_cap", 110)) if "pitch_cap" in g.columns else 110,
            "pitches_per_pa": float(g.iloc[0].get("pitches_per_pa", 3.9)) if "pitches_per_pa" in g.columns else 3.9,
        })
    return tasks


def _binomial_tail_prob(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    q = 1.0 - p
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * (q ** (n - i))
    return float(min(max(total, 0.0), 1.0))


# ---------------- Heuristic path ----------------

def make_picks(
    data_or_tasks: Union[pd.DataFrame, List[Dict]],
    keep_samples: bool = False,
    cap_mode: Optional[str] = None,  # accepted for API symmetry (unused)
) -> pd.DataFrame:
    tasks = _to_tasks_from_dataframe(data_or_tasks) if isinstance(data_or_tasks, pd.DataFrame) else list(data_or_tasks or [])

    rows: List[Dict] = []
    for task in tasks:
        try:
            _validate_task(task)
        except Exception as e:
            rows.append({
                "pitcher_name": task.get("pitcher", {}).get("name"),
                "pitcher_id": task.get("pitcher", {}).get("id"),
                "expected_k": 0.0,
                "std_k": 0.0,
                "prob_6_plus": 0.0,
                "prob_8_plus": 0.0,
                "prob_10_plus": 0.0,
                "status": "error",
                "error": f"Invalid task: {e}",
            })
            continue

        pitcher = task["pitcher"]
        lineup = task.get("opponent_lineup", [])
        context = {k: v for k, v in task.items() if k not in {"pitcher", "opponent_lineup"}}
        expected_bf = max(12, min(int(task.get("expected_bf", 24)), 50))

        if not lineup:
            rows.append({
                "pitcher_name": pitcher.get("name"),
                "pitcher_id": pitcher.get("id"),
                "expected_k": 0.0,
                "std_k": 0.0,
                "prob_6_plus": 0.0,
                "prob_8_plus": 0.0,
                "prob_10_plus": 0.0,
                "status": "ok",
                "error": "",
            })
            continue

        pks = [float(max(0.02, min(0.60, default_true_k_probability(pitcher, b, context)))) for b in lineup]
        avg_pk = float(np.mean(pks))

        exp_k = expected_bf * avg_pk
        std_k = math.sqrt(expected_bf * avg_pk * (1.0 - avg_pk))
        prob6 = _binomial_tail_prob(expected_bf, avg_pk, 6)
        prob8 = _binomial_tail_prob(expected_bf, avg_pk, 8)
        prob10 = _binomial_tail_prob(expected_bf, avg_pk, 10)

        rows.append({
            "pitcher_name": pitcher.get("name"),
            "pitcher_id": pitcher.get("id"),
            "expected_k": float(exp_k),
            "std_k": float(std_k),
            "prob_6_plus": float(prob6),
            "prob_8_plus": float(prob8),
            "prob_10_plus": float(prob10),
            "status": "ok",
            "error": "",
        })

    df = pd.DataFrame(rows)
    preferred = [
        "pitcher_name", "pitcher_id", "expected_k", "std_k",
        "prob_6_plus", "prob_8_plus", "prob_10_plus", "status", "error",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]


# ---------------- Smart MC (threaded) ----------------

def _run_one_sim(
    predictor: SmartMonteCarloPredictor,
    task: Dict,
    n_sims: int,
    base_seed: Optional[int],
    cap_mode: Optional[str] = None,
) -> Dict:
    pid = task.get("pitcher", {}).get("id") or task.get("pitcher", {}).get("mlbid") or task.get("pitcher", {}).get("name") or "unknown"
    h = np.uint32(abs(hash(str(pid))) % (2**32))
    seed = int(h if base_seed is None else (h ^ np.uint32(base_seed)))

    task_in = dict(task)
    if cap_mode is not None and "cap_mode" not in task_in:
        task_in["cap_mode"] = cap_mode

    try:
        out = predictor.simulate_pitcher(task_in, n_sims=n_sims, rng_seed=seed)
        return {
            "pitcher_id": out.get("pitcher_id"),
            "pitcher_name": out.get("pitcher_name"),
            "expected_k": out.get("expected_k", 0.0),
            "std_k": out.get("std_k", 0.0),
            "prob_6_plus": out.get("prob_6_plus", 0.0),
            "prob_8_plus": out.get("prob_8_plus", 0.0),
            "prob_10_plus": out.get("prob_10_plus", 0.0),
            "meta": out.get("meta", {}),
            "samples": out.get("samples"),
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        return {
            "pitcher_id": task.get("pitcher", {}).get("id"),
            "pitcher_name": task.get("pitcher", {}).get("name"),
            "expected_k": 0.0,
            "std_k": 0.0,
            "prob_6_plus": 0.0,
            "prob_8_plus": 0.0,
            "prob_10_plus": 0.0,
            "meta": {"exception": str(e)},
            "samples": None,
            "status": "error",
            "error": str(e),
        }


def make_picks_smart_mc(
    tasks: List[Dict],
    models_dir: Optional[Union[str, Path]] = None,
    n_sims: int = 5000,
    n_jobs: int = 4,
    rng_seed: Optional[int] = None,
    max_pas_per_game: int = 50,
    per_game_timeout_s: float = 2.0,
    default_pitch_cap: int = 110,
    keep_samples: bool = False,
    cap_mode: Optional[str] = None,  # "hybrid" | "pitch" | "bf" | "none"
) -> pd.DataFrame:
    guards = SimulationGuards(
        max_pas_per_game=int(max_pas_per_game),
        per_game_timeout_s=float(per_game_timeout_s),
        default_pitch_cap=int(default_pitch_cap),
    )
    predictor = SmartMonteCarloPredictor(
        models_dir=models_dir,
        guards=guards,
        rng_seed=rng_seed,
    )

    for t in tasks:
        _validate_task(t)

    results: List[Dict] = Parallel(
        n_jobs=int(n_jobs),
        backend="threading",
        prefer="threads",
        batch_size="auto",
    )(
        delayed(_run_one_sim)(
            predictor=predictor,
            task=task,
            n_sims=int(n_sims),
            base_seed=rng_seed,
            cap_mode=cap_mode,
        )
        for task in tasks
    )

    df = pd.DataFrame(results)
    if not keep_samples and "samples" in df.columns:
        df = df.drop(columns=["samples"])

    preferred = [
        "pitcher_name", "pitcher_id", "expected_k", "std_k",
        "prob_6_plus", "prob_8_plus", "prob_10_plus", "status", "error",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]

