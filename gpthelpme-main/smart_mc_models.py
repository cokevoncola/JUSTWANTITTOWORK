# src/smart_mc_models.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np

try:
    import joblib  # optional
except Exception:  # pragma: no cover
    joblib = None

Number = Union[int, float]


def _clamp(x: Number, lo: Number, hi: Number) -> Number:
    return max(lo, min(hi, x))


def _safe_get(d: Dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def default_true_k_probability(
    pitcher: Dict,
    batter: Dict,
    context: Optional[Dict] = None,
) -> float:
    """
    Conservative per-PA K probability blend with clamps and optional context nudges.
    Requires batter['hand'] in {'L','R','S'}; sim/adapter fills 'R' if unknown.
    """
    hand = (batter.get("hand") or "").upper()[:1] or "R"

    p_all = _safe_get(pitcher, "k_rate", default=0.22)
    p_vl = _safe_get(pitcher, "k_rate_vs_LHB", default=p_all)
    p_vr = _safe_get(pitcher, "k_rate_vs_RHB", default=p_all)
    p_split = p_vl if hand == "L" else p_vr

    b_all = _safe_get(batter, "k_rate", default=0.23)
    b_vl = _safe_get(batter, "k_rate_vs_LHP", default=b_all)
    b_vr = _safe_get(batter, "k_rate_vs_RHP", default=b_all)
    b_split = 0.5 * (b_vl + b_vr) if hand == "S" else (b_vl if hand == "L" else b_vr)

    def logit(p: float) -> float:
        p = _clamp(p, 1e-6, 1 - 1e-6)
        return math.log(p / (1 - p))

    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    prior = 0.225
    w_p, w_b, w_prior = 0.45, 0.45, 0.10
    z = (
        w_p * logit(_clamp(p_split, 0.05, 0.45))
        + w_b * logit(_clamp(b_split, 0.05, 0.45))
        + w_prior * logit(prior)
    )
    p = sigmoid(z)

    if context:
        p = _clamp(p * float(context.get("park_k_multiplier", 1.0)), 0.02, 0.6)
        p = _clamp(p + float(context.get("arsenal_k_delta", 0.0)), 0.02, 0.6)

    return float(_clamp(p, 0.02, 0.60))


@dataclass
class SimulationGuards:
    """Safety guards to prevent runaway simulations."""
    max_pas_per_game: int = 50
    per_game_timeout_s: float = 2.0
    default_pitch_cap: int = 110
    min_pas_per_game: int = 12
    bf_sigma: float = 2.5
    min_pitches_per_pa: float = 3.2
    max_pitches_per_pa: float = 4.5


class SmartMonteCarloPredictor:
    """Stateless, thread-safe Smart MC simulator."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        true_k_fn: Optional[Callable[[Dict, Dict, Optional[Dict]], float]] = None,
        guards: Optional[SimulationGuards] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else None
        self.true_k_fn = true_k_fn or default_true_k_probability
        self.guards = guards or SimulationGuards()
        self.base_rng = np.random.default_rng(rng_seed)

        self._models = {}
        if self.models_dir and joblib is not None:
            try:
                comp = self.models_dir / "comprehensive_trained_models.joblib"
                if comp.exists():
                    self._models["comprehensive"] = joblib.load(comp)
            except Exception:
                self._models["comprehensive"] = None

    def _now(self) -> float:
        return time.perf_counter()

    def _deadline(self, timeout_s: float) -> float:
        return self._now() + float(timeout_s)

    def simulate_pitcher(
        self,
        matchup: Dict,
        n_sims: int = 5000,
        rng_seed: Optional[int] = None,
    ) -> Dict:
        """
        Returns: dict with keys:
          'pitcher_id','pitcher_name','samples','expected_k','std_k',
          'prob_6_plus','prob_8_plus','prob_10_plus','meta'
        """
        g = self.guards
        pitcher = matchup.get("pitcher", {})
        lineup: List[Dict] = list(matchup.get("opponent_lineup", []))
        context = {k: v for k, v in matchup.items() if k not in {"pitcher", "opponent_lineup"}}

        if not lineup:
            return {
                "pitcher_id": pitcher.get("id") or pitcher.get("mlbid"),
                "pitcher_name": pitcher.get("name") or pitcher.get("player_name"),
                "samples": np.zeros(n_sims, dtype=int),
                "expected_k": 0.0,
                "std_k": 0.0,
                "prob_6_plus": 0.0,
                "prob_8_plus": 0.0,
                "prob_10_plus": 0.0,
                "meta": {"reason": "empty_lineup", "guards": vars(g)},
            }

        seed = int(
            rng_seed if rng_seed is not None
            else self.base_rng.integers(0, np.iinfo(np.uint32).max)
        )
        rng = np.random.default_rng(seed)

        # ---- Cap logic (respects cap_mode) ----
        expected_bf = int(_clamp(int(matchup.get("expected_bf") or 24), g.min_pas_per_game, g.max_pas_per_game))
        pitch_cap = int(matchup.get("pitch_cap") or g.default_pitch_cap)
        p_per_pa = _clamp(float(matchup.get("pitches_per_pa") or 3.9), g.min_pitches_per_pa, g.max_pitches_per_pa)

        bf_from_pitch_cap = int(max(g.min_pas_per_game, pitch_cap // p_per_pa))
        cap_mode = (matchup.get("cap_mode") or "hybrid").lower()
        bf_env = int(max(g.min_pas_per_game, round(expected_bf + 3 * g.bf_sigma)))

        if cap_mode == "pitch":
            hard_bf_cap = min(g.max_pas_per_game, bf_from_pitch_cap)
        elif cap_mode == "bf":
            hard_bf_cap = min(g.max_pas_per_game, bf_env)
        elif cap_mode == "none":
            hard_bf_cap = int(g.max_pas_per_game)
        else:  # "hybrid" or unknown
            hard_bf_cap = min(g.max_pas_per_game, bf_from_pitch_cap, bf_env)

        # ---- Pre-compute lineup K% ----
        lineup_pks = np.array(
            [float(_clamp(self.true_k_fn(pitcher, b, context), 0.02, 0.60)) for b in lineup],
            dtype=float,
        )
        lineup_len = len(lineup_pks)

        deadline = self._deadline(matchup.get("per_game_timeout_s", g.per_game_timeout_s))

        # Sample BF per sim and clamp
        bf_noise = rng.normal(0.0, g.bf_sigma, size=n_sims)
        bf_draws = np.round(expected_bf + bf_noise).astype(int)
        bf_draws = np.clip(bf_draws, g.min_pas_per_game, hard_bf_cap)

        unique_bf = np.unique(bf_draws)
        samples = np.zeros(n_sims, dtype=np.int16)

        if self._now() > deadline:
            return {
                "pitcher_id": pitcher.get("id") or pitcher.get("mlbid"),
                "pitcher_name": pitcher.get("name") or pitcher.get("player_name"),
                "samples": samples,
                "expected_k": 0.0,
                "std_k": 0.0,
                "prob_6_plus": 0.0,
                "prob_8_plus": 0.0,
                "prob_10_plus": 0.0,
                "meta": {"reason": "timeout_before_simulation", "guards": vars(g)},
            }

        for bf in unique_bf:
            if self._now() > deadline:
                break
            idx = np.where(bf_draws == bf)[0]
            if idx.size == 0:
                continue

            reps = int(math.ceil(bf / lineup_len))
            pks = np.tile(lineup_pks, reps)[:bf]
            U = rng.random(size=(idx.size, bf))
            samples[idx] = (U < pks).sum(axis=1).astype(np.int16)

        expected_k = float(samples.mean())
        std_k = float(samples.std(ddof=1)) if n_sims > 1 else 0.0
        p6 = float((samples >= 6).mean())
        p8 = float((samples >= 8).mean())
        p10 = float((samples >= 10).mean())

        return {
            "pitcher_id": pitcher.get("id") or pitcher.get("mlbid"),
            "pitcher_name": pitcher.get("name") or pitcher.get("player_name"),
            "samples": samples,
            "expected_k": expected_k,
            "std_k": std_k,
            "prob_6_plus": p6,
            "prob_8_plus": p8,
            "prob_10_plus": p10,
            "meta": {
                "applied_expected_bf": int(expected_bf),
                "applied_pitch_cap": int(pitch_cap),
                "applied_pitches_per_pa": float(p_per_pa),
                "hard_bf_cap": int(hard_bf_cap),
                "applied_cap_mode": str(cap_mode),
                "deadline_seconds": float(matchup.get("per_game_timeout_s", g.per_game_timeout_s)),
                "guards": vars(g),
                "seed": int(seed),
            },
        }
