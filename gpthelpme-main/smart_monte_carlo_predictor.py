#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Monte Carlo Predictor (runtime engine)

- Loads the comprehensive models bundle (main/poisson/quantile/isotonic)
- Aligns runtime features to the bundle's exact `feature_columns` order
- Uses season profiles (and optional split multipliers) for L/R and Home/Away effects
- Runs a vectorized Monte Carlo sim with a soft/hard pitch-cap, stamina, and hook risk
- (Optional) Loads pitch-select / whiff / BIP heads, with safe fallbacks
- Returns ML base, calibrated mean, percentiles, win-prob-ish "confidence", and matchup grade

You can instantiate this from anywhere:

    from src.smart_monte_carlo_predictor import SmartMonteCarloPredictor
    p = SmartMonteCarloPredictor(models_dir="models", pitch_count_cap=95)

"""

from __future__ import annotations

import os
import json
import math
import warnings
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ------------------------------ Utilities ------------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _grade_from_mean(mean_k: float) -> str:
    """Simple, transparent tiering for matchup grade. Adjust thresholds as you like."""
    if mean_k >= 9.5: return "A+"
    if mean_k >= 8.5: return "A"
    if mean_k >= 7.5: return "A-"
    if mean_k >= 6.5: return "B+"
    if mean_k >= 5.8: return "B"
    if mean_k >= 5.0: return "B-"
    if mean_k >= 4.2: return "C+"
    if mean_k >= 3.5: return "C"
    if mean_k >= 2.8: return "C-"
    if mean_k >= 2.0: return "D"
    return "F"


def _conf_from_interval(p10: float, p90: float) -> float:
    """
    0..1 "confidence" from the 80% interval width.
    Narrower band → higher confidence. Tuned to typical SP K spread.
    """
    if any(map(lambda v: v is None or (isinstance(v, float) and np.isnan(v)), [p10, p90])):
        return 0.5
    width = max(0.1, p90 - p10)  # avoid div by 0
    # 2.0 Ks spread → ~0.8, 4.0 spread → ~0.4
    conf = np.clip(1.6 / (width + 0.6), 0.15, 0.95)
    return float(conf)


# ------------------------ Core Predictor Class -------------------------

class SmartMonteCarloPredictor:
    def __init__(
        self,
        models_dir: str = "models",
        pitch_count_cap: int = 95,
        cap_mode: str = "soft",              # "soft" or "hard"
        hook_aggr: float = 1.0,              # >1.0 = more hook risk, <1.0 = less
        use_heads: bool = True,              # try pitch-select/whiff/BIP heads if available
        random_seed: int = 42,
    ):
        self.models_dir = models_dir
        self.pitch_count_cap = int(pitch_count_cap)
        self.cap_mode = str(cap_mode).lower()
        self.hook_aggr = float(hook_aggr)
        self.use_heads = bool(use_heads)

        self.random_state = np.random.RandomState(int(random_seed))

        # Data stores
        self.pitcher_profiles: Dict[int, Dict[str, Any]] = {}
        self.batter_profiles: Dict[int, Dict[str, Any]] = {}
        self.pitcher_id_to_name: Dict[int, str] = {}
        self.batter_id_to_name: Dict[int, str] = {}
        self.pitcher_name_to_id: Dict[str, int] = {}
        self.batter_name_to_id: Dict[str, int] = {}

        # Optional arsenals
        self.pitcher_arsenals: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.batter_arsenals: Dict[int, Dict[str, Dict[str, float]]] = {}

        # Pitch behavior knobs (heuristic defaults)
        self.pitch_zone_mult = {'FF':1.05,'FC':1.02,'SI':1.00,'SL':0.95,'CH':0.92,'CU':0.90,'KC':0.90,'FS':0.96}
        self.pitch_gb_delta  = {'SI':+0.08,'SL':+0.04,'CH':+0.02,'CU':+0.03,'KC':+0.03,'FC':0.00,'FS':-0.02,'FF':-0.05}
        self.default_usage_mix = {'FF':0.45,'SL':0.20,'CH':0.15,'CU':0.10,'SI':0.10}

        # Load everything we can
        self._load_cm_bundle()
        self._load_heads_if_any()
        self._load_profiles()
        self._load_arsenals_optional()

    # --------------------------- Data Loading ---------------------------

    def _load_cm_bundle(self):
        """Comprehensive models bundle (trained by train_comprehensive_models.py)."""
        try:
            path = os.path.join(self.models_dir, "comprehensive_trained_models.joblib")
            bundle = joblib.load(path)
            self._cm_models  = bundle.get("models", {})              # e.g., main, poisson, q10, q50, q90, isotonic
            self._cm_scalers = bundle.get("scalers", {})             # optional: {name: scaler}
            self._cm_feats   = bundle.get("feature_columns", [])     # exact order
            self._cm_stats   = bundle.get("training_stats", {})
            self._calibrator = bundle.get("calibrator", None)
            print("✅ Loaded comprehensive models bundle.")
        except Exception as e:
            print(f"ℹ️ Comprehensive bundle not loaded ({e}) — ML base will fall back to priors.")
            self._cm_models, self._cm_scalers, self._cm_feats, self._cm_stats, self._calibrator = {}, {}, [], {}, None

    def _load_heads_if_any(self):
        """Optional pitch-select/whiff/BIP heads from train_tendencies.py."""
        self._heads = {}
        for name in ("pitch_select", "whiff", "bip"):
            fp = os.path.join(self.models_dir, f"{name}.joblib")
            if os.path.exists(fp):
                try:
                    self._heads[name] = joblib.load(fp)
                except Exception:
                    pass

    def _load_profiles(self):
        """
        Load pitcher & batter profiles from parquet if available.
        Expected minimal fields:
          pitcher: mlbid, player_name, k_rate, bb_rate, swstr_rate, zone_rate, contact_rate, chase_rate, hard_contact_rate, barrel_rate, gb_rate, throws
          batter:  mlbid, player_name, k_rate, bb_rate, contact_rate, chase_rate, power_rate, gb_rate, bats
        If split columns exist (e.g., *_vsL, *_vsR, *_home, *_away), we convert them into multipliers.
        """
        pit_path = os.path.join(self.models_dir, "pitcher_profiles.parquet")
        bat_path = os.path.join(self.models_dir, "batter_profiles.parquet")

        # ---- Pitchers
        if os.path.exists(pit_path):
            try:
                pdf = pd.read_parquet(pit_path)
                # take last row per pitcher as "season" (or already pre-aggregated)
                grp = pdf.groupby("pitcher", as_index=False).tail(1)
                for _, r in grp.iterrows():
                    pid = int(r.get("pitcher", r.get("mlbid", -1)))
                    if pid < 0:
                        continue
                    name = str(r.get("player_name", f"Pitcher {pid}"))
                    prof = {
                        "mlbid": pid,
                        "player_name": name,
                        "k_rate": _safe_float(r.get("k_rate", r.get("k_rate_season", 0.22)), 0.22),
                        "bb_rate": _safe_float(r.get("bb_rate", 0.08), 0.08),
                        "swstr_rate": _safe_float(r.get("swstr_rate", 0.12), 0.12),
                        "zone_rate": _safe_float(r.get("zone_rate", 0.45), 0.45),
                        "contact_rate": _safe_float(r.get("contact_rate", 0.77), 0.77),
                        "chase_rate": _safe_float(r.get("chase_rate", 0.30), 0.30),
                        "hard_contact_rate": _safe_float(r.get("hard_contact_rate", 0.35), 0.35),
                        "barrel_rate": _safe_float(r.get("barrel_rate", 0.08), 0.08),
                        "gb_rate": _safe_float(r.get("gb_rate", 0.44), 0.44),
                        "throws": str(r.get("p_throws", r.get("throws", "R")) or "R"),
                        "stamina_factor": 1.0,
                        "splits": {}
                    }
                    # capture split multipliers if present
                    for key in ("k_rate", "swstr_rate", "gb_rate"):
                        base = prof[key]
                        if base <= 0:
                            continue
                        for suff, label in (("_vsL", "vsL"), ("_vsR", "vsR"),
                                            ("_home", "home"), ("_away", "away"),
                                            ("_vsL_home", "vsL_home"), ("_vsR_home", "vsR_home")):
                            col = f"{key}{suff}"
                            if col in r and pd.notna(r[col]):
                                prof["splits"].setdefault(label, {})[key] = float(r[col]) / base
                    self.pitcher_profiles[pid] = prof
                    self.pitcher_id_to_name[pid] = name
                    self.pitcher_name_to_id[name] = pid
                print(f"✅ Loaded {len(self.pitcher_profiles)} pitcher profiles (parquet)")
            except Exception as e:
                print(f"ℹ️ Could not parse pitcher_profiles.parquet: {e}")

        # ---- Batters
        if os.path.exists(bat_path):
            try:
                bdf = pd.read_parquet(bat_path)
                grp = bdf.groupby("batter", as_index=False).tail(1)
                for _, r in grp.iterrows():
                    bid = int(r.get("batter", r.get("mlbid", -1)))
                    if bid < 0:
                        continue
                    name = str(r.get("player_name", f"Batter {bid}"))
                    prof = {
                        "mlbid": bid,
                        "player_name": name,
                        "k_rate": _safe_float(r.get("k_rate", 0.22), 0.22),
                        "bb_rate": _safe_float(r.get("bb_rate", 0.08), 0.08),
                        "contact_rate": _safe_float(r.get("contact_rate", 0.75), 0.75),
                        "chase_rate": _safe_float(r.get("chase_rate", 0.30), 0.30),
                        "power_rate": _safe_float(r.get("power_rate", r.get("hardhit_rate", 0.35)), 0.35),
                        "gb_rate": _safe_float(r.get("gb_rate", 0.43), 0.43),
                        "bats": str(r.get("stand", r.get("bats", "R")) or "R"),
                        "splits": {}
                    }
                    for key in ("k_rate", "contact_rate", "gb_rate"):
                        base = prof[key]
                        if base <= 0:
                            continue
                        for suff, label in (("_vsR", "vsR"), ("_vsL", "vsL"),
                                            ("_home", "home"), ("_away", "away"),
                                            ("_vsR_home", "vsR_home"), ("_vsL_home", "vsL_home")):
                            col = f"{key}{suff}"
                            if col in r and pd.notna(r[col]):
                                prof["splits"].setdefault(label, {})[key] = float(r[col]) / base
                    self.batter_profiles[bid] = prof
                    self.batter_id_to_name[bid] = name
                    self.batter_name_to_id[name] = bid
                print(f"✅ Loaded {len(self.batter_profiles)} batter profiles (parquet)")
            except Exception as e:
                print(f"ℹ️ Could not parse batter_profiles.parquet: {e}")
        else:
            print("ℹ️ No batter_profiles.parquet — batters fall back to league-average.")

    def _load_arsenals_optional(self):
        """Optional CSVs for pitch usage/whiff by type."""
        # Pitchers
        try:
            pa = pd.read_csv(os.path.join(self.models_dir, "pitcher_arsenal2025.csv"))
            for _, row in pa.iterrows():
                pid = int(row.get("mlbid", 0)) if pd.notna(row.get("mlbid", np.nan)) else None
                if not pid:
                    continue
                pt = str(row.get("pitch_type", "")).strip()
                usage = _safe_float(row.get("pitch_usage", 0.0), 0.0)
                whiff = row.get("whiff_percent", np.nan)
                self.pitcher_arsenals.setdefault(pid, {'usage': {}, 'whiff': {}})
                if pt:
                    self.pitcher_arsenals[pid]['usage'][pt] = usage
                    if pd.notna(whiff):
                        self.pitcher_arsenals[pid]['whiff'][pt] = float(whiff) / 100.0
        except Exception:
            pass

        # Batters
        try:
            ba = pd.read_csv(os.path.join(self.models_dir, "batterarsenal2025_stats.csv"))
            for _, row in ba.iterrows():
                bid = int(row.get("mlbid", 0)) if pd.notna(row.get("mlbid", np.nan)) else None
                if not bid:
                    continue
                pt = str(row.get("pitch_type", "")).strip()
                usage = _safe_float(row.get("pitch_usage", 0.0), 0.0)
                whiff = row.get("whiff_percent", np.nan)
                self.batter_arsenals.setdefault(bid, {'usage': {}, 'whiff': {}})
                if pt:
                    self.batter_arsenals[bid]['usage'][pt] = usage
                    if pd.notna(whiff):
                        self.batter_arsenals[bid]['whiff'][pt] = float(whiff) / 100.0
        except Exception:
            pass

    # ----------------------- ID Resolving Helpers -----------------------

    def _default_pitcher_profile(self, mlbid: int, name: str) -> Dict[str, Any]:
        return {
            'mlbid': mlbid, 'player_name': name,
            'k_rate': 0.22, 'bb_rate': 0.08,
            'swstr_rate': 0.12, 'zone_rate': 0.45, 'contact_rate': 0.77,
            'chase_rate': 0.30, 'hard_contact_rate': 0.35, 'barrel_rate': 0.08,
            'total_batters': 100, 'strikeouts': 22, 'gb_rate': 0.44,
            'stamina_factor': 1.0, 'throws': 'R', 'splits': {}
        }

    def _default_batter_profile(self, mlbid: int, name: str) -> Dict[str, Any]:
        return {
            'mlbid': mlbid, 'player_name': name,
            'k_rate': 0.22, 'bb_rate': 0.08, 'contact_rate': 0.75,
            'chase_rate': 0.30, 'power_rate': 0.35, 'gb_rate': 0.43,
            'overall_take_rate': None, 'bats': 'R', 'splits': {}
        }

    def _resolve_pitcher(self, pitcher_id_or_name) -> Tuple[int, str, Dict[str, Any]]:
        pid, name = None, None
        if isinstance(pitcher_id_or_name, (int, np.integer)):
            pid = int(pitcher_id_or_name)
            name = self.pitcher_id_to_name.get(pid)
        elif isinstance(pitcher_id_or_name, str):
            name = pitcher_id_or_name
            pid = self.pitcher_name_to_id.get(name)

        if pid is not None and pid in self.pitcher_profiles:
            prof = self.pitcher_profiles[pid].copy()
            return pid, prof.get("player_name", name or f"Pitcher {pid}"), prof

        if pid is not None:
            nm = name or f"Pitcher {pid}"
            return pid, nm, self._default_pitcher_profile(pid, nm)

        # fallback: hash name to fake id
        fake = -abs(hash(str(pitcher_id_or_name))) % (10**9)
        nm = str(pitcher_id_or_name)
        return fake, nm, self._default_pitcher_profile(fake, nm)

    def _resolve_batter(self, batter_id_or_name) -> Tuple[int, str, Dict[str, Any]]:
        bid, name = None, None
        if isinstance(batter_id_or_name, (int, np.integer)):
            bid = int(batter_id_or_name)
            name = self.batter_id_to_name.get(bid)
        elif isinstance(batter_id_or_name, str):
            name = batter_id_or_name
            bid = self.batter_name_to_id.get(name)

        if bid is not None and bid in self.batter_profiles:
            prof = self.batter_profiles[bid].copy()
            return bid, prof.get("player_name", name or f"Batter {bid}"), prof

        if bid is not None:
            nm = name or f"Batter {bid}"
            return bid, nm, self._default_batter_profile(bid, nm)

        fake = -abs(hash(str(batter_id_or_name))) % (10**9)
        nm = str(batter_id_or_name)
        return fake, nm, self._default_batter_profile(fake, nm)

    # ------------------- Feature Alignment for ML Base -------------------

    def _feature_value_map(self, pitcher_pid: int, lineup_bids, venue: str) -> dict:
        """Produces a {feature_name: value} dict for the bundle's feature list."""
        p = self.pitcher_profiles.get(pitcher_pid, self._default_pitcher_profile(pitcher_pid, f"Pitcher {pitcher_pid}"))
        bats = [self.batter_profiles.get(b, self._default_batter_profile(b, f"B{i+1}"))
                for i, b in enumerate(lineup_bids or [])]

        arr = lambda vals, d: np.array([_safe_float(v, d) for v in vals], dtype=float)

        lineup_k   = arr([b.get("k_rate", 0.22) for b in bats], 0.22)
        lineup_con = arr([b.get("contact_rate", 0.75) for b in bats], 0.75)

        m = {
            # pitcher core
            "p_k_rate":       _safe_float(p.get("k_rate", 0.22), 0.22),
            "p_bb_rate":      _safe_float(p.get("bb_rate", 0.08), 0.08),
            "p_swstr_rate":   _safe_float(p.get("swstr_rate", 0.12), 0.12),
            "p_zone_rate":    _safe_float(p.get("zone_rate", 0.45), 0.45),
            "p_contact_rate": _safe_float(p.get("contact_rate", 0.77), 0.77),
            "p_chase_rate":   _safe_float(p.get("chase_rate", 0.30), 0.30),
            "p_hardhit":      _safe_float(p.get("hard_contact_rate", 0.35), 0.35),
            "p_barrel":       _safe_float(p.get("barrel_rate", 0.08), 0.08),
            "p_gb_rate":      _safe_float(p.get("gb_rate", 0.44), 0.44),

            # lineup aggregates
            "opp_avg_k":        float(lineup_k.mean()) if lineup_k.size else 0.22,
            "opp_std_k":        float(lineup_k.std())  if lineup_k.size else 0.0,
            "opp_avg_contact":  float(lineup_con.mean()) if lineup_con.size else 0.75,
            "opp_std_contact":  float(lineup_con.std())  if lineup_con.size else 0.0,
            "opp_share_hiK":    float((lineup_k > 0.25).mean()) if lineup_k.size else 0.0,
            "opp_share_loK":    float((lineup_k < 0.18).mean()) if lineup_k.size else 0.0,

            # interactions
            "pK_x_oppK":        _safe_float(p.get("k_rate", 0.22), 0.22) * (float(lineup_k.mean()) if lineup_k.size else 0.22),
            "pSwStr_x_oppMiss": _safe_float(p.get("swstr_rate", 0.12), 0.12) * (1.0 - (float(lineup_con.mean()) if lineup_con.size else 0.75)),
            "pZone_x_pK":       _safe_float(p.get("zone_rate", 0.45), 0.45) * _safe_float(p.get("k_rate", 0.22), 0.22),

            # venue flag (simple)
            "venue_neutral":    1.0 if (venue or "").lower() == "neutral" else 0.0,
        }

        # If your bundle includes richer split/recency features by name, it’s OK to leave them 0.0 here;
        # the model will just treat them as absent for inference. If you want to wire them:
        # m["p_k_rate_vsL"] = p["splits"].get("vsL", {}).get("k_rate", 1.0) * m["p_k_rate"]
        return m

    def _create_feature_vector_v2(self, pitcher_pid: int, lineup_bids, venue: str):
        if not getattr(self, "_cm_feats", []):
            return np.zeros((1, 0), dtype=float)
        m = self._feature_value_map(pitcher_pid, lineup_bids, venue)
        vec = np.array([_safe_float(m.get(col, 0.0), 0.0) for col in self._cm_feats], dtype=float).reshape(1, -1)
        return vec

    def _predict_with(self, model_name: str, X: np.ndarray) -> np.ndarray:
        mdl = self._cm_models.get(model_name)
        if mdl is None:
            raise KeyError(f"model '{model_name}' not in bundle")
        sc = self._cm_scalers.get(model_name)
        if sc is not None:
            try:
                X = sc.transform(X)
            except Exception:
                pass
        return mdl.predict(X)

    def get_ml_base_prediction(self, pitcher_pid: int, lineup_bids, venue: str) -> Tuple[float, Dict[str, float]]:
        """
        Returns (ml_base_mean, {'q10':..., 'q50':..., 'q90':...}) if quantiles exist.
        Isotonic calibration is applied if present.
        """
        # ultimate fallback
        fallback = self.pitcher_profiles.get(pitcher_pid, {}).get("k_rate", 0.22) * 25.0
        if not getattr(self, "_cm_models", None) or not getattr(self, "_cm_feats", None):
            return float(fallback), {}

        X = self._create_feature_vector_v2(pitcher_pid, lineup_bids, venue)

        base = None
        extras: Dict[str, float] = {}

        # main tree head
        try:
            base = float(self._predict_with("main", X)[0])
        except Exception as e:
            print("   Model main error:", e)

        # auxiliary Poisson head (blend)
        try:
            pois = float(self._predict_with("poisson", X)[0])
            base = float(pois) if base is None else 0.7 * float(base) + 0.3 * float(pois)
        except Exception:
            pass

        if base is None:
            base = fallback

        # quantiles for direct p10/p50/p90
        for qname in ("q10", "q50", "q90"):
            try:
                extras[qname] = float(self._predict_with(qname, X)[0])
            except Exception:
                pass

        # isotonic calibrator
        try:
            if "isotonic" in self._cm_models:
                iso = self._cm_models["isotonic"]
                base = float(iso.predict([base])[0])
            elif self._calibrator is not None:
                base = float(self._calibrator.predict([base])[0])
        except Exception as e:
            print("   Model isotonic error:", e)

        return float(base), extras

    # ------------------------ Public Entry Point ------------------------

    def vectorized_pitch_simulation(
        self,
        pitcher_id_or_name=None,
        opposing_lineup: Optional[List[Any]] = None,
        venue: str = "neutral",
        simulations: int = 3000,
        use_heads: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Main API."""
        if pitcher_id_or_name is None:
            raise ValueError("No pitcher identifier provided.")

        if opposing_lineup is None or len(opposing_lineup) == 0:
            opposing_lineup = [{"name": "League Avg Batter"} for _ in range(9)]

        pid, pname, pprof = self._resolve_pitcher(pitcher_id_or_name)

        # Resolve lineup → IDs/profiles
        lineup_ids, lineup_names, lineup_profiles = [], [], []
        for b in opposing_lineup:
            if isinstance(b, dict):
                if 'mlbid' in b and pd.notna(b['mlbid']):
                    bid, bname, bprof = self._resolve_batter(int(b['mlbid']))
                elif 'name' in b:
                    bid, bname, bprof = self._resolve_batter(b['name'])
                else:
                    bid, bname, bprof = self._resolve_batter(b.get('batter') or b.get('player_name') or "")
            elif isinstance(b, (int, np.integer, str)):
                bid, bname, bprof = self._resolve_batter(b)
            else:
                bid, bname, bprof = self._resolve_batter(str(b))
            lineup_ids.append(bid); lineup_names.append(bname); lineup_profiles.append(bprof)

        ml_base, q_extras = self.get_ml_base_prediction(pid, lineup_ids, venue)

        dist, batter_ks_agg = self._run_simulations(
            pitcher_profile=pprof,
            lineup_profiles=lineup_profiles,
            venue=venue,
            simulations=int(simulations),
            use_heads=self.use_heads if use_heads is None else bool(use_heads),
        )

        # shift MC to match ML mean if quantiles are missing or you prefer ML anchor
        sim_mean = float(np.mean(dist)) if dist.size else float('nan')
        if np.isfinite(sim_mean):
            dist = dist + (float(ml_base) - sim_mean)

        # aggregate outputs
        mean_val = float(np.mean(dist)) if dist.size else float('nan')
        median_val = float(np.median(dist)) if dist.size else float('nan')
        std_val = float(np.std(dist)) if dist.size else float('nan')

        # percentiles: prefer quantile heads if present
        if all(k in q_extras for k in ("q10", "q50", "q90")):
            p10, p50, p90 = q_extras["q10"], q_extras["q50"], q_extras["q90"]
            # get p25/p75 by MC as a hybrid, or just interpolate
            p25, p75 = np.percentile(dist, [25, 75]).tolist() if dist.size else [float('nan')]*2
        else:
            p10, p25, p75, p90 = (np.percentile(dist, [10, 25, 75, 90]).tolist() if dist.size else [float('nan')]*4)
            p50 = float(np.percentile(dist, 50)) if dist.size else float('nan')

        probs = {
            'prob_over_4_5': float(np.mean(dist >= 5)) if dist.size else float('nan'),
            'prob_over_5_5': float(np.mean(dist >= 6)) if dist.size else float('nan'),
            'prob_over_6_5': float(np.mean(dist >= 7)) if dist.size else float('nan'),
            'prob_over_7_5': float(np.mean(dist >= 8)) if dist.size else float('nan'),
            'prob_over_8_5': float(np.mean(dist >= 9)) if dist.size else float('nan'),
        }

        # "confidence": from p90-p10 if available, else from MC band
        conf = _conf_from_interval(p10, p90)

        # matchup grade from final mean
        matchup_grade = _grade_from_mean(mean_val)

        # by-batter breakdown
        batter_analysis = {}
        for i, bname in enumerate(lineup_names):
            arr = np.array(batter_ks_agg.get(i, []), dtype=float)
            if arr.size == 0:
                continue
            batter_analysis[bname] = {
                'avg_ks': float(np.mean(arr)),
                'k_probability': float(np.mean(arr > 0)),
                'multiple_k_prob': float(np.mean(arr > 1)),
            }

        return {
            'pitcher_name': pname,
            'pitcher_mlbid': pid,
            'venue': venue,
            'ml_base_prediction': float(ml_base),
            'final_prediction': float(mean_val),
            'mean_strikeouts': float(mean_val),
            'median': float(median_val if np.isfinite(median_val) else p50),
            'std_dev': float(std_val),

            'percentile_10': float(p10),
            'percentile_25': float(p25),
            'percentile_50': float(p50),
            'percentile_75': float(p75),
            'percentile_90': float(p90),

            **probs,
            'confidence': float(conf),
            'matchup_grade': matchup_grade,

            'simulations_run': int(simulations),
            'methodology': 'Comprehensive ML + Monte Carlo',
            'batter_breakdown': batter_analysis,
            'distribution': dist.astype(float).round(3).tolist()
        }

    # ----------------------- Simulation internals -----------------------

    def _venue_k_factor(self, venue: str) -> float:
        table = {
            'Coors Field': 0.88, 'Fenway Park': 0.95, 'Yankee Stadium': 1.02,
            'Petco Park': 1.08, 'Oracle Park': 1.05, 'loanDepot park': 1.03,
            'Marlins Park': 1.03, 'T-Mobile Park': 1.02, 'Tropicana Field': 1.01,
            'Kauffman Stadium': 1.04, 'Minute Maid Park': 0.97,
        }
        return float(table.get(venue, 1.00))

    def _run_simulations(self, pitcher_profile: Dict[str, Any], lineup_profiles: List[Dict[str, Any]],
                         venue: str, simulations: int, use_heads: bool) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        results = []
        batter_ks_tracking: Dict[int, List[int]] = {i: [] for i in range(len(lineup_profiles))}
        for _ in range(simulations):
            game = self._simulate_game(pitcher_profile, lineup_profiles, venue, use_heads)
            results.append(game['total_ks'])
            for i, k in enumerate(game['batter_ks']):
                batter_ks_tracking[i].append(k)
        return np.array(results, dtype=float), batter_ks_tracking

    def _stamina_curve(self, pitch_count: int, inning: int, pitcher: Dict[str, Any]) -> float:
        if self.cap_mode == "hard" and pitch_count >= self.pitch_count_cap:
            return 0.0
        if pitch_count < 60:
            base = 1.0
        elif pitch_count < 80:
            base = 0.95
        elif pitch_count < 100:
            base = 0.85
        else:
            base = 0.72
        inning_pen = 0.97 if inning >= 7 else 1.0
        # hook aggressiveness: >1.0 accelerates fatigue, <1.0 slows it
        return float(base * inning_pen * pitcher.get('stamina_factor', 1.0) / max(0.5, self.hook_aggr))

    def _simulate_game(self, pitcher: Dict[str, Any], lineup: List[Dict[str, Any]], venue: str, use_heads: bool) -> Dict[str, Any]:
        total_ks = 0
        batter_ks = [0] * len(lineup)

        outs = 0
        inning = 1
        lineup_pos = 0
        pitch_count = 0
        stamina = 1.0

        on1, on2, on3 = False, False, False
        venue_mult = self._venue_k_factor(venue)
        pitcher_eff = pitcher.copy()
        pitcher_eff['swstr_rate'] = pitcher.get('swstr_rate', 0.12) * venue_mult

        while outs < 27 and inning <= 9 and stamina > 0.25 and pitch_count < (self.pitch_count_cap + 12):
            i = lineup_pos % len(lineup)
            batter = lineup[i]

            dp_context = bool(on1 and outs <= 1)

            pa_outcome = self._simulate_plate_appearance_with_count(
                pitcher=pitcher_eff,
                batter=batter,
                runners_on=(on1 or on2 or on3),
                dp_context=dp_context,
                use_heads=use_heads
            )

            pitch_count += pa_outcome['pitches']
            stamina = self._stamina_curve(pitch_count, inning, pitcher)

            outcome = pa_outcome['outcome']
            if outcome == 'strikeout':
                total_ks += 1
                batter_ks[i] += 1
                outs += 1

            elif outcome in ('walk', 'hbp'):
                on1, on2, on3 = self._advance_walk(on1, on2, on3)

            elif outcome == 'ground_out':
                if on1 and outs <= 1:
                    dp_chance = 0.35 * (0.9 + 0.4 * (pitcher.get('gb_rate', 0.44) - 0.44)/0.2)
                    if self.random_state.rand() < dp_chance:
                        outs += 2
                        on1 = False
                    else:
                        outs += 1
                        if self.random_state.rand() < 0.3 and on1:
                            if not on2: on2 = True
                            on1 = False
                else:
                    outs += 1
                    if on3 and self.random_state.rand() < 0.2:
                        on3 = False
                    if on2 and self.random_state.rand() < 0.15:
                        on3 = True; on2 = False
                    if on1 and self.random_state.rand() < 0.15:
                        on2 = True; on1 = False

            elif outcome in ('fly_out', 'line_out', 'pop_out'):
                if outs <= 1 and outcome == 'fly_out' and on3 and self.random_state.rand() < 0.3:
                    on3 = False
                outs += 1

            elif outcome == 'single':
                on1, on2, on3 = self._advance_on_hit(1, on1, on2, on3)

            elif outcome == 'double':
                on1, on2, on3 = self._advance_on_hit(2, on1, on2, on3)

            elif outcome == 'triple':
                on1, on2, on3 = self._advance_on_hit(3, on1, on2, on3)

            elif outcome == 'home_run':
                on1, on2, on3 = False, False, False

            lineup_pos += 1
            if outs >= inning * 3:
                inning += 1

        return {
            'total_ks': total_ks,
            'batter_ks': batter_ks,
            'innings_pitched': min(inning - 1, 9),
            'final_pitch_count': pitch_count,
            'final_stamina': stamina
        }

    # ---------------------- Pitch / PA Mechanics ----------------------

    def _choose_pitch_type(self, pitcher: Dict[str, Any], batter: Dict[str, Any],
                           balls: int, strikes: int, runners_on: bool, dp_context: bool) -> Tuple[str, Dict[str, float]]:
        """
        If heads present, you can wire them here; for now we keep robust heuristics with
        usage prior + simple two-strike / DP boosts, blended with pitcher and batter pitch-type whiff hints.
        """
        pid = pitcher.get('mlbid')
        bid = batter.get('mlbid')

        p_usage = (self.pitcher_arsenals.get(pid, {}).get('usage') if pid is not None else None) or self.default_usage_mix
        p_whiff = (self.pitcher_arsenals.get(pid, {}).get('whiff') if pid is not None else None) or {}
        b_whiff = (self.batter_arsenals.get(bid, {}).get('whiff') if bid is not None else None) or {}

        types = list(p_usage.keys())
        weights = np.array([max(1e-6, p_usage[t]) for t in types], dtype=float)
        weights = weights / weights.sum()

        # two-strike boost for high-whiff pitches
        if strikes == 2:
            p_wh = np.array([p_whiff.get(t, pitcher.get('swstr_rate', 0.12)) for t in types], dtype=float)
            mean_pwh = float(np.clip(p_wh.mean(), 0.05, 0.4))
            weights *= np.clip(1.0 + 0.8 * (p_wh - mean_pwh), 0.2, 1.8)

        # batter whiff tendencies by pitch type
        if len(b_whiff) > 0:
            b_wh = np.array([b_whiff.get(t, 0.25) for t in types], dtype=float)
            mean_bwh = float(np.clip(b_wh.mean(), 0.05, 0.5))
            weights *= np.clip(1.0 + 0.6 * (b_wh - mean_bwh), 0.2, 1.8)

        # double play bias to GB-friendly types
        if dp_context:
            gb = np.array([self.pitch_gb_delta.get(t, 0.0) for t in types], dtype=float)
            weights *= np.clip(1.0 + 2.0 * gb, 0.2, 2.0)

        weights = weights / weights.sum()
        choice = self.random_state.choice(types, p=weights)

        zone_mult = float(self.pitch_zone_mult.get(choice, 1.0))
        p_wh_val = p_whiff.get(choice, pitcher.get('swstr_rate', 0.12))
        b_wh_val = b_whiff.get(choice, 0.25 if len(b_whiff) else 0.25)
        whiff_mult = float(np.clip(1.0 + 0.8*(p_wh_val - 0.18) + 0.6*(b_wh_val - 0.25), 0.6, 1.6))
        gb_delta = float(self.pitch_gb_delta.get(choice, 0.0))
        return choice, {'zone_mult': zone_mult, 'whiff_mult': whiff_mult, 'gb_delta': gb_delta}

    def _apply_lr_homeaway_scaling(self, pitcher: Dict[str, Any], batter: Dict[str, Any], in_zone: bool) -> Tuple[float, float]:
        """
        Returns (swstr_mult, gb_mult) based on pitcher/batter splits (L/R & H/A).
        """
        throws = str(pitcher.get("throws", "R") or "R").upper()
        bats = str(batter.get("bats", "R") or "R").upper()

        # Which split to pick (pitcher vs L or vs R)
        vs_key = "vsL" if bats == "L" else "vsR"

        # Start multipliers at 1.0
        swstr_mult = 1.0
        gb_mult = 1.0

        # Pitcher splits
        ps = pitcher.get("splits", {})
        if vs_key in ps and "swstr_rate" in ps[vs_key]:
            swstr_mult *= float(ps[vs_key]["swstr_rate"])
        if vs_key in ps and "gb_rate" in ps[vs_key]:
            gb_mult *= float(ps[vs_key]["gb_rate"])

        # Batter splits (lower contact → higher whiff)
        bs = batter.get("splits", {})
        opp_key = "vsR" if throws == "R" else "vsL"
        if opp_key in bs and "contact_rate" in bs[opp_key]:
            contact_mult = float(bs[opp_key]["contact_rate"])
            # if contact multiplier <1, increase whiff
            swstr_mult *= float(np.clip(1.2 - 0.8 * contact_mult, 0.75, 1.25))

        # Home/away nuance: stack mild H/A multipliers if available
        for tag in ("home", "away"):
            if tag in ps and "swstr_rate" in ps[tag]:
                swstr_mult *= float(np.clip(ps[tag]["swstr_rate"], 0.85, 1.15))
            if tag in ps and "gb_rate" in ps[tag]:
                gb_mult *= float(np.clip(ps[tag]["gb_rate"], 0.85, 1.15))
            if tag in bs and "contact_rate" in bs[tag]:
                swstr_mult *= float(np.clip(1.15 - 0.7 * bs[tag]["contact_rate"], 0.80, 1.20))

        return float(swstr_mult), float(gb_mult)

    def _simulate_plate_appearance_with_count(self, pitcher: Dict[str, Any], batter: Dict[str, Any],
                                              runners_on: bool, dp_context: bool, use_heads: bool) -> Dict[str, Any]:
        balls, strikes, pitches = 0, 0, 0

        BASE_SWING = {
            (0,0): 0.47, (0,1): 0.49, (0,2): 0.63,
            (1,0): 0.45, (1,1): 0.52, (1,2): 0.65,
            (2,0): 0.42, (2,1): 0.50, (2,2): 0.67,
            (3,0): 0.38, (3,1): 0.46, (3,2): 0.70,
        }
        BASE_FOUL = {k: v+0.0 for k, v in {
            (0,0): 0.33, (0,1): 0.34, (0,2): 0.42,
            (1,0): 0.32, (1,1): 0.35, (1,2): 0.43,
            (2,0): 0.31, (2,1): 0.35, (2,2): 0.44,
            (3,0): 0.30, (3,1): 0.34, (3,2): 0.45,
        }.items()}

        def _get(d, b, s, default):
            return d.get((b,s), d.get((max(0,b-1),s), default))

        zone_rate   = _safe_float(pitcher.get('zone_rate', 0.45), 0.45)
        swstr_rate  = _safe_float(pitcher.get('swstr_rate', 0.12), 0.12)
        chase_rate  = _safe_float(pitcher.get('chase_rate', 0.30), 0.30)
        contact     = _safe_float(batter.get('contact_rate', 0.75), 0.75)
        take_batter = batter.get('overall_take_rate', None)

        take_tweak = 1.05 if runners_on else 1.0

        while True:
            pitches += 1

            pitch_type, pitch_params = self._choose_pitch_type(
                pitcher=pitcher, batter=batter, balls=balls, strikes=strikes,
                runners_on=runners_on, dp_context=dp_context
            )
            z_mult   = pitch_params['zone_mult']
            gb_delta = pitch_params['gb_delta']

            # L/R & H/A scale
            swstr_mult_lr, gb_mult_lr = self._apply_lr_homeaway_scaling(pitcher, batter, in_zone=True)

            # in-zone?
            in_zone = (np.random.rand() < np.clip(zone_rate * z_mult, 0.20, 0.85))

            # swing propensity
            base_swing = _get(BASE_SWING, balls, strikes, 0.5)
            if take_batter is not None:
                swing_adj = 1.0 - 0.15 * (take_batter - 0.50)
            else:
                swing_adj = 1.0
            if strikes == 2:
                swing_adj *= 1.20

            if in_zone:
                swing_prob = np.clip(base_swing * swing_adj * 1.05, 0.05, 0.95)
            else:
                swing_prob = np.clip(chase_rate * 0.95 * swing_adj * take_tweak, 0.02, 0.70)

            if np.random.rand() < swing_prob:
                # whiff probability with type & split multipliers
                whiff_base = np.clip( (swstr_rate * (1 - contact)) * 2.2, 0.05, 0.65 )
                whiff_prob = float(np.clip(whiff_base * pitch_params['whiff_mult'] * swstr_mult_lr, 0.03, 0.85))

                if np.random.rand() < whiff_prob:
                    if strikes < 2:
                        strikes += 1
                        continue
                    else:
                        return {'outcome': 'strikeout', 'pitches': pitches}

                foul_rate = _get(BASE_FOUL, balls, strikes, 0.34)
                if strikes == 2:
                    foul_rate = min(0.70, foul_rate + 0.10)
                if np.random.rand() < foul_rate:
                    if strikes < 2:
                        strikes += 1
                    continue

                gb_pitcher = _safe_float(pitcher.get('gb_rate', 0.44), 0.44)
                gb_batter  = _safe_float(batter.get('gb_rate', 0.43), 0.43)
                gb_mix = np.clip(0.5 * gb_pitcher + 0.5 * gb_batter + gb_delta, 0.1, 0.8) * gb_mult_lr

                quality = np.random.rand()
                if quality < 0.72:
                    if np.random.rand() < gb_mix:
                        outcome = 'ground_out'
                    else:
                        outcome = np.random.choice(['fly_out','line_out','pop_out'], p=[0.45,0.35,0.20])
                    return {'outcome': outcome, 'pitches': pitches}
                else:
                    power = _safe_float(batter.get('power_rate', 0.35), 0.35)
                    p_single, p_double, p_triple, p_hr = 0.72, 0.18, 0.03, 0.07
                    p_hr *= (0.85 + 0.9 * (power - 0.35) / 0.25) * (1.0 - 0.4 * max(0.0, gb_delta))
                    total = p_single + p_double + p_triple + p_hr
                    p_single/=total; p_double/=total; p_triple/=total; p_hr/=total
                    outcome = np.random.choice(['single','double','triple','home_run'],
                                               p=[p_single,p_double,p_triple,p_hr])
                    return {'outcome': outcome, 'pitches': pitches}

            else:
                # take
                if in_zone:
                    if strikes < 2:
                        strikes += 1
                        continue
                    else:
                        return {'outcome': 'strikeout', 'pitches': pitches}
                else:
                    if balls < 3:
                        balls += 1
                        continue
                    else:
                        return {'outcome': 'walk', 'pitches': pitches}

    # ------------------------ Base-running helpers ------------------------

    def _advance_walk(self, on1, on2, on3):
        if on1 and on2 and on3:
            return True, True, True
        if on1 and on2:
            return True, True, True
        if on1:
            return True, True, on3
        return True, on2, on3

    def _advance_on_hit(self, bases: int, on1, on2, on3):
        if bases == 1:
            if on3: on3 = False
            if on2 and np.random.rand() < 0.6: on2 = False
            if on1: on2 = True; on1 = False
            on1_new = True
            return on1_new, on2, on3
        if bases == 2:
            on3 = False
            on2 = False if on2 else on2
            if on1: on3 = True; on1 = False
            on2_new = True
            return on1, on2_new, on3
        if bases == 3:
            on1, on2, on3 = False, False, False
            on3_new = True
            return on1, on2, on3_new
        return on1, on2, on3

