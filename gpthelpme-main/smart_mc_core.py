# src/smart_mc_core.py
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------- small helpers -----------------------------

def _safe(v, d=None):
    return v if v is not None and not (isinstance(v, float) and np.isnan(v)) else d

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def _namekey(s: Any) -> str:
    return str(s).strip().lower()

def _try_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


# ----------------------------- config knobs ------------------------------

DEFAULT_VENUE_FACTORS = {
    "coors field": 0.88,
    "fenway park": 0.95,
    "yankee stadium": 1.02,
    "petco park": 1.08,
    "oracle park": 1.05,
    "loandepot park": 1.03,
    "marlins park": 1.03,
    "t-mobile park": 1.02,
    "tropicana field": 1.01,
    "kauffman stadium": 1.04,
    "minute maid park": 0.97,
}

PITCH_ZONE_MULT = {"FF": 1.05, "FC": 1.02, "SI": 1.00, "SL": 0.95, "CH": 0.92, "CU": 0.90, "KC": 0.90, "FS": 0.96}
PITCH_GB_DELTA = {"SI": +0.08, "SL": +0.04, "CH": +0.02, "CU": +0.03, "KC": +0.03, "FC": 0.00, "FS": -0.02, "FF": -0.05}

DEFAULT_USAGE_MIX = {"FF": 0.45, "SL": 0.20, "CH": 0.15, "CU": 0.10, "SI": 0.10}


@dataclass
class HeadModels:
    pitch_select: Any = None
    whiff: Any = None
    bip: Any = None


# --------------------------- main predictor core --------------------------

class SmartMonteCarloPredictor:
    """
    Minimal, dependency-light core:

    - loads models/comprehensive_trained_models.joblib (expects keys: feature_columns, model or models, scaler, calibration)
    - loads models/pitcher_profiles.parquet & models/batter_profiles.parquet (from your build)
    - builds feature vectors **by name** to match training columns exactly
    - sim loop is handedness-aware (uses pitcher & batter splits)
    - optional trained heads: models/pitch_select.joblib, whiff.joblib, bip.joblib

    Use:
      p = SmartMonteCarloPredictor(models_dir="models", use_heads=True)
      res = p.vectorized_pitch_simulation(669923, lineup, venue="Yankee Stadium", simulations=2000)
    """

    def __init__(
        self,
        models_dir: str = "models",
        pitch_count_cap: int = 95,
        rng_seed: int = 42,
        use_heads: bool = False,
        cap_mode: str = "soft",          # "soft" or "hard"
        hook_aggr: float = 1.0           # >1.0 = earlier hook; <1.0 = longer leash
    ):
        self.models_dir = models_dir
        self.pitch_count_cap = int(pitch_count_cap)
        self.cap_mode = str(cap_mode).lower()
        self.hook_aggr = float(hook_aggr)
        self.rng = np.random.default_rng(rng_seed)

        # loaded assets
        self.model = None
        self.models = {}
        self.scaler = None
        self.feature_columns: List[str] = []
        self.training_stats: Dict[str, Any] = {}
        self.calibration: Dict[str, Any] = {}

        self.pitcher_df = None
        self.batter_df = None
        self.pitcher_by_id: Dict[int, Dict[str, Any]] = {}
        self.batter_by_id: Dict[int, Dict[str, Any]] = {}
        self.pitcher_name_to_id: Dict[str, int] = {}
        self.batter_name_to_id: Dict[str, int] = {}

        self.venue_factors = DEFAULT_VENUE_FACTORS.copy()

        # optional heads
        self.heads = HeadModels()
        self.use_heads = bool(use_heads)

        # load everything
        self._load_bundle()
        self._load_profiles_and_splits()
        if self.use_heads:
            self._load_heads()

    # ------------------------ loading ------------------------

    def _load_bundle(self):
        path = os.path.join(self.models_dir, "comprehensive_trained_models.joblib")
        try:
            bundle = joblib.load(path)
        except Exception as e:
            print(f"⚠️ could not load {path}: {e}")
            bundle = {}

        # common keys we use
        self.model = bundle.get("model")  # single estimator
        self.models = bundle.get("models", {})  # optional dict of named estimators / quantiles
        self.scaler = bundle.get("scaler") or bundle.get("standardizer")
        self.feature_columns = list(bundle.get("feature_columns", []))
        self.training_stats = bundle.get("training_stats", {})
        self.calibration = bundle.get("calibration", bundle.get("training_stats", {}).get("calibration", {}))

        if self.feature_columns:
            print(f"✅ Loaded comprehensive bundle: {len(self.feature_columns)} feature columns.")
        else:
            print("⚠️ Bundle missing feature_columns; will fall back to whatever the model expects.")

    def _load_profiles_and_splits(self):
        # Pitchers
        p_path = os.path.join(self.models_dir, "pitcher_profiles.parquet")
        b_path = os.path.join(self.models_dir, "batter_profiles.parquet")

        try:
            self.pitcher_df = pd.read_parquet(p_path)
        except Exception as e:
            print(f"⚠️ pitcher_profiles missing/unreadable ({e}); using empty.")
            self.pitcher_df = pd.DataFrame()

        try:
            self.batter_df = pd.read_parquet(b_path)
        except Exception as e:
            print(f"⚠️ batter_profiles missing/unreadable ({e}); using empty.")
            self.batter_df = pd.DataFrame()

        # fast dict lookups by id
        if not self.pitcher_df.empty and "pitcher" in self.pitcher_df.columns:
            for _, r in self.pitcher_df.iterrows():
                pid = _try_int(r.get("pitcher"))
                if pid is None:
                    continue
                self.pitcher_by_id[pid] = r.to_dict()
                nm = str(r.get("player_name") or "").strip()
                if nm:
                    self.pitcher_name_to_id[_namekey(nm)] = pid

        if not self.batter_df.empty and "batter" in self.batter_df.columns:
            for _, r in self.batter_df.iterrows():
                bid = _try_int(r.get("batter"))
                if bid is None:
                    continue
                self.batter_by_id[bid] = r.to_dict()
                nm = str(r.get("player_name") or "").strip()
                if nm:
                    self.batter_name_to_id[_namekey(nm)] = bid

        print(f"ℹ️ profiles loaded: {len(self.pitcher_by_id)} pitchers, {len(self.batter_by_id)} batters.")

    def _load_heads(self):
        def try_load(name):
            fp = os.path.join(self.models_dir, f"{name}.joblib")
            try:
                return joblib.load(fp)
            except Exception:
                return None

        self.heads.pitch_select = try_load("pitch_select")
        self.heads.whiff = try_load("whiff")
        self.heads.bip = try_load("bip")

        loaded = [k for k, m in [("pitch_select", self.heads.pitch_select),
                                 ("whiff", self.heads.whiff),
                                 ("bip", self.heads.bip)] if m is not None]
        if loaded:
            print("✅ loaded heads:", ", ".join(loaded))
        else:
            print("ℹ️ heads not found; running heuristic sim path")

    # --------------------- ID & profile accessors ---------------------

    def _resolve_pitcher(self, id_or_name) -> Tuple[int, Dict[str, Any]]:
        if isinstance(id_or_name, (int, np.integer)):
            pid = int(id_or_name)
            prof = self.pitcher_by_id.get(pid)
            if prof is not None:
                return pid, prof
            # fabricate league avg
            return pid, {"pitches": 300, "swstr_season": 0.12, "zone_season": 0.45, "contact_season": 0.78,
                         "gb_season": 0.44, "hard_season": 0.35, "p_throws": "R"}
        # name string
        key = _namekey(id_or_name)
        pid = self.pitcher_name_to_id.get(key)
        if pid is not None and pid in self.pitcher_by_id:
            return pid, self.pitcher_by_id[pid]
        fake = -abs(hash(key)) % (10 ** 9)
        return fake, {"pitches": 300, "swstr_season": 0.12, "zone_season": 0.45, "contact_season": 0.78,
                      "gb_season": 0.44, "hard_season": 0.35, "p_throws": "R"}

    def _resolve_batter(self, id_or_name) -> Tuple[int, Dict[str, Any]]:
        if isinstance(id_or_name, (int, np.integer)):
            bid = int(id_or_name)
            prof = self.batter_by_id.get(bid)
            if prof is not None:
                return bid, prof
            return bid, {"k_season": 0.22, "contact_season": 0.76, "gb_season": 0.43, "hard_season": 0.35, "stand": "R"}
        key = _namekey(id_or_name)
        bid = self.batter_name_to_id.get(key)
        if bid is not None and bid in self.batter_by_id:
            return bid, self.batter_by_id[bid]
        fake = -abs(hash(key)) % (10 ** 9)
        return fake, {"k_season": 0.22, "contact_season": 0.76, "gb_season": 0.43, "hard_season": 0.35, "stand": "R"}

    # --------------------- feature vector builder ---------------------

    def _feature_dict(self, pid: int, lineup_bids: List[int], venue: str) -> Dict[str, float]:
        """Collect many possible features; we'll pick/align to joblib's feature_columns later."""
        p = self.pitcher_by_id.get(pid, {})
        p_hand = str(p.get("p_throws") or "R").upper()[:1]
        venue_k = float(self.venue_factors.get(_namekey(venue), 1.0))

        # pitcher overall season & rolling (names from build_profiles.py)
        feat = {}
        # base rates (season)
        feat["p_csw_season"] = _safe(p.get("csw_season"), 0.28)
        feat["p_swstr_season"] = _safe(p.get("swstr_season"), 0.12)
        feat["p_zone_season"] = _safe(p.get("zone_season"), 0.45)
        feat["p_contact_season"] = _safe(p.get("contact_season"), 0.78)
        feat["p_gb_season"] = _safe(p.get("gb_season"), 0.44)
        feat["p_hard_season"] = _safe(p.get("hard_season"), 0.35)

        # 5/15/30 rollings
        for w in (5, 15, 30):
            feat[f"p_csw_rw{w}"] = _safe(p.get(f"csw_rw{w}"), feat["p_csw_season"])
            feat[f"p_swstr_rw{w}"] = _safe(p.get(f"swstr_rw{w}"), feat["p_swstr_season"])
            feat[f"p_zone_rw{w}"] = _safe(p.get(f"zone_rw{w}"), feat["p_zone_season"])
            feat[f"p_contact_rw{w}"] = _safe(p.get(f"contact_rw{w}"), feat["p_contact_season"])
            feat[f"p_gb_rw{w}"] = _safe(p.get(f"gb_rw{w}"), feat["p_gb_season"])
            feat[f"p_hard_rw{w}"] = _safe(p.get(f"hard_rw{w}"), feat["p_hard_season"])

        # splits — pitchers (season-level) produced by build_profiles (labels like vsL, vsR, home, away, combos)
        # we normalize into intuitive keys; if absent, backfill with season.
        def pull(label_prefix: str, base_key: str) -> float:
            # e.g., ("swstr", "vsL") -> swstr_season_vsL
            col = f"{base_key}_season_{label_prefix}"
            return _safe(p.get(col), feat[f"p_{base_key}_season"])

        for side in ("vsL", "vsR"):
            feat[f"p_swstr_{side}"] = pull(side, "swstr")
            feat[f"p_csw_{side}"] = pull(side, "csw")
            feat[f"p_zone_{side}"] = pull(side, "zone")
            feat[f"p_contact_{side}"] = pull(side, "contact")
        for ha in ("home", "away"):
            feat[f"p_swstr_{ha}"] = pull(ha, "swstr")
            feat[f"p_csw_{ha}"] = pull(ha, "csw")

        # lineup aggregates
        opp_k = []
        opp_contact = []
        opp_k_vs_hand = []
        opp_contact_vs_hand = []
        for bid in lineup_bids:
            b = self.batter_by_id.get(bid, {})
            b_hand = str(b.get("stand") or "R").upper()[:1]
            # generic
            opp_k.append(_safe(b.get("k_season"), 0.22))
            opp_contact.append(_safe(b.get("contact_season"), 0.76))
            # vs specific pitcher hand if present (build_profiles creates swstr/contact splits; k_season may not be split)
            # approximate k from (1 - contact) + whiff-ish; still useful signal if present
            tag = "vsR" if p_hand == "R" else "vsL"
            contact_vs = _safe(b.get(f"contact_season_{tag}"), b.get("contact_season"))
            swstr_vs = _safe(b.get(f"swstr_season_{tag}"), None)
            k_est = _safe(b.get("k_season"), None)
            if k_est is None:
                # rough estimate: higher swstr -> higher K; lower contact -> higher K
                k_est = _clip01(0.40 * (swstr_vs if swstr_vs is not None else 0.12) + 0.60 * (1 - (contact_vs if contact_vs is not None else 0.76)))
            opp_k_vs_hand.append(float(k_est))
            opp_contact_vs_hand.append(float(_safe(contact_vs, 0.76)))

        def safe_mean(arr, d):
            return float(np.mean(arr)) if arr else d

        def safe_std(arr):
            return float(np.std(arr)) if len(arr) > 1 else 0.0

        feat["opp_k_mean"] = safe_mean(opp_k, 0.22)
        feat["opp_k_std"] = safe_std(opp_k)
        feat["opp_contact_mean"] = safe_mean(opp_contact, 0.76)
        feat["opp_contact_std"] = safe_std(opp_contact)
        feat["opp_k_vs_hand_mean"] = safe_mean(opp_k_vs_hand, feat["opp_k_mean"])
        feat["opp_contact_vs_hand_mean"] = safe_mean(opp_contact_vs_hand, feat["opp_contact_mean"])
        feat["opp_n_hiK"] = float(sum(1 for x in opp_k if x >= 0.27))
        feat["opp_n_loK"] = float(sum(1 for x in opp_k if x <= 0.18))

        feat["venue_k_mult"] = venue_k

        # generic fallbacks for any other numeric features the model might expect
        return {k: (0.0 if v is None else float(v)) for k, v in feat.items()}

    def _align_vector(self, feat_dict: Dict[str, float]) -> np.ndarray:
        """Align to self.feature_columns; missing -> 0.0"""
        if not self.feature_columns:
            # If the bundle didn't store names, just pass the dict values in stable order
            arr = np.array(list(feat_dict.values()), dtype=float).reshape(1, -1)
            return arr
        vec = [feat_dict.get(col, 0.0) for col in self.feature_columns]
        arr = np.array(vec, dtype=float).reshape(1, -1)
        return arr

    def _predict_ml_base(self, pid: int, lineup_bids: List[int], venue: str) -> float:
        feat = self._feature_dict(pid, lineup_bids, venue)
        X = self._align_vector(feat)
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print("   scaler error:", e)

        preds = []
        if self.model is not None:
            try:
                preds.append(float(self.model.predict(X)[0]))
            except Exception as e:
                print("   model error:", e)
        for name, mdl in self.models.items():
            try:
                preds.append(float(mdl.predict(X)[0]))
            except Exception:
                pass

        if not preds:
            # trivial fallback: K ~= 25 * pitcher season k-rate approximation from swstr/contact
            p = self.pitcher_by_id.get(pid, {})
            sw = _safe(p.get("swstr_season"), 0.12)
            ct = _safe(p.get("contact_season"), 0.78)
            est_k_rate = _clip01(0.55 * sw * 1.8 + 0.45 * (1 - ct))
            return 25.0 * est_k_rate

        base = float(np.mean(preds))
        return self._calibrate(base)

    def _calibrate(self, yhat: float) -> float:
        cal = self.calibration or {}
        ctype = str(cal.get("type", "linear")).lower()
        if ctype == "linear":
            slope = float(cal.get("slope", 1.0))
            intercept = float(cal.get("intercept", 0.0))
            return float(slope * yhat + intercept)
        # identity / unknown
        return float(yhat)

    # -------------------------- public API --------------------------

    def vectorized_pitch_simulation(
        self,
        pitcher_id_or_name: Any,
        opposing_lineup: List[Any],
        venue: str = "neutral",
        simulations: int = 3000,
    ) -> Dict[str, Any]:
        pid, pprof = self._resolve_pitcher(pitcher_id_or_name)

        # lineup → IDs
        lineup_ids: List[int] = []
        lineup_profiles: List[Dict[str, Any]] = []
        for b in opposing_lineup:
            if isinstance(b, dict):
                ident = b.get("mlbid") if b.get("mlbid") is not None else (b.get("name") or b.get("player_name"))
            else:
                ident = b
            bid, bprof = self._resolve_batter(ident)
            lineup_ids.append(bid)
            lineup_profiles.append(bprof)

        ml_base = self._predict_ml_base(pid, lineup_ids, venue)

        # sim
        dist = self._run_sims(pprof, lineup_profiles, venue, n=int(simulations))
        # shift to match ML mean
        if len(dist) > 0:
            dist = dist + (ml_base - float(np.mean(dist)))

        # summary
        mean = float(np.mean(dist)) if len(dist) else float("nan")
        pcts = np.percentile(dist, [10, 25, 50, 75, 90]) if len(dist) else [np.nan] * 5
        p10, p25, p50, p75, p90 = map(float, pcts)
        probs = {f"prob_over_{k:.1f}": float(np.mean(dist >= k)) for k in (4.5, 5.5, 6.5, 7.5, 8.5)} if len(dist) else {}

        grade = self._grade_matchup(mean, probs)
        conf = self._confidence(dist)

        return {
            "pitcher_mlbid": pid,
            "ml_base_prediction": float(ml_base),
            "final_prediction": float(mean),
            "median": p50,
            "percentile_10": p10,
            "percentile_25": p25,
            "percentile_75": p75,
            "percentile_90": p90,
            "std_dev": float(np.std(dist)) if len(dist) else float("nan"),
            **probs,
            "matchup_grade": grade,
            "model_confidence": conf,
            "distribution": [float(x) for x in dist],
            "simulations_run": int(simulations),
        }

    # ----------------------- simulation internals -----------------------

    def _run_sims(self, p: Dict[str, Any], lineup: List[Dict[str, Any]], venue: str, n: int) -> np.ndarray:
        """Simple PA-level sim; handedness-aware whiff/contact; pitch-count cap with soft/hard modes."""
        if not lineup:
            lineup = [{"stand": "R"} for _ in range(9)]

        p_hand = str(p.get("p_throws") or "R").upper()[:1]
        venue_mult = float(self.venue_factors.get(_namekey(venue), 1.0))

        ks = np.zeros(n, dtype=float)
        for i in range(n):
            outs = 0
            ip = 0
            pitch_count = 0
            batter_idx = 0
            k_this_game = 0

            on1 = on2 = on3 = False
            while outs < 27 and ip < 9 and pitch_count < self.pitch_count_cap:
                b = lineup[batter_idx % len(lineup)]
                batter_idx += 1
                # derive batter vs-hand contact/K approximations
                tag = "vsR" if p_hand == "R" else "vsL"
                b_ct = _safe(b.get(f"contact_season_{tag}"), b.get("contact_season"))
                b_ct = _safe(b_ct, 0.76)
                b_sw = _safe(b.get(f"swstr_season_{tag}"), None)

                # pitcher vs-hand swstr/zone/contact
                p_sw = _safe(p.get(f"swstr_season_{tag}"), p.get("swstr_season"))
                p_zn = _safe(p.get(f"zone_season_{tag}"), p.get("zone_season"))
                p_ct = _safe(p.get(f"contact_season_{tag}"), p.get("contact_season"))

                # base swing/whiff model influenced by pitcher/batter
                # whiff baseline ~ pitcher swstr × (1 - batter contact)
                base_whiff = _clip01((p_sw if p_sw is not None else 0.12) * (1 - b_ct) * 2.0)
                # venue slightly nudges whiff up/down
                base_whiff = _clip01(base_whiff * (0.9 + 0.2 * (venue_mult - 1.0)))

                # simulate a PA crudely: chance of K vs ball-in-play/walk
                # increase called strikes from zone rate; reduce if high contact
                z_effect = _clip01((p_zn if p_zn is not None else 0.45) * (1.1 - 0.3 * (b_ct - 0.75)))
                # probability of K across the PA (rough but stable)
                p_strikeout = _clip01(0.35 * base_whiff + 0.25 * (1 - b_ct) + 0.15 * z_effect)

                # walk probability (damp with zone, bump with low contact)
                p_walk = _clip01(0.08 * (1.05 - z_effect) * (1.05 - (1 - b_ct)))

                # BIP → out or hit (GB bias from pitcher+pitch type family)
                gb = _safe(p.get("gb_season"), 0.44)
                p_out_on_bip = _clip01(0.70 + 0.15 * (gb - 0.44))  # more GB -> more outs on balls in play

                r = self.rng.random()
                if r < p_strikeout:
                    k_this_game += 1
                    outs += 1
                elif r < p_strikeout + p_walk:
                    # walk → pitch count +, base advance simple
                    pass
                else:
                    # BIP
                    if self.rng.random() < p_out_on_bip:
                        outs += 1
                    # else it's a hit; ignore base state complexity for K counting

                # pitch count ~ 3.8 to 4.5 per PA depending on outcome type
                pitch_count += int(3 + self.rng.integers(1, 4))
                # soft/hard cap
                if self.cap_mode == "soft":
                    # stamina penalty: exit earlier if hook_aggr is high
                    if pitch_count > 70 + 25 / max(self.hook_aggr, 0.25):
                        if self.rng.random() < 0.04 * self.hook_aggr:
                            break
                # hard checked at while condition

                if outs and outs % 3 == 0:
                    ip += 1

            ks[i] = k_this_game

        return ks

    # ----------------------- grade & confidence -----------------------

    def _grade_matchup(self, mean_ks: float, probs: Dict[str, float]) -> str:
        """simple letter grade: combine mean and prob_over ladders"""
        if not np.isfinite(mean_ks):
            return "N/A"
        score = (
            0.50 * (mean_ks / 10.0) +
            0.12 * probs.get("prob_over_4_5", 0.0) +
            0.18 * probs.get("prob_over_6_5", 0.0) +
            0.20 * probs.get("prob_over_8_5", 0.0)
        )
        if score >= 0.90: return "A+"
        if score >= 0.80: return "A"
        if score >= 0.70: return "A-"
        if score >= 0.62: return "B+"
        if score >= 0.55: return "B"
        if score >= 0.48: return "B-"
        if score >= 0.40: return "C+"
        if score >= 0.33: return "C"
        if score >= 0.26: return "C-"
        if score >= 0.18: return "D"
        return "F"

    def _confidence(self, dist: np.ndarray) -> float:
        """0..1 heuristic based on dispersion & sample size."""
        if len(dist) < 50:
            return 0.35
        sd = float(np.std(dist))
        # tighter distributions → higher confidence
        conf = 1.0 / (1.0 + sd / 2.5)
        # cap to 0.15..0.95
        return float(np.clip(conf, 0.15, 0.95))


# ----------------------------- batch helper ------------------------------

def export_batch_to_csv(
    predictor: SmartMonteCarloPredictor,
    games: Iterable[Dict[str, Any]],
    out_path: str,
    n_sims_default: int = 3000,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for g in games:
        lineup = g.get("lineup") or g.get("lineup_mlbids") or g.get("batters") or [{"name": "League Avg Batter"} for _ in range(9)]
        venue = g.get("venue") or g.get("park") or g.get("stadium") or "neutral"
        ident = g.get("pitcher") or g.get("pitcher_mlbid") or g.get("pitcher_name")
        n_sims = int(g.get("n_sims", n_sims_default))
        if ident is None:
            continue
        try:
            res = predictor.vectorized_pitch_simulation(ident, lineup, venue=venue, simulations=n_sims)
            rows.append({
                "MLBID": res.get("pitcher_mlbid"),
                "Pitcher": g.get("pitcher_name") or g.get("pitcher"),
                "Team": g.get("team"),
                "Opponent": g.get("opponent") or g.get("lineup_label"),
                "Venue": venue,
                "ML_Prediction": res.get("ml_base_prediction"),
                "Smart_MC": res.get("final_prediction"),
                "P10": res.get("percentile_10"),
                "P25": res.get("percentile_25"),
                "P50": res.get("median"),
                "P75": res.get("percentile_75"),
                "P90": res.get("percentile_90"),
                "Std": res.get("std_dev"),
                "Over4.5": res.get("prob_over_4_5"),
                "Over5.5": res.get("prob_over_5_5"),
                "Over6.5": res.get("prob_over_6_5"),
                "Over7.5": res.get("prob_over_7_5"),
                "Over8.5": res.get("prob_over_8_5"),
                "Matchup_Grade": res.get("matchup_grade"),
                "Confidence": res.get("model_confidence"),
            })
        except Exception as e:
            rows.append({"Pitcher": g.get("pitcher") or g.get("pitcher_name"),
                         "Team": g.get("team"), "Opponent": g.get("opponent"),
                         "Error": f"{type(e).__name__}: {e}"})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(df)} rows → {out_path}")
    return df
