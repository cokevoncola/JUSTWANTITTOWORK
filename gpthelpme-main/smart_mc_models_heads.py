from __future__ import annotations

import os, json, math, warnings
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ------------------------------ Utilities ------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return float(default)

def _onehot(value: str) -> str:
    s = str(value or "").strip()
    return s if s else "UNK"

def _base_state_tuple(on1: bool, on2: bool, on3: bool) -> str:
    return f"{int(bool(on1))}{int(bool(on2))}{int(bool(on3))}"

def _bucket(v: float, bins: List[float], labels: List[str]) -> str:
    for b, lab in zip(bins, labels):
        if v <= b:
            return lab
    return labels[-1]


# ------------------------------ Base Heuristic MC ------------------------------
# We import the previous heuristic class and extend it.
# If you changed its path/name, adjust this import.
from .smart_mc_models import SmartMonteCarloPredictor as HeuristicMC  # type: ignore


# ------------------------- Heads-Driven Smart MC -------------------------

class SmartMonteCarloPredictorHeads(HeuristicMC):
    """
    Smart MC that calls the three trained heads:
      • pitch_select.joblib  -> P(pitch_label | context)
      • whiff.joblib         -> P(whiff | context + chosen pitch)
      • bip.joblib           -> P(BIP class | context + chosen pitch)  where classes ~ ['FB','GB','LD','PU']

    Also includes:
      • L/R split awareness (from pitcher/batter profiles parquet)
      • pitch-count: soft/hard hooks (inherited)
      • model confidence tracking across simulated decisions
    """

    def __init__(
        self,
        pitch_count_cap: int = 95,
        random_seed: int = 42,
        cap_mode: str = "soft",
        hook_aggressiveness: float = 1.0,
    ):
        super().__init__(
            pitch_count_cap=pitch_count_cap,
            random_seed=random_seed,
            cap_mode=cap_mode,
            hook_aggressiveness=hook_aggressiveness,
        )
        # load heads (robust to different saving layouts)
        self.ps_model, self.ps_features, self.ps_classes = self._load_head("models/pitch_select.joblib")
        self.wh_model, self.wh_features, self.wh_classes = self._load_head("models/whiff.joblib")
        self.bip_model, self.bip_features, self.bip_classes = self._load_head("models/bip.joblib")

        # confidence trackers (reset per simulation call)
        self._conf_buffers_reset()

    # ---------------------- model loading & helpers ----------------------

    def _conf_buffers_reset(self):
        self._pitchselect_maxps: List[float] = []
        self._whiff_binconf: List[float] = []     # 2*|p-0.5|
        self._bip_maxps_entropy: List[float] = [] # 1 - H(p)/log(k)

    def _load_head(self, path: str):
        """
        Returns (estimator_or_pipeline, feature_names or None, classes or None).
        Accepts:
          - a sklearn Pipeline/Estimator with .predict_proba and .classes_
          - a dict with keys like {'model': <estimator>, 'features': [...], 'classes': [...]}
        """
        try:
            obj = joblib.load(path)
        except Exception as e:
            print(f"ℹ️ Could not load {path}: {e}")
            return None, None, None

        est, feats, classes = obj, None, None
        if isinstance(obj, dict):
            est = obj.get("model") or obj.get("pipe") or obj.get("estimator") or obj
            feats = obj.get("features") or obj.get("feature_columns")
            classes = obj.get("classes")

        # Try to infer classes if not present
        if classes is None:
            try:
                classes = getattr(est, "classes_", None)
            except Exception:
                classes = None

        # Normalize classes to list of strings
        if classes is not None:
            classes = [str(c) for c in list(classes)]

        # Try to infer feature names
        if feats is None:
            feats = getattr(est, "feature_names_in_", None)
            if feats is not None:
                feats = list(map(str, feats))

        print(f"✅ Loaded head {os.path.basename(path)}"
              f"{' with features' if feats else ''}{' and classes' if classes is not None else ''}")
        return est, feats, classes

    def _align_columns(self, df: pd.DataFrame, required: Optional[List[str]]) -> pd.DataFrame:
        if not required:
            return df
        out = df.copy()
        for col in required:
            if col not in out.columns:
                # Try to fill reasonable defaults
                if col in ("balls","strikes","outs","inning","tto","pitch_count"):
                    out[col] = 0
                elif "base" in col:
                    out[col] = "000"
                elif col in ("stand","p_throws","home_away","prev_pitch","pitch_label"):
                    out[col] = "UNK"
                elif col.endswith("_eff") or col.endswith("_rate"):
                    out[col] = 0.0
                else:
                    out[col] = 0
        # Keep only expected columns order if desired
        return out[required] if set(required).issubset(out.columns) else out

    # ----------------- context feature builder (single row) -----------------

    def _context_row(
        self,
        pitcher: Dict[str, Any],
        batter: Dict[str, Any],
        *,
        balls: int,
        strikes: int,
        outs: int,
        inning: int,
        on1: bool,
        on2: bool,
        on3: bool,
        prev_pitch: str,
        tto: float,
        pitch_count: int,
        home_away: str = "A",
        venue: str = "neutral",
        chosen_pitch: Optional[str] = None,
    ) -> pd.DataFrame:
        # effective (L/R-aware) core rates
        zone_eff, swstr_eff, contact_eff, gb_p_eff, gb_b_eff = self._lr_effective(pitcher, batter)

        # categorical bits
        stand = str(batter.get("bats") or batter.get("stand") or "R").upper()[:1] or "R"
        pthrows = str(pitcher.get("throws") or pitcher.get("p_throws") or "R").upper()[:1] or "R"
        if stand not in ("L","R"): stand = "R"
        if pthrows not in ("L","R"): pthrows = "R"

        base_state = _base_state_tuple(on1, on2, on3)
        runners_on = int(on1 or on2 or on3)
        dp_context = int(on1 and outs <= 1)

        row = {
            # game/count
            "balls": int(balls), "strikes": int(strikes),
            "outs": int(outs), "inning": int(inning),
            "base_state": base_state, "runners_on": runners_on, "dp_context": dp_context,
            "prev_pitch": _onehot(prev_pitch),
            "tto": float(tto),
            "pitch_count": int(pitch_count),
            "pitch_count_bucket": _bucket(pitch_count, [59,79,99], ["lt60","60_79","80_99","100p"]),
            "home_away": _onehot(home_away or "A"),
            # identities / handedness
            "stand": stand, "p_throws": pthrows,
            "pid": int(pitcher.get("mlbid") or -1),
            "bid": int(batter.get("mlbid") or -1),
            # effective core rates
            "zone_eff": float(zone_eff), "swstr_eff": float(swstr_eff),
            "contact_eff": float(contact_eff),
            "gb_p_eff": float(gb_p_eff), "gb_b_eff": float(gb_b_eff),
            # raw-ish rates (could be in training)
            "zone_rate": float(pitcher.get("zone_rate", 0.45)),
            "swstr_rate": float(pitcher.get("swstr_rate", 0.12)),
            "contact_rate": float(batter.get("contact_rate", 0.75)),
            "gb_rate_p": float(pitcher.get("gb_rate", 0.44)),
            "gb_rate_b": float(batter.get("gb_rate", 0.43)),
            "venue": _onehot(venue),
        }
        if chosen_pitch is not None:
            row["pitch_label"] = _onehot(chosen_pitch)

        return pd.DataFrame([row])

    # ----------------- overrides: pitch selection & PA logic -----------------

    def _choose_pitch_type(self, pitcher: Dict[str, Any], batter: Dict[str, Any],
                           balls: int, strikes: int, runners_on: bool, dp_context: bool,
                           *, outs: int, inning: int, on1: bool, on2: bool, on3: bool,
                           prev_pitch: str, tto: float, pitch_count: int, home_away: str, venue: str):
        """
        Heads-driven pitch select. Falls back to heuristic if head isn't available.
        Returns (choice, params) where params contains zone_mult, whiff_mult, gb_delta (for downstream).
        """
        # If no pitch-select model, use heuristic from base class
        if self.ps_model is None or self.ps_classes is None:
            return super()._choose_pitch_type(pitcher, batter, balls, strikes, runners_on, dp_context)

        ctx = self._context_row(
            pitcher, batter,
            balls=balls, strikes=strikes,
            outs=outs, inning=inning,
            on1=on1, on2=on2, on3=on3,
            prev_pitch=prev_pitch, tto=tto,
            pitch_count=pitch_count,
            home_away=home_away, venue=venue
        )
        ctx = self._align_columns(ctx, self.ps_features)

        # Predict probabilities
        try:
            proba = self.ps_model.predict_proba(ctx)
            # sklearn pipelines return shape (1, n_classes)
            probs = np.asarray(proba)[0]
            # Track confidence (max prob)
            self._pitchselect_maxps.append(float(np.max(probs)))
            # Map classes to probs
            class_to_p = {str(c): float(p) for c, p in zip(self.ps_classes, probs)}
            # Normalize to a common pitch label set we use in sim
            types = list(class_to_p.keys())
            weights = np.array([max(1e-9, class_to_p[t]) for t in types], dtype=float)
            weights /= weights.sum()
            choice = np.random.choice(types, p=weights)
        except Exception as e:
            # robust fallback
            return super()._choose_pitch_type(pitcher, batter, balls, strikes, runners_on, dp_context)

        # down-stream tuning knobs (keep same as heuristic for stability)
        zone_mult = float(self.pitch_zone_mult.get(choice, 1.0))
        gb_delta  = float(self.pitch_gb_delta.get(choice, 0.0))

        # If we have arsenal whiff map, use it to craft a whiff multiplier baseline
        pid = pitcher.get("mlbid")
        b_whiff = (self.batter_arsenals.get(batter.get("mlbid"), {}).get("whiff") if batter.get("mlbid") else {}) or {}
        p_whiff = (self.pitcher_arsenals.get(pid, {}).get("whiff") if pid else {}) or {}
        p_wh_val = p_whiff.get(choice, pitcher.get("swstr_rate", 0.12))
        b_wh_val = b_whiff.get(choice, 0.25 if len(b_whiff) else 0.25)
        whiff_mult = float(np.clip(1.0 + 0.8*(p_wh_val - 0.18) + 0.6*(b_wh_val - 0.25), 0.6, 1.6))

        return choice, {'zone_mult': zone_mult, 'whiff_mult': whiff_mult, 'gb_delta': gb_delta}

    def _simulate_plate_appearance_with_count(self, pitcher: Dict[str, Any], batter: Dict[str, Any],
                                              runners_on: bool, dp_context: bool,
                                              *, outs: int, inning: int, on1: bool, on2: bool, on3: bool,
                                              prev_pitch: str, tto: float, pitch_count: int, home_away: str, venue: str) -> Dict[str, Any]:
        """
        Same shape as base method, but:
          • pitch selection from head
          • whiff probability from head
          • contact BIP class from head
        """
        balls, strikes, pitches = 0, 0, 0

        # Count-state reference (kept from heuristic for swing/foul cadence)
        BASE_SWING = {(0,0):0.47,(0,1):0.49,(0,2):0.63,(1,0):0.45,(1,1):0.52,(1,2):0.65,(2,0):0.42,(2,1):0.50,(2,2):0.67,(3,0):0.38,(3,1):0.46,(3,2):0.70}
        BASE_FOUL  = {(0,0):0.33,(0,1):0.34,(0,2):0.42,(1,0):0.32,(1,1):0.35,(1,2):0.43,(2,0):0.31,(2,1):0.35,(2,2):0.44,(3,0):0.30,(3,1):0.34,(3,2):0.45}
        def _get(d, b, s, default): return d.get((b,s), d.get((max(0,b-1),s), default))

        # L/R-adjusted core rates
        zone_rate, swstr_rate, contact, gb_pitcher_eff, gb_batter_eff = self._lr_effective(pitcher, batter)
        chase_rate = float(pitcher.get('chase_rate', 0.30))
        take_batter = batter.get('overall_take_rate', None)
        take_tweak = 1.05 if runners_on else 1.0

        current_prev_pitch = prev_pitch

        while True:
            pitches += 1

            # --- choose pitch via head ---
            pitch_choice, pitch_params = self._choose_pitch_type(
                pitcher, batter, balls, strikes, runners_on, dp_context,
                outs=outs, inning=inning, on1=on1, on2=on2, on3=on3,
                prev_pitch=current_prev_pitch, tto=tto, pitch_count=pitch_count, home_away=home_away, venue=venue
            )
            z_mult   = pitch_params['zone_mult']
            gb_delta = pitch_params['gb_delta']

            # Build context row (for whiff/BIP heads)
            base_ctx = self._context_row(
                pitcher, batter,
                balls=balls, strikes=strikes, outs=outs, inning=inning,
                on1=on1, on2=on2, on3=on3,
                prev_pitch=current_prev_pitch, tto=tto, pitch_count=pitch_count,
                home_away=home_away, venue=venue, chosen_pitch=pitch_choice
            )

            # --- zone or not (kept from heuristic to govern take/swing) ---
            in_zone = (np.random.rand() < np.clip(zone_rate * z_mult, 0.2, 0.8))
            base_swing = _get(BASE_SWING, balls, strikes, 0.5)

            swing_adj = 1.0
            if take_batter is not None:
                swing_adj *= (1.0 - 0.15 * (take_batter - 0.50))
            if strikes == 2:
                swing_adj *= 1.25
            swing_prob = np.clip((base_swing * swing_adj * 1.05) if in_zone else (chase_rate * 0.95 * swing_adj * take_tweak), 0.02, 0.95)

            # --- TAKE ---
            if np.random.rand() >= swing_prob:
                # take: zone -> strike; out of zone -> ball
                if in_zone:
                    if strikes < 2:
                        strikes += 1
                        current_prev_pitch = pitch_choice
                        continue
                    else:
                        return {'outcome': 'strikeout', 'pitches': pitches}
                else:
                    if balls < 3:
                        balls += 1
                        current_prev_pitch = pitch_choice
                        continue
                    else:
                        return {'outcome': 'walk', 'pitches': pitches}

            # --- SWING: WHIFF vs CONTACT via head ---
            whiff_p = None
            if self.wh_model is not None:
                row_wh = self._align_columns(base_ctx.copy(), self.wh_features)
                try:
                    # Classifier (preferred): get P(1)
                    if hasattr(self.wh_model, "predict_proba"):
                        p = np.asarray(self.wh_model.predict_proba(row_wh))[0]
                        # if classes are [0,1] find index of '1'
                        if self.wh_classes is not None and len(self.wh_classes) == p.shape[0]:
                            if "1" in self.wh_classes:
                                whiff_p = float(p[self.wh_classes.index("1")])
                            elif 1 in self.wh_classes:
                                whiff_p = float(p[list(self.wh_classes).index(1)])
                            else:
                                # fallback: assume the higher-index is "1"
                                whiff_p = float(p[-1])
                        else:
                            whiff_p = float(p[-1])
                    else:
                        # Regressor: clip to [0,1]
                        y = float(self.wh_model.predict(row_wh)[0])
                        whiff_p = float(np.clip(y, 0.0, 1.0))
                except Exception:
                    whiff_p = None

            if whiff_p is None:
                # fallback heuristic whiff prob
                whiff_prob_base = np.clip((swstr_rate * (1 - contact)) * 2.2, 0.05, 0.65)
                whiff_p = float(np.clip(whiff_prob_base, 0.03, 0.85))

            # confidence contribution for whiff (binary): 2*|p-0.5|
            self._whiff_binconf.append(float(2.0 * abs(whiff_p - 0.5)))

            if np.random.rand() < whiff_p:
                if strikes < 2:
                    strikes += 1
                    current_prev_pitch = pitch_choice
                    continue
                else:
                    return {'outcome': 'strikeout', 'pitches': pitches}

            # --- FOUL check stays heuristic (keeps count life-like) ---
            BASE_FOUL  = {(0,0):0.33,(0,1):0.34,(0,2):0.42,(1,0):0.32,(1,1):0.35,(1,2):0.43,(2,0):0.31,(2,1):0.35,(2,2):0.44,(3,0):0.30,(3,1):0.34,(3,2):0.45}
            foul_rate = _get(BASE_FOUL, balls, strikes, 0.34)
            if strikes == 2:
                foul_rate = min(0.70, foul_rate + 0.10)
            if np.random.rand() < foul_rate:
                if strikes < 2:
                    strikes += 1
                current_prev_pitch = pitch_choice
                continue

            # --- CONTACT: use BIP head for shape ---
            bip_probs = None
            bip_classes = self.bip_classes or ["FB","GB","LD","PU"]
            if self.bip_model is not None:
                row_bip = self._align_columns(base_ctx.copy(), self.bip_features)
                try:
                    p = np.asarray(self.bip_model.predict_proba(row_bip))[0]
                    bip_probs = np.array([p[list(self.bip_classes).index(c)] if c in self.bip_classes else 1e-9
                                          for c in bip_classes], dtype=float)
                except Exception:
                    bip_probs = None
            if bip_probs is None:
                # fallback simple distribution
                bip_probs = np.array([0.36, 0.42, 0.17, 0.05], dtype=float)  # FB, GB, LD, PU
            bip_probs = np.clip(bip_probs, 1e-9, None); bip_probs = bip_probs / bip_probs.sum()

            # confidence contribution for BIP (multi): 1 - H(p)/log(k)
            H = float(-np.sum(bip_probs * np.log(bip_probs)))
            conf_bip = 1.0 - H / math.log(len(bip_probs))
            self._bip_maxps_entropy.append(float(conf_bip))

            bip_choice = str(np.random.choice(bip_classes, p=bip_probs))

            # turn BIP class + gb/fb tendency into OUT vs HIT (simplified)
            gb_pitcher = gb_pitcher_eff
            gb_batter  = gb_batter_eff
            gb_mix = np.clip(0.5 * gb_pitcher + 0.5 * gb_batter + (0.06 if bip_choice == "GB" else (-0.04 if bip_choice in ("FB","PU") else 0.0)), 0.1, 0.85)

            # "quality" of contact (crude proxy)
            quality = np.random.rand()
            if quality < 0.72:
                if bip_choice == "GB":
                    return {'outcome': 'ground_out', 'pitches': pitches}
                else:
                    # air out flavors
                    outcome = np.random.choice(['fly_out','line_out','pop_out'], p=[0.50,0.35,0.15])
                    return {'outcome': outcome, 'pitches': pitches}
            else:
                # Hit type mix skewed by BIP class
                if bip_choice == "GB":
                    p_single, p_double, p_triple, p_hr = 0.84, 0.12, 0.01, 0.03
                elif bip_choice == "LD":
                    p_single, p_double, p_triple, p_hr = 0.58, 0.27, 0.03, 0.12
                elif bip_choice == "PU":
                    p_single, p_double, p_triple, p_hr = 0.35, 0.18, 0.02, 0.45
                else:  # FB
                    p_single, p_double, p_triple, p_hr = 0.48, 0.26, 0.03, 0.23
                arr = np.array([p_single, p_double, p_triple, p_hr], dtype=float)
                arr /= arr.sum()
                outcome = np.random.choice(['single','double','triple','home_run'], p=arr)
                return {'outcome': outcome, 'pitches': pitches}

    # ----------------------- Simulation (override to feed heads) -----------------------

    def _simulate_game(self, pitcher: Dict[str, Any], lineup: List[Dict[str, Any]], venue: str) -> Dict[str, Any]:
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

        # hazard for soft hook
        hazard = 0.0

        # cache home/away if known (we don't know here; leave "A")
        home_away = "A"

        # "previous pitch" memory per PA
        prev_pitch = "NONE"

        while outs < 27 and inning <= 9:
            # hard/soft caps
            if self.cap_mode == "hard" and pitch_count >= self.pitch_count_cap:
                break
            if self.cap_mode == "soft" and pitch_count >= 75:
                if self._should_hook(pitch_count, inning, stamina, hazard):
                    break
            if stamina <= 0.25:
                break

            i = lineup_pos % len(lineup)
            batter = lineup[i]

            dp_context = bool(on1 and outs <= 1)
            tto = (lineup_pos // len(lineup)) + 1  # 1st, 2nd, 3rd time through
            pa = self._simulate_plate_appearance_with_count(
                pitcher=pitcher_eff, batter=batter, runners_on=(on1 or on2 or on3), dp_context=dp_context,
                outs=outs, inning=inning, on1=on1, on2=on2, on3=on3,
                prev_pitch=prev_pitch, tto=float(tto),
                pitch_count=pitch_count, home_away=home_away, venue=venue
            )
            pitch_count += pa['pitches']
            stamina = self._stamina_curve(pitch_count, inning, pitcher)

            outcome = pa['outcome']

            # update hazard
            if outcome in ('walk', 'hbp', 'single', 'double', 'triple', 'home_run'):
                hazard = min(1.0, hazard + 0.12)
            else:
                hazard = max(0.0, hazard - 0.06)

            # handle outcome (same as heuristic)
            if outcome == 'strikeout':
                total_ks += 1
                batter_ks[i] += 1
                outs += 1
                prev_pitch = "K"
            elif outcome in ('walk', 'hbp'):
                on1, on2, on3 = self._advance_walk(on1, on2, on3)
                prev_pitch = "BB"
            elif outcome == 'ground_out':
                if on1 and outs <= 1:
                    dp_chance = 0.35 * (0.9 + 0.4 * (pitcher.get('gb_rate', 0.44) - 0.44)/0.2)
                    if self.random_state.rand() < dp_chance:
                        outs += 2; on1 = False
                    else:
                        outs += 1
                        if self.random_state.rand() < 0.3 and on1:
                            if not on2: on2 = True
                            on1 = False
                else:
                    outs += 1
                    if on3 and self.random_state.rand() < 0.2:  on3 = False
                    if on2 and self.random_state.rand() < 0.15: on3 = True; on2 = False
                    if on1 and self.random_state.rand() < 0.15: on2 = True; on1 = False
                prev_pitch = "GO"
            elif outcome in ('fly_out', 'line_out', 'pop_out'):
                if outs <= 1 and outcome == 'fly_out' and on3 and self.random_state.rand() < 0.3:
                    on3 = False
                outs += 1
                prev_pitch = "AO"
            elif outcome == 'single':
                on1, on2, on3 = self._advance_on_hit(1, on1, on2, on3); prev_pitch = "1B"
            elif outcome == 'double':
                on1, on2, on3 = self._advance_on_hit(2, on1, on2, on3); prev_pitch = "2B"
            elif outcome == 'triple':
                on1, on2, on3 = self._advance_on_hit(3, on1, on2, on3); prev_pitch = "3B"
            elif outcome == 'home_run':
                on1, on2, on3 = False, False, False; prev_pitch = "HR"

            lineup_pos += 1
            if outs >= inning * 3:
                inning += 1
                prev_pitch = "NONE"  # reset PA memory next inning

        return {
            'total_ks': total_ks,
            'batter_ks': batter_ks,
            'innings_pitched': min(inning - 1, 9),
            'final_pitch_count': pitch_count,
            'final_stamina': stamina
        }

    # ----------------- Public API (adds model confidence) -----------------

    def vectorized_pitch_simulation(
        self,
        pitcher_id_or_name=None,
        opposing_lineup: Optional[List[Any]] = None,
        venue: str = "neutral",
        simulations: int = 3000,
    ) -> Dict[str, Any]:
        # reset confidence buffers for this run
        self._conf_buffers_reset()

        # run base to get dist & percentiles etc.
        res = super().vectorized_pitch_simulation(
            pitcher_id_or_name=pitcher_id_or_name,
            opposing_lineup=opposing_lineup,
            venue=venue,
            simulations=simulations,
        )

        # summarize model confidence
        def _mean_or_nan(arr):
            arr = np.asarray(arr, dtype=float)
            return float(np.mean(arr)) if arr.size else float('nan')

        ps_conf = _mean_or_nan(self._pitchselect_maxps)             # ~[0..1]
        wh_conf = _mean_or_nan(self._whiff_binconf)                 # ~[0..1]
        bip_conf= _mean_or_nan(self._bip_maxps_entropy)             # ~[0..1]

        # weighted average (tweak weights if you like)
        weights = np.array([0.45, 0.30, 0.25], dtype=float)
        comps = np.array([
            ps_conf if math.isfinite(ps_conf) else 0.5,
            wh_conf if math.isfinite(wh_conf) else 0.5,
            bip_conf if math.isfinite(bip_conf) else 0.5,
        ], dtype=float)
        model_conf = float(np.dot(weights, comps) / weights.sum()) * 100.0

        res["model_confidence_pct"] = float(round(model_conf, 1))
        res["components_conf"] = {
            "pitch_select_maxp": float(round(ps_conf * 100.0, 1)) if math.isfinite(ps_conf) else float('nan'),
            "whiff_bin":        float(round(wh_conf * 100.0, 1)) if math.isfinite(wh_conf) else float('nan'),
            "bip_entropy":      float(round(bip_conf * 100.0, 1)) if math.isfinite(bip_conf) else float('nan'),
        }
        return res

    # ---------- matchup grade helper (reuse the one from heuristic) ----------

    @staticmethod
    def matchup_grade(mean_k: float, pitcher_baseline_k: float, iqr: float) -> tuple[str, float]:
        # identical to base class; duplicate here for explicitness
        delta = float(mean_k - pitcher_baseline_k)
        if   delta >= 2.0: letter = "A+"
        elif delta >= 1.5: letter = "A"
        elif delta >= 1.0: letter = "A-"
        elif delta >= 0.5: letter = "B"
        elif delta >= 0.0: letter = "B-"
        elif delta >= -0.5: letter = "C"
        elif delta >= -1.0: letter = "D"
        else:               letter = "F"
        denom = max(1.0, float(mean_k))
        conf = max(0.0, 1.0 - min(1.0, float(iqr) / denom)) * 100.0
        return letter, float(round(conf, 1))
