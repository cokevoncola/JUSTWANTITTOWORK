# src/pick_selector.py
from __future__ import annotations
import math
import re
from typing import Tuple
import numpy as np
import pandas as pd

# ---------- Odds & probability helpers ----------
def _american_to_prob(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return (-o) / ((-o) + 100.0)
    return np.nan

def _decimal_to_prob(odds: float) -> float:
    try:
        o = float(odds)
        return 1.0 / o if o > 1.0 else np.nan
    except Exception:
        return np.nan

def _implied_prob(odds_val) -> float:
    if odds_val is None or (isinstance(odds_val, float) and np.isnan(odds_val)):
        return np.nan
    try:
        v = float(odds_val)
    except Exception:
        return np.nan
    # crude detection: typical decimal range
    if 1.5 <= abs(v) <= 25.0:
        return _decimal_to_prob(v)
    return _american_to_prob(v)

# ---------- Distribution helpers ----------
def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def _estimate_sigma(row: pd.Series) -> float:
    if "std_dev" in row and pd.notna(row["std_dev"]) and float(row["std_dev"]) > 0:
        return float(row["std_dev"])

    p10, p90 = row.get("percentile_10"), row.get("percentile_90")
    if pd.notna(p10) and pd.notna(p90):
        try:
            return max(0.2, float(p90) - float(p10)) / 2.563
        except Exception:
            pass

    p25, p75 = row.get("percentile_25"), row.get("percentile_75")
    if pd.notna(p25) and pd.notna(p75):
        try:
            return max(0.2, float(p75) - float(p25)) / 1.349
        except Exception:
            pass

    return 1.2  # fallback width if nothing else available

def _nearest_half(x: float) -> float:
    return float(np.round(x * 2.0) / 2.0)

# ---------- Model prob helpers ----------
def _extract_prob_over_cols(df: pd.DataFrame) -> dict:
    """
    Map prob_over_* columns to lines:
      'prob_over_6_5' -> 6.5 ; 'prob_over_7' -> 7.0
    """
    mapping = {}
    for c in df.columns:
        m = re.match(r"prob_over_(\d+)_?(\d+)?$", str(c))
        if m:
            whole = int(m.group(1))
            frac = m.group(2)
            if frac is None:
                line = float(whole)
            else:
                line = whole + (0.5 if frac == "5" else float(f"0.{frac}"))
            mapping[round(line, 2)] = c
    return mapping

def _resolve_line(row: pd.Series) -> float:
    """
    1) Betting_Line if present
    2) nearest 0.5 to final_prediction
    3) nearest 0.5 to ML_Prediction
    4) default 5.5
    """
    if "Betting_Line" in row and pd.notna(row["Betting_Line"]):
        try:
            return float(row["Betting_Line"])
        except Exception:
            pass
    mu = row.get("final_prediction", row.get("mean_strikeouts", row.get("ML_Prediction", np.nan)))
    if pd.notna(mu):
        return _nearest_half(float(mu))
    return 5.5

def _model_prob_over(row: pd.Series, line: float, prob_cols_map: dict) -> float:
    line_rounded = round(line, 2)
    # prefer model-native prob_over_* if matching the requested line
    if line_rounded in prob_cols_map:
        c = prob_cols_map[line_rounded]
        val = row.get(c)
        if pd.notna(val):
            return float(val)
    # fallback: approximate via Normal with estimated sigma
    mu = float(row.get("final_prediction", row.get("mean_strikeouts", row.get("ML_Prediction", np.nan))))
    if not pd.notna(mu):
        return np.nan
    sigma = _estimate_sigma(row)
    return 1.0 - _norm_cdf(line, mu, sigma)

def _confidence_score(row: pd.Series, line: float) -> float:
    mu = float(row.get("final_prediction", row.get("mean_strikeouts", row.get("ML_Prediction", 0.0))))
    sigma = _estimate_sigma(row)
    z = abs(mu - line) / max(1e-6, sigma)
    return float(np.clip(1.0 / (1.0 + math.exp(-1.25 * (z - 0.75))), 0.0, 1.0))

# ---------- Public APIs ----------
def rank_strikeout_edges(
    results_df: pd.DataFrame,
    top_n: int = 10,
    min_edge: float = 0.06,
    min_model_prob: float = 0.58
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return top-N Over and Under candidates, ranked by edge & confidence.
    (Useful for on-screen tables and separate CSV downloads.)
    """
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = results_df.copy()
    prob_cols_map = _extract_prob_over_cols(df)

    df["line_eval"] = df.apply(_resolve_line, axis=1)
    df["model_prob_over"] = df.apply(lambda r: _model_prob_over(r, r["line_eval"], prob_cols_map), axis=1)
    df["model_prob_under"] = 1.0 - df["model_prob_over"]

    df["imp_prob_over"] = df["Over_Odds"].apply(_implied_prob) if "Over_Odds" in df.columns else np.nan
    df["imp_prob_under"] = df["Under_Odds"].apply(_implied_prob) if "Under_Odds" in df.columns else np.nan

    df["edge_over"]  = df["model_prob_over"]  - df["imp_prob_over"].fillna(0.5)
    df["edge_under"] = df["model_prob_under"] - df["imp_prob_under"].fillna(0.5)

    df["confidence"] = df.apply(lambda r: _confidence_score(r, r["line_eval"]), axis=1)

    overs = df[(df["model_prob_over"]  >= float(min_model_prob)) & (df["edge_over"]  >= float(min_edge))].copy()
    unders = df[(df["model_prob_under"] >= float(min_model_prob)) & (df["edge_under"] >= float(min_edge))].copy()

    overs.sort_values(by=["edge_over", "confidence", "model_prob_over"], ascending=[False, False, False], inplace=True)
    unders.sort_values(by=["edge_under", "confidence", "model_prob_under"], ascending=[False, False, False], inplace=True)

    preferred = [
        "Pitcher","MLBID","Team","Opponent","Venue",
        "Betting_Line","Over_Odds","Under_Odds",
        "ML_Prediction","final_prediction","mean_strikeouts",
        "percentile_10","percentile_25","percentile_75","percentile_90","std_dev",
        "line_eval","model_prob_over","model_prob_under",
        "imp_prob_over","imp_prob_under",
        "edge_over","edge_under","confidence","matchup_grade","Smart_MC_Range"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return overs[cols].head(top_n), unders[cols].head(top_n)

def augment_with_edges(
    results_df: pd.DataFrame,
    min_edge: float = 0.06,
    min_model_prob: float = 0.58
) -> pd.DataFrame:
    """
    Return the SAME table as your results but augmented with:
      - line_eval, model_prob_over/under
      - imp_prob_over/under (if odds given)
      - edge_over/under, confidence
      - Recommended_Pick (Over / Under / Pass)
      - Edge (the chosen edge), Model_Prob (the chosen probability)
    Use this DataFrame for your main CSV download so everything is baked in.
    """
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return results_df

    df = results_df.copy()
    prob_cols_map = _extract_prob_over_cols(df)

    df["line_eval"] = df.apply(_resolve_line, axis=1)
    df["model_prob_over"] = df.apply(lambda r: _model_prob_over(r, r["line_eval"], prob_cols_map), axis=1)
    df["model_prob_under"] = 1.0 - df["model_prob_over"]

    df["imp_prob_over"] = df["Over_Odds"].apply(_implied_prob) if "Over_Odds" in df.columns else np.nan
    df["imp_prob_under"] = df["Under_Odds"].apply(_implied_prob) if "Under_Odds" in df.columns else np.nan

    df["edge_over"]  = df["model_prob_over"]  - df["imp_prob_over"].fillna(0.5)
    df["edge_under"] = df["model_prob_under"] - df["imp_prob_under"].fillna(0.5)

    df["confidence"] = df.apply(lambda r: _confidence_score(r, r["line_eval"]), axis=1)

    # Recommended pick & chosen stats
    def _pick(row):
        over_ok  = (row["model_prob_over"]  >= float(min_model_prob)) and (row["edge_over"]  >= float(min_edge))
        under_ok = (row["model_prob_under"] >= float(min_model_prob)) and (row["edge_under"] >= float(min_edge))

        if over_ok and (row["edge_over"] >= row["edge_under"]):
            return "Over", float(row["edge_over"]), float(row["model_prob_over"])
        if under_ok:
            return "Under", float(row["edge_under"]), float(row["model_prob_under"])
        return "Pass", 0.0, float(max(row["model_prob_over"], row["model_prob_under"]))

    picks = df.apply(_pick, axis=1, result_type="expand")
    df["Recommended_Pick"] = picks[0]
    df["Edge"] = picks[1]
    df["Model_Prob"] = picks[2]

    return df
