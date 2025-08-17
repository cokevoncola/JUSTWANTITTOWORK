# src/profile_enrichment.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
def inspect_profile_file(path: str) -> dict:
    import os
    import pandas as pd

    def _norm(s: str) -> str:
        s = str(s).strip().lower().replace("\ufeff", "")
        out = []
        for ch in s:
            out.append(ch if ch.isalnum() else "_")
        ns = "".join(out)
        while "__" in ns:
            ns = ns.replace("__", "_")
        return ns.strip("_")

    def _pick(df: pd.DataFrame, *cands: str):
        cols = {_norm(c): c for c in df.columns}
        for k in cands:
            if _norm(k) in cols:
                return cols[_norm(k)]
        for raw in df.columns:
            n = _norm(raw)
            if any(_norm(k) in n for k in cands):
                return raw
        return None

    info = {"exists": os.path.exists(path), "path": path}
    if not os.path.exists(path):
        return info
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {**info, "error": f"failed_to_read_parquet: {e}"}
    info["n_rows"] = len(df)
    info["columns"] = list(df.columns)
    info["norm_columns"] = [_norm(c) for c in df.columns]

    id_c   = _pick(df, "mlbid", "mlb_id", "player_id", "id")
    name_c = _pick(df, "name", "player_name")
    throws_c = _pick(df, "throws", "throwing_hand", "hand")
    bats_c   = _pick(df, "bats", "bat_hand", "batter_hand", "hand")
    k_all   = _pick(df, "k_rate", "k_pct", "kpercent", "k_percentage", "k_per_pa")
    k9      = _pick(df, "k9", "k_9", "strikeouts_per_9")
    kvl     = _pick(df, "k_rate_vs_lhb", "k_pct_vs_lhb", "kpercent_vs_lhb", "k_vs_left")
    kvr     = _pick(df, "k_rate_vs_rhb", "k_pct_vs_rhb", "kpercent_vs_rhb", "k_vs_right")
    klhp    = _pick(df, "k_rate_vs_lhp", "k_pct_vs_lhp", "kpercent_vs_lhp", "k_vs_left")
    krhp    = _pick(df, "k_rate_vs_rhp", "k_pct_vs_rhp", "kpercent_vs_rhp", "k_vs_right")

    info["guessed_columns"] = dict(
        id=id_c, name=name_c, throws=throws_c, bats=bats_c,
        k_rate=k_all, k9=k9,
        k_rate_vs_LHB=kvl, k_rate_vs_RHB=kvr,
        k_rate_vs_LHP=klhp, k_rate_vs_RHP=krhp
    )
    take = [c for c in [id_c, name_c, throws_c, bats_c, k_all, k9, kvl, kvr, klhp, krhp] if c]
    info["sample"] = df[take].head(5).to_dict(orient="records") if take else []
    return info

# ------------- helpers -------------

def _norm(s: str) -> str:
    s = str(s).strip().lower().replace("\ufeff", "")
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    ns = "".join(out)
    while "__" in ns:
        ns = ns.replace("__", "_")
    return ns.strip("_")


def _pick(pdf: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {_norm(c): c for c in pdf.columns}
    # exact normalized match
    for k in cands:
        if _norm(k) in cols:
            return cols[_norm(k)]
    # loose contains
    for raw in pdf.columns:
        n = _norm(raw)
        if any(_norm(k) in n for k in cands):
            return raw
    return None


def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _clip(p: Optional[float], lo=0.02, hi=0.60) -> Optional[float]:
    if p is None:
        return None
    return max(lo, min(hi, float(p)))


def _k9_to_per_pa(k9: float) -> float:
    # 9 IP ~ ~38.7 PA ⇒ p(K per PA) ≈ K9 / 38.7
    return max(0.02, min(0.60, k9 / 38.7))


def _name_key(x: str) -> str:
    return _norm(str(x)).replace("_", "")


# ------------- schema inspector -------------

def inspect_profile_file(path: str) -> Dict[str, Any]:
    """Return columns, normalized columns, and best-guess field mapping."""
    info = {"exists": os.path.exists(path), "path": path}
    if not os.path.exists(path):
        return info
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {**info, "error": f"failed_to_read_parquet: {e}"}
    info["n_rows"] = len(df)
    info["columns"] = list(df.columns)
    info["norm_columns"] = [ _norm(c) for c in df.columns ]

    # our best-guess mapping for this file
    id_c   = _pick(df, "mlbid", "mlb_id", "player_id", "id")
    name_c = _pick(df, "name", "player_name")
    throws_c = _pick(df, "throws", "throwing_hand", "hand")
    bats_c   = _pick(df, "bats", "bat_hand", "batter_hand", "hand")

    k_all   = _pick(df, "k_rate", "k_pct", "kpercent", "k_percentage", "k_per_pa")
    k9      = _pick(df, "k9", "k_9", "strikeouts_per_9")
    kvl     = _pick(df, "k_rate_vs_lhb", "k_pct_vs_lhb", "kpercent_vs_lhb", "k_vs_left")
    kvr     = _pick(df, "k_rate_vs_rhb", "k_pct_vs_rhb", "kpercent_vs_rhb", "k_vs_right")
    klhp    = _pick(df, "k_rate_vs_lhp", "k_pct_vs_lhp", "kpercent_vs_lhp", "k_vs_left")
    krhp    = _pick(df, "k_rate_vs_rhp", "k_pct_vs_rhp", "kpercent_vs_rhp", "k_vs_right")

    info["guessed_columns"] = dict(
        id=id_c, name=name_c, throws=throws_c, bats=bats_c,
        k_rate=k_all, k9=k9,
        k_rate_vs_LHB=kvl, k_rate_vs_RHB=kvr,  # pitchers
        k_rate_vs_LHP=klhp, k_rate_vs_RHP=krhp # batters
    )
    # sample 5 rows of the columns we care about
    take = [c for c in [id_c, name_c, throws_c, bats_c, k_all, k9, kvl, kvr, klhp, krhp] if c]
    info["sample"] = df[take].head(5).to_dict(orient="records") if take else []
    return info


# ------------- lookups -------------

def load_pitcher_lookup(path: str = "models/pitcher_profiles.parquet") -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      lookup: {mlbid -> {"k_rate","k_rate_vs_LHB","k_rate_vs_RHB","throws","name"}}
      meta: coverage + which columns used + name index size
    """
    lookup: Dict[int, Dict[str, Any]] = {}
    meta = {"exists": os.path.exists(path), "n_rows": 0, "n_enriched": 0, "used_cols": {}}

    if not os.path.exists(path):
        return lookup, meta

    pdf = pd.read_parquet(path)
    meta["n_rows"] = len(pdf)

    id_c = _pick(pdf, "mlbid", "mlb_id", "player_id", "id")
    name_c = _pick(pdf, "name", "player_name")
    throws_c = _pick(pdf, "throws", "throwing_hand", "hand")

    k_all_c = _pick(pdf, "k_rate", "k_pct", "kpercent", "k_percentage", "k_per_pa")
    k9_c    = _pick(pdf, "k9", "k_9", "strikeouts_per_9")
    kvl_c   = _pick(pdf, "k_rate_vs_lhb", "k_pct_vs_lhb", "kpercent_vs_lhb", "k_vs_left")
    kvr_c   = _pick(pdf, "k_rate_vs_rhb", "k_pct_vs_rhb", "kpercent_vs_rhb", "k_vs_right")

    meta["used_cols"] = dict(id=id_c, name=name_c, throws=throws_c, k_rate=k_all_c, k9=k9_c, vsL=kvl_c, vsR=kvr_c)

    name_index: Dict[str, int] = {}

    for _, r in pdf.iterrows():
        # build id
        pid = r.get(id_c)
        try:
            pid = int(pid)
        except Exception:
            pid = None

        # build record
        k_all = _as_float(r.get(k_all_c)) if k_all_c else None
        k9 = _as_float(r.get(k9_c)) if k9_c else None
        if k_all is None and k9 is not None:
            k_all = _k9_to_per_pa(k9)
        k_vs_l = _as_float(r.get(kvl_c)) if kvl_c else None
        k_vs_r = _as_float(r.get(kvr_c)) if kvr_c else None
        if k_vs_l is None and k_all is not None:
            k_vs_l = k_all
        if k_vs_r is None and k_all is not None:
            k_vs_r = k_all

        rec = {
            "name": r.get(name_c),
            "throws": (str(r.get(throws_c)).upper()[:1] if throws_c and pd.notna(r.get(throws_c)) else None),
            "k_rate": _clip(k_all),
            "k_rate_vs_LHB": _clip(k_vs_l),
            "k_rate_vs_RHB": _clip(k_vs_r),
        }

        if pid is not None:
            lookup[pid] = rec

        nm_key = _name_key(r.get(name_c)) if name_c in r else None
        if nm_key:
            name_index[nm_key] = pid if pid is not None else -1

    meta["name_index_size"] = len(name_index)
    meta["id_index_size"] = len(lookup)
    meta["_name_index"] = name_index  # kept for optional name fallback
    meta["_raw_cols"] = list(pdf.columns)
    meta["n_enriched"] = sum(1 for v in lookup.values() if v.get("k_rate") is not None)
    return lookup, meta


def load_batter_lookup(path: str = "models/batter_profiles.parquet") -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      lookup: {mlbid -> {"hand","k_rate","k_rate_vs_LHP","k_rate_vs_RHP","name"}}
      meta: coverage + which columns used + name index size
    """
    lookup: Dict[int, Dict[str, Any]] = {}
    meta = {"exists": os.path.exists(path), "n_rows": 0, "n_enriched": 0, "used_cols": {}}

    if not os.path.exists(path):
        return lookup, meta

    pdf = pd.read_parquet(path)
    meta["n_rows"] = len(pdf)

    id_c = _pick(pdf, "mlbid", "mlb_id", "player_id", "id")
    name_c = _pick(pdf, "name", "player_name")
    bats_c = _pick(pdf, "bats", "bat_hand", "batter_hand", "hand")

    k_all_c = _pick(pdf, "k_rate", "k_pct", "kpercent", "k_percentage", "k_per_pa")
    k9_c    = _pick(pdf, "k9", "k_9", "strikeouts_per_9")
    klhp_c  = _pick(pdf, "k_rate_vs_lhp", "k_pct_vs_lhp", "kpercent_vs_lhp", "k_vs_left")
    krhp_c  = _pick(pdf, "k_rate_vs_rhp", "k_pct_vs_rhp", "kpercent_vs_rhp", "k_vs_right")

    meta["used_cols"] = dict(id=id_c, name=name_c, bats=bats_c, k_rate=k_all_c, k9=k9_c, vsL=klhp_c, vsR=krhp_c)

    name_index: Dict[str, int] = {}

    for _, r in pdf.iterrows():
        bid = r.get(id_c)
        try:
            bid = int(bid)
        except Exception:
            bid = None

        hand = (str(r.get(bats_c)).upper()[:1] if bats_c and pd.notna(r.get(bats_c)) else None)

        k_all = _as_float(r.get(k_all_c)) if k_all_c else None
        k9 = _as_float(r.get(k9_c)) if k9_c else None
        if k_all is None and k9 is not None:
            k_all = _k9_to_per_pa(k9)
        k_vs_l = _as_float(r.get(klhp_c)) if klhp_c else None
        k_vs_r = _as_float(r.get(krhp_c)) if krhp_c else None
        if k_vs_l is None and k_all is not None:
            k_vs_l = k_all
        if k_vs_r is None and k_all is not None:
            k_vs_r = k_all

        rec = {
            "name": r.get(name_c),
            "hand": hand,
            "k_rate": _clip(k_all),
            "k_rate_vs_LHP": _clip(k_vs_l),
            "k_rate_vs_RHP": _clip(k_vs_r),
        }

        if bid is not None:
            lookup[bid] = rec

        nm_key = _name_key(r.get(name_c)) if name_c in r else None
        if nm_key:
            name_index[nm_key] = bid if bid is not None else -1

    meta["name_index_size"] = len(name_index)
    meta["id_index_size"] = len(lookup)
    meta["_name_index"] = name_index
    meta["_raw_cols"] = list(pdf.columns)
    meta["n_enriched"] = sum(1 for v in lookup.values() if v.get("k_rate") is not None)
    return lookup, meta


# ------------- enrichment -------------

def enrich_tasks_with_profiles(
    tasks: List[dict],
    pitcher_lookup: Dict[int, Dict[str, Any]],
    batter_lookup: Dict[int, Dict[str, Any]],
    pitcher_meta: Optional[Dict[str, Any]] = None,
    batter_meta: Optional[Dict[str, Any]] = None,
    enable_name_fallback: bool = True,
) -> dict:
    """
    Mutates tasks in-place to fill pitcher/batter fields from lookups.
    Tries ID match first; if not found and enable_name_fallback=True, tries a normalized name match.
    Returns a coverage dict with hit counts, missing lists, and per-task coverage.
    """
    name_index_p = (pitcher_meta or {}).get("_name_index", {}) if enable_name_fallback else {}
    name_index_b = (batter_meta or {}).get("_name_index", {}) if enable_name_fallback else {}

    pitcher_hits = 0
    batter_hits = 0
    missing_pitchers: List[dict] = []
    missing_batters: List[dict] = []
    per_task: List[dict] = []

    for t in tasks:
        pit = t.get("pitcher", {}) or {}
        pid = pit.get("id") or pit.get("mlbid")
        try:
            pid = int(pid)
        except Exception:
            pid = None

        # --- Pitcher match (ID first, then name) ---
        p_rec = None
        if pid is not None and pid in pitcher_lookup:
            p_rec = pitcher_lookup[pid]
        elif enable_name_fallback:
            key = _name_key(pit.get("name"))
            alt_pid = name_index_p.get(key)
            if alt_pid is not None and alt_pid in pitcher_lookup:
                p_rec = pitcher_lookup[alt_pid]

        if p_rec:
            pit.setdefault("name", p_rec.get("name"))
            pit.setdefault("throws", p_rec.get("throws"))
            pit.setdefault("k_rate", p_rec.get("k_rate"))
            pit.setdefault("k_rate_vs_LHB", p_rec.get("k_rate_vs_LHB"))
            pit.setdefault("k_rate_vs_RHB", p_rec.get("k_rate_vs_RHB"))
            pitcher_hits += 1
        else:
            missing_pitchers.append({"pitcher_id": pid, "pitcher_name": pit.get("name")})

        t["pitcher"] = pit

        # --- Batters ---
        opp = t.get("opponent_lineup", []) or []
        b_hit_count = 0
        b_total = len(opp)

        for b in opp:
            bid = b.get("batter_id")
            try:
                bid = int(bid)
            except Exception:
                bid = None

            b_rec = None
            if bid is not None and bid in batter_lookup:
                b_rec = batter_lookup[bid]
            elif enable_name_fallback:
                # Some CSVs lack MLBIDs; try name fallback if present
                bname = b.get("name") or ""
                key = _name_key(bname)
                alt_bid = name_index_b.get(key)
                if alt_bid is not None and alt_bid in batter_lookup:
                    b_rec = batter_lookup[alt_bid]

            if b_rec:
                b.setdefault("hand", (b_rec.get("hand") or b.get("hand") or "R")[:1].upper())
                b.setdefault("k_rate", b_rec.get("k_rate"))
                b.setdefault("k_rate_vs_LHP", b_rec.get("k_rate_vs_LHP"))
                b.setdefault("k_rate_vs_RHP", b_rec.get("k_rate_vs_RHP"))
                batter_hits += 1
                b_hit_count += 1
            else:
                if bid is not None or b.get("hand") is None:
                    missing_batters.append({"batter_id": bid, "hand": b.get("hand"), "name": b.get("name")})
                b["hand"] = (b.get("hand") or "R")[:1].upper()

        t["opponent_lineup"] = opp
        per_task.append({
            "pitcher_id": pid,
            "pitcher_name": pit.get("name"),
            "batter_hits": b_hit_count,
            "batter_total": b_total,
        })

    return {
        "pitcher_hits": pitcher_hits,
        "batter_hits": batter_hits,
        "n_tasks": len(tasks),
        "avg_batters_per_task": (sum(len(t.get("opponent_lineup", [])) for t in tasks) / max(1, len(tasks))),
        "missing_pitchers": missing_pitchers[:64],
        "missing_batters": missing_batters[:128],
        "per_task": per_task[:128],
    }
