# src/smart_csv_processor.py
# -*- coding: utf-8 -*-

import os, re, warnings
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
warnings.filterwarnings("ignore")

# ------------------ small utils ------------------

def _norm(s: str) -> str:
    s = str(s).strip().lower().replace("\ufeff", "")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _find(df: pd.DataFrame, *cands: str) -> Optional[str]:
    by_norm = {_norm(c): c for c in df.columns}
    # exact
    for k in cands:
        n = _norm(k)
        if n in by_norm:
            return by_norm[n]
    # substring fallback
    for raw in df.columns:
        n = _norm(raw)
        if any(_norm(k) in n for k in cands):
            return raw
    return None

def _as_str(x, default=""):
    try:
        return default if pd.isna(x) else str(x).strip()
    except Exception:
        return default

def _as_int(x, default=None):
    try:
        if pd.isna(x): return default
        return int(float(str(x).strip()))
    except Exception:
        return default

def _is_home(val) -> Optional[bool]:
    v = _as_str(val).lower()
    if v in {"h","home"}: return True
    if v in {"a","away","visitor","v"}: return False
    return None

def _file_ok(path: str, max_mb: int = 50) -> bool:
    try:
        return (os.path.getsize(path)/(1024*1024)) <= max_mb
    except Exception:
        return False

# ------------------ SP detection logic ------------------

TOKENS_SP = {"sp","starter","opener","rhp","lhp"}
EQUALS_P  = {"p","pit","pitcher"}

def _looks_like_pitcher_pos(pos_raw: str) -> bool:
    pos = _as_str(pos_raw).lower()
    if not pos: return False
    parts = set(t for t in re.split(r"[^a-z]+", pos) if t)
    if TOKENS_SP & parts: return True
    return pos in EQUALS_P

def _truthy(val) -> bool:
    v = _as_str(val).lower()
    return v in {"1","true","t","yes","y","sp"}

def _pick_pitcher_strict(players: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    tagged = [p for p in players if p.get("_sp_flag") or _looks_like_pitcher_pos(p.get("position"))]
    if not tagged: return None
    tagged.sort(key=lambda r: r.get("batting_order", 999))
    return tagged[0]

# ------------------ profiles (optional) ------------------

def _load_batter_handedness(batter_profiles_path: str = "models/batter_profiles.parquet") -> Dict[int, str]:
    """
    Return {mlbid -> 'L'|'R'|'S'} if file exists and contains those fields.
    Falls back to empty dict if unavailable.
    """
    try:
        if not os.path.exists(batter_profiles_path):
            return {}
        pdf = pd.read_parquet(batter_profiles_path)
        id_col = next((c for c in pdf.columns if _norm(c) in {"mlbid","mlb_id","id","player_id"}), None)
        hand_col = next((c for c in pdf.columns if _norm(c) in {"bats","bat_hand","batter_hand","hand"}), None)
        if not id_col or not hand_col:
            return {}
        out = {}
        for _, r in pdf[[id_col, hand_col]].dropna().iterrows():
            try:
                out[int(r[id_col])] = _as_str(r[hand_col]).upper()[:1] or "R"
            except Exception:
                continue
        return out
    except Exception:
        return {}

def _load_pitcher_throws(pitcher_profiles_path: str = "models/pitcher_profiles.parquet") -> Dict[int, str]:
    """
    Return {mlbid -> 'L'|'R'} if available (used only for meta/debug).
    """
    try:
        if not os.path.exists(pitcher_profiles_path):
            return {}
        pdf = pd.read_parquet(pitcher_profiles_path)
        id_col = next((c for c in pdf.columns if _norm(c) in {"mlbid","mlb_id","id","player_id"}), None)
        throws_col = next((c for c in pdf.columns if _norm(c) in {"throws","throwing_hand","hand"}), None)
        if not id_col or not throws_col:
            return {}
        out = {}
        for _, r in pdf[[id_col, throws_col]].dropna().iterrows():
            try:
                out[int(r[id_col])] = _as_str(r[throws_col]).upper()[:1] or "R"
            except Exception:
                continue
        return out
    except Exception:
        return {}

# ------------------ main ------------------

def process_smart_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Build prediction tasks from a row-per-player CSV.
    Required (tolerant) columns: team, game_number, home/away, mlb id, player name.
    Pitchers detected via:
      • position/pos/role containing SP/RHP/LHP/etc
      • any column whose *header* contains 'sp' with truthy value
      • game-level home/away SP columns, if present
      • 'batting order' cell value equal to SP/P/RHP/LHP
    Also attaches batter handedness from `models/batter_profiles.parquet` if present.
    """
    if not os.path.exists(file_path):
        print(f"❌ file not found: {file_path}")
        return []
    if not _file_ok(file_path):
        print("❌ file too large")
        return []

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file_path)
    df.columns = [c.replace("\ufeff","").strip() for c in df.columns]

    # Core columns (tolerant)
    team_c = _find(df, "team", "team_code", "team_name")
    gno_c  = _find(df, "game_number", "game_num", "game_no", "gno", "game_id")
    ha_c   = _find(df, "home_away", "home/away", "homeaway", "ha", "side")
    id_c   = _find(df, "mlb_id", "mlbid", "mlb id", "player_id", "id")
    name_c = _find(df, "player_name", "name", "player")

    # Optional helpful ones
    bo_c   = _find(df, "batting_order", "bo", "order", "spot")
    pos_c  = _find(df, "position", "pos", "role")

    # Any column that *looks like* an SP boolean flag (header contains 'sp')
    sp_bool_cols = [c for c in df.columns if re.search(r"\bsp\b", _norm(c))]
    # Game-level SP identifiers (various naming we’ve seen)
    home_sp_id_c = _find(df, "home_sp_mlbid", "home_sp_mlb_id", "home_starting_pitcher_mlbid", "home_sp_id")
    away_sp_id_c = _find(df, "away_sp_mlbid", "away_sp_mlb_id", "away_starting_pitcher_mlbid", "away_sp_id")
    home_sp_nm_c = _find(df, "home_sp_name", "home_starting_pitcher", "home_sp")
    away_sp_nm_c = _find(df, "away_sp_name", "away_starting_pitcher", "away_sp")

    print("🔎 Column map ->",
          dict(team=team_c, game_number=gno_c, ha=ha_c, mlbid=id_c, name=name_c,
               batting_order=bo_c, position=pos_c,
               sp_bool_cols=sp_bool_cols,
               home_sp_id=home_sp_id_c, away_sp_id=away_sp_id_c,
               home_sp_name=home_sp_nm_c, away_sp_name=away_sp_nm_c))

    required = [team_c, gno_c, ha_c, id_c, name_c]
    if any(c is None for c in required):
        print("❌ Missing required columns; headers:", list(df.columns))
        return []

    # Normalize helpful series
    gno_s = df[gno_c].astype(str).str.strip()
    ha_s  = df[ha_c].astype(str).str.strip().str.lower()

    # Load handedness maps (optional)
    batter_hand_map = _load_batter_handedness()
    pitcher_throw_map = _load_pitcher_throws()

    # Group into games
    games: Dict[Tuple[str], Dict[str, Any]] = {}
    for _, r in df.iterrows():
        team = _as_str(r.get(team_c))
        gno  = _as_str(r.get(gno_c))
        side = _is_home(r.get(ha_c))
        name = _as_str(r.get(name_c))
        pid  = _as_int(r.get(id_c))
        bo_raw = r.get(bo_c, None) if bo_c else None
        bo_num = _as_int(bo_raw, 999)
        pos    = _as_str(r.get(pos_c))

        if not team or not gno or side is None or not name:
            continue

        key = (gno,)
        if key not in games:
            games[key] = {
                "game_id": gno,
                "home_team": None, "away_team": None,
                "home_players": [], "away_players": [],
            }

        sp_flag = False
        if sp_bool_cols and any(_truthy(r.get(c)) for c in sp_bool_cols):
            sp_flag = True
        bo_text = _as_str(bo_raw).lower()
        if bo_c and bo_text in {"sp","p","pitcher","rhp","lhp"}:
            sp_flag = True

        row = {"name": name, "mlbid": pid, "team": team,
               "batting_order": bo_num, "position": pos, "_sp_flag": sp_flag}

        if side:
            games[key]["home_team"] = team
            games[key]["home_players"].append(row)
        else:
            games[key]["away_team"] = team
            games[key]["away_players"].append(row)

    # Tag SPs via game-level columns, if present
    def _tag_sp_by_game_side(game_id: str, side_is_home: bool, plist: List[Dict[str,Any]]):
        sp_id_col = home_sp_id_c if side_is_home else away_sp_id_c
        sp_nm_col = home_sp_nm_c if side_is_home else away_sp_nm_c
        if not (sp_id_col or sp_nm_col): return
        mask_game = gno_s.eq(str(game_id))
        mask_side = ha_s.isin(["h","home"]) if side_is_home else ha_s.isin(["a","away","visitor","v"])
        mask = mask_game & mask_side
        if mask.any():
            sub = df.loc[mask]
            sp_ids = set(_as_int(x) for x in sub[sp_id_col].tolist()) if sp_id_col and sp_id_col in sub else set()
            sp_nms = set(_as_str(x).lower() for x in sub[sp_nm_col].tolist()) if sp_nm_col and sp_nm_col in sub else set()
            for p in plist:
                if (p.get("mlbid") in sp_ids) or (_as_str(p.get("name")).lower() in sp_nms):
                    p["_sp_flag"] = True

    for (gno,), g in games.items():
        _tag_sp_by_game_side(gno, True,  g["home_players"])
        _tag_sp_by_game_side(gno, False, g["away_players"])

    # Build tasks (strict: require SP on that side)
    tasks: List[Dict[str, Any]] = []
    for (gno,), g in sorted(games.items(), key=lambda kv: int(_as_int(kv[0][0], 999))):
        if not g["home_team"] or not g["away_team"]:
            continue

        g["home_players"].sort(key=lambda r: r.get("batting_order", 999))
        g["away_players"].sort(key=lambda r: r.get("batting_order", 999))

        def _pick(pl):  # strict SP
            return _pick_pitcher_strict(pl)

        home_sp = _pick(g["home_players"])
        away_sp = _pick(g["away_players"])

        def _mk_lineup(plist, exclude_mlbid):
            # build up to 9 batters, dedup by mlbid/name, attach hand if we have it
            out, seen = [], set()
            for p in plist:
                if p.get("mlbid") == exclude_mlbid:
                    continue
                key = (p.get("mlbid"), _as_str(p.get("name")))
                if key in seen:
                    continue
                seen.add(key)
                mlbid = p.get("mlbid")
                hand = None
                try:
                    hand = batter_hand_map.get(int(mlbid)) if mlbid is not None else None
                except Exception:
                    hand = None
                out.append({
                    "name": p["name"],
                    "mlbid": mlbid,
                    "hand": (hand or "R"),
                    "batting_order": p.get("batting_order"),
                })
                if len(out) == 9:
                    break
            return out

        if away_sp:
            opp = _mk_lineup(g["home_players"], exclude_mlbid=away_sp.get("mlbid"))
            tasks.append({
                "pitcher_name": away_sp["name"],
                "pitcher_mlbid": away_sp.get("mlbid"),
                "pitcher_throws": pitcher_throw_map.get(away_sp.get("mlbid")),
                "team": g["away_team"],
                "opponent": g["home_team"],
                "venue": g["home_team"],
                "lineup": opp,
                "game_id": g["game_id"], "home_away": "away",
            })
        else:
            pos_set = sorted({_as_str(p.get("position")) for p in g["away_players"]}) or [""]
            print(f"⚠️ No AWAY SP found for game {g['game_id']} ({g['away_team']} @ {g['home_team']}). "
                  f"Positions seen (away): {pos_set[:6]} | any SP via bo_text? "
                  f"{any(_as_str(p.get('batting_order')).lower() in {'sp','p','pitcher','rhp','lhp'} for p in g['away_players'])}")

        if home_sp:
            opp = _mk_lineup(g["away_players"], exclude_mlbid=home_sp.get("mlbid"))
            tasks.append({
                "pitcher_name": home_sp["name"],
                "pitcher_mlbid": home_sp.get("mlbid"),
                "pitcher_throws": pitcher_throw_map.get(home_sp.get("mlbid")),
                "team": g["home_team"],
                "opponent": g["away_team"],
                "venue": g["home_team"],
                "lineup": opp,
                "game_id": g["game_id"], "home_away": "home",
            })
        else:
            pos_set = sorted({_as_str(p.get("position")) for p in g["home_players"]}) or [""]
            print(f"⚠️ No HOME SP found for game {g['game_id']} ({g['away_team']} @ {g['home_team']}). "
                  f"Positions seen (home): {pos_set[:6]} | any SP via bo_text? "
                  f"{any(_as_str(p.get('batting_order')).lower() in {'sp','p','pitcher','rhp','lhp'} for p in g['home_players'])}")

    print(f"✅ Created {len(tasks)} prediction tasks from {len(games)} games (strict SP required)")
    return tasks
