"""
================================================================================
Forward-Looking Projections Ingestion Module (src/data/projections.py)
================================================================================
Parses external weekly forward-looking fantasy projections (FanDuel Research,
NumberFire, ESPN Fantasy, FantasyPros, etc.) and overwrites backward-looking
season average FPPG values for players and team defenses.
================================================================================
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from pydfs_lineup_optimizer import LineupOptimizer

logger = logging.getLogger("FanDuelSimOptimizer")


def normalize_name(text: str) -> str:
    """Normalizes player or team name for robust matching across vendor data sources."""
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    # Remove common generational and numerical name suffixes
    t = re.sub(r"\b(jr|sr|iii|ii|iv)\b", "", t)
    return " ".join(t.split())


def find_projections_csv(
    explicit_path: Optional[Path] = None,
    slate: Optional[str] = None,
    week: Optional[Union[str, int]] = None,
    slate_date: Optional[str] = None,
) -> Optional[Path]:
    """Discovers projection CSV files from explicit path, data/{week}/{slate}/, or fallback paths."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    norm_week = None
    if week:
        # Import inline or normalize here
        s_week = str(week).strip().lower()
        m = re.search(r"\d+", s_week)
        norm_week = f"week-{int(m.group(0)):02d}" if m else s_week

    if slate:
        search_dirs: list[Path] = []
        if norm_week:
            search_dirs.append(Path(f"data/{norm_week}/{slate}"))
        if slate_date:
            search_dirs.append(Path(f"data/{slate_date}/{slate}"))
        if not norm_week and not slate_date:
            for p in sorted(Path("data").glob(f"week-*/{slate}"), reverse=True):
                if p not in search_dirs:
                    search_dirs.append(p)
        search_dirs.append(Path(f"data/{slate}"))
        search_dirs.append(Path(f"data/projections/{slate}"))

        for s_dir in search_dirs:
            if s_dir.is_dir():
                for pat in ["*projection*.csv", "*cheatsheet*.csv", "projections.csv"]:
                    matches = [
                        p for p in sorted(s_dir.glob(pat))
                        if not p.name.startswith("Completed-") and p.name != "completed_lineups.csv"
                    ]
                    if matches:
                        return matches[0]

    # Only fall back to week-05 candidates if neither explicit week nor slate_date was given,
    # or if week-05 was explicitly requested.
    if norm_week in (None, "week-05") and not slate_date:
        candidates = [
            Path("data/week-05/main-slate/projections.csv"),
            Path("data/week-05/main-slate/DFF_NFL_cheatsheet_2026-10-04.csv"),
            Path("data/week-05/sunday-night/projections.csv"),
            Path("data/week-05/monday-night/projections.csv"),
        ]
        if slate:
            candidates = [c for c in candidates if c.parent.name == slate]
        for c in candidates:
            if c.exists():
                return c

    if not norm_week and not slate_date and not slate:
        proj_dir = Path("data/projections")
        if proj_dir.is_dir():
            csv_files = [
                p for p in sorted(proj_dir.glob("*.csv"))
                if not p.name.startswith("Completed-") and p.name != "completed_lineups.csv"
            ]
            if csv_files:
                return csv_files[0]

    return None


def apply_forward_projections(
    optimizer: LineupOptimizer,
    proj_path: Path,
    zero_unprojected: bool = False,
) -> Tuple[int, int]:
    """
    Loads weekly forward-looking projections and overwrites backward-looking historical
    FPPG values for matched players and team defenses.

    If zero_unprojected is False (default), players not present in the projection file
    retain their baseline values, allowing partial projection files (e.g. QB-only updates)
    to merge cleanly. If True, unprojected players are zeroed out.

    Returns:
        (updated_count, unprojected_count)
    """
    logger.info("Applying forward-looking projections from: %s", proj_path)
    proj_df = pd.read_csv(proj_path)

    # Determine projection score column (prioritizing forward-looking ppg_projection)
    score_col = None
    for candidate_col in [
        "ppg_projection",
        "PPG_Projection",
        "ppg_proj",
        "PPG_Proj",
        "proj_fppg",
        "Proj_FPPG",
        "ppg",
        "PPG",
        "projection",
        "Projection",
        "fantasy",
        "Fantasy",
        "Points",
        "points",
        "FPPG",
        "fppg",
    ]:
        if candidate_col in proj_df.columns:
            score_col = candidate_col
            break

    if not score_col:
        logger.warning("Could not determine fantasy score column in %s; skipping projection merge.", proj_path)
        return 0, 0

    # Determine ownership column if present
    own_col = None
    for candidate_col in ["ownership_projection", "ownership", "projected_ownership", "own", "Ownership", "Proj_Own", "proj_own"]:
        if candidate_col in proj_df.columns:
            own_col = candidate_col
            break

    own_is_percent = False
    if own_col:
        _own_num = pd.to_numeric(proj_df[own_col].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
        own_is_percent = bool((_own_num.max(skipna=True) or 0.0) > 1.0)

    player_map: Dict[str, float] = {}
    player_own_map: Dict[str, float] = {}
    def_map: Dict[str, float] = {}
    def_own_map: Dict[str, float] = {}
    projected_positions: set[str] = set()

    has_first_last = (
        ("first_name" in proj_df.columns and "last_name" in proj_df.columns)
        or ("First Name" in proj_df.columns and "Last Name" in proj_df.columns)
    )
    fn_col = "first_name" if "first_name" in proj_df.columns else "First Name"
    ln_col = "last_name" if "last_name" in proj_df.columns else "Last Name"

    player_col = None
    for cand in ["player", "Player", "name", "Name", "player_name", "Player_Name"]:
        if cand in proj_df.columns:
            player_col = cand
            break

    team_col = None
    for cand in ["team", "Team", "team_abbrev", "TeamAbbrev", "TEAM", "Team_Abbrev"]:
        if cand in proj_df.columns:
            team_col = cand
            break

    pos_col = None
    for cand in ["position", "Position", "pos", "Pos", "POS"]:
        if cand in proj_df.columns:
            pos_col = cand
            break

    for _, row in proj_df.iterrows():
        if has_first_last:
            fn = str(row[fn_col]).strip() if pd.notna(row[fn_col]) else ""
            ln = str(row[ln_col]).strip() if pd.notna(row[ln_col]) else ""
            p_val = f"{fn} {ln}".strip() if ln else fn
        elif player_col:
            p_val = str(row[player_col]).strip() if pd.notna(row[player_col]) else ""
        else:
            p_val = ""

        t_val = str(row[team_col]).strip() if team_col and pd.notna(row[team_col]) else ""
        pos_val = str(row[pos_col]).strip().upper() if pos_col and pd.notna(row[pos_col]) else ""
        if pos_val:
            if pos_val in ("DEF", "DST", "D"):
                projected_positions.update(["D", "DEF", "DST"])
            else:
                projected_positions.add(pos_val)

        val = None
        if score_col in row and pd.notna(row[score_col]) and str(row[score_col]).strip() != "":
            try:
                val = float(str(row[score_col]).strip().replace("$", ""))
            except (ValueError, TypeError):
                val = None

        own_val = None
        if own_col and own_col in row and pd.notna(row[own_col]) and str(row[own_col]).strip() != "":
            try:
                raw_own = float(str(row[own_col]).strip().replace("%", ""))
                own_val = raw_own / 100.0 if own_is_percent else raw_own
            except (ValueError, TypeError):
                own_val = None

        if val is None:
            # Fallback for ragged CSV rows (e.g. D/ST rows with fewer columns than offensive rows)
            # Scan trailing columns in reverse, skipping recognized non-projection columns (salary, team, etc.)
            for col_name, item in zip(reversed(proj_df.columns), reversed(row.values)):
                col_lower = str(col_name).strip().lower()
                if any(k in col_lower for k in ("salary", "team", "player", "name", "rank", "game", "date", "id", "pos")):
                    continue
                if pd.notna(item) and str(item).strip() != "":
                    try:
                        clean_num = float(str(item).strip().replace("$", ""))
                        if 0.0 <= clean_num <= 100.0:
                            val = clean_num
                            break
                    except (ValueError, TypeError):
                        continue

        if val is None:
            val = 0.0

        is_defense = (
            pos_val in ("DEF", "DST", "D")
            or p_val.endswith("D/ST")
            or "D/ST" in p_val
            or "DST" in p_val
            or "DEF" in p_val
            or p_val.lower().endswith("defense")
        )
        if is_defense:
            projected_positions.update(["D", "DEF", "DST"])
            if t_val:
                def_map[normalize_name(t_val)] = val
                if own_val is not None:
                    def_own_map[normalize_name(t_val)] = own_val
            clean_team = (
                p_val.replace("D/ST", "")
                .replace("DST", "")
                .replace("DEF", "")
                .replace("Defense", "")
                .replace("defense", "")
                .strip()
            )
            if clean_team:
                def_map[normalize_name(clean_team)] = val
                if own_val is not None:
                    def_own_map[normalize_name(clean_team)] = own_val
            if p_val:
                def_map[normalize_name(p_val)] = val
                if own_val is not None:
                    def_own_map[normalize_name(p_val)] = own_val
        else:
            player_map[normalize_name(p_val)] = val
            if own_val is not None:
                player_own_map[normalize_name(p_val)] = own_val

    # Augment projected_positions with positions from matched pool players
    if def_map:
        projected_positions.update(["D", "DEF", "DST"])

    for player in optimizer.player_pool.all_players:
        if normalize_name(player.full_name) in player_map:
            projected_positions.update(player.positions)

    updated_count = 0
    untouched_count = 0

    def_covered = bool(def_map or any(p in projected_positions for p in ("D", "DEF", "DST")))

    for player in optimizer.player_pool.all_players:
        is_def = "D" in player.positions or "DST" in player.positions or player.positions in (["D"], ["DST"])
        if is_def:
            d_key = normalize_name(player.full_name)
            t_key = normalize_name(player.team) if player.team else ""
            found = False
            for cand in [d_key, t_key] + d_key.split():
                if cand in def_map:
                    player.fppg = def_map[cand]
                    if cand in def_own_map:
                        player.projected_ownership = def_own_map[cand]
                    updated_count += 1
                    found = True
                    break
            if not found:
                # Only zero out if the projection file actually provided defensive projections
                if zero_unprojected and def_covered:
                    player.fppg = 0.0
                untouched_count += 1
        else:
            k = normalize_name(player.full_name)
            if k in player_map:
                player.fppg = player_map[k]
                if k in player_own_map:
                    player.projected_ownership = player_own_map[k]
                updated_count += 1
            else:
                pos_covered = any(pos in projected_positions for pos in player.positions)
                # Only zero out if this player's position was covered in the projection file
                if zero_unprojected and pos_covered:
                    player.fppg = 0.0
                untouched_count += 1

    logger.info(
        "Successfully updated forward-looking projections for %d active players/defenses (%d retained baseline / zeroed=%s).",
        updated_count,
        untouched_count,
        zero_unprojected,
    )
    return updated_count, untouched_count
