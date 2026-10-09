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

    if slate:
        search_dirs: list[Path] = []
        if week:
            # Import inline or normalize here
            s_week = str(week).strip().lower()
            m = re.search(r"\d+", s_week)
            norm_week = f"week-{int(m.group(0)):02d}" if m else s_week
            search_dirs.append(Path(f"data/{norm_week}/{slate}"))
        if slate_date:
            search_dirs.append(Path(f"data/{slate_date}/{slate}"))
        for p in sorted(Path("data").glob(f"week-*/{slate}"), reverse=True):
            if p not in search_dirs:
                search_dirs.append(p)
        for p in sorted(Path("data").glob(f"week-*/{slate}"), reverse=True):
            if p not in search_dirs:
                search_dirs.append(p)
        search_dirs.append(Path(f"data/{slate}"))
        search_dirs.append(Path(f"data/projections/{slate}"))

        for s_dir in search_dirs:
            if s_dir.is_dir():
                for pat in ["*projection*.csv", "*cheatsheet*.csv", "projections.csv", "*.csv"]:
                    matches = sorted(s_dir.glob(pat))
                    if matches:
                        return matches[0]

    candidates = [
        Path("data/week-05/main-slate/projections.csv"),
        Path("data/week-05/main-slate/DFF_NFL_cheatsheet_2026-10-04.csv"),
        Path("data/week-05/sunday-night/projections.csv"),
        Path("data/week-05/monday-night/projections.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    proj_dir = Path("data/projections")
    if proj_dir.is_dir():
        csv_files = sorted(proj_dir.glob("*.csv"))
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

    # Determine projection score column
    score_col = None
    for candidate_col in ["ppg_projection", "PPG_Projection", "fantasy", "Projection", "projection", "Points", "points", "FPPG", "fppg"]:
        if candidate_col in proj_df.columns:
            score_col = candidate_col
            break

    if not score_col:
        logger.warning("Could not determine fantasy score column in %s; skipping projection merge.", proj_path)
        return 0, 0

    player_map: Dict[str, float] = {}
    def_map: Dict[str, float] = {}

    has_first_last = "first_name" in proj_df.columns and "last_name" in proj_df.columns
    player_col = "player" if "player" in proj_df.columns else "Player" if "Player" in proj_df.columns else None
    team_col = "team" if "team" in proj_df.columns else "Team" if "Team" in proj_df.columns else None
    pos_col = "position" if "position" in proj_df.columns else "Position" if "Position" in proj_df.columns else None

    for _, row in proj_df.iterrows():
        if has_first_last:
            fn = str(row["first_name"]).strip() if pd.notna(row["first_name"]) else ""
            ln = str(row["last_name"]).strip() if pd.notna(row["last_name"]) else ""
            p_val = f"{fn} {ln}".strip() if ln else fn
        elif player_col:
            p_val = str(row[player_col]).strip() if pd.notna(row[player_col]) else ""
        else:
            p_val = ""

        t_val = str(row[team_col]).strip() if team_col and pd.notna(row[team_col]) else ""
        pos_val = str(row[pos_col]).strip() if pos_col and pd.notna(row[pos_col]) else ""

        try:
            val = float(row[score_col]) if pd.notna(row[score_col]) else 0.0
        except (ValueError, TypeError):
            val = 0.0

        if pos_val in ("DEF", "DST", "D") or p_val.endswith("D/ST") or "D/ST" in p_val:
            if t_val:
                def_map[normalize_name(t_val)] = val
            clean_team = p_val.replace("D/ST", "").strip()
            if clean_team:
                def_map[normalize_name(clean_team)] = val
        else:
            player_map[normalize_name(p_val)] = val

    updated_count = 0
    untouched_count = 0

    for player in optimizer.player_pool.all_players:
        is_def = "D" in player.positions or player.positions == ["D"]
        if is_def:
            d_key = normalize_name(player.full_name)
            t_key = normalize_name(player.team) if player.team else ""
            found = False
            for cand in [d_key, t_key] + d_key.split():
                if cand in def_map:
                    player.fppg = def_map[cand]
                    updated_count += 1
                    found = True
                    break
            if not found:
                if zero_unprojected:
                    player.fppg = 0.0
                untouched_count += 1
        else:
            k = normalize_name(player.full_name)
            if k in player_map:
                player.fppg = player_map[k]
                updated_count += 1
            else:
                if zero_unprojected:
                    player.fppg = 0.0
                untouched_count += 1

    logger.info(
        "Successfully updated forward-looking projections for %d active players/defenses (%d retained baseline / zeroed=%s).",
        updated_count,
        untouched_count,
        zero_unprojected,
    )
    return updated_count, untouched_count
