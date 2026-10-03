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
from typing import Dict, Optional, Tuple

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


def find_projections_csv(explicit_path: Optional[Path] = None) -> Optional[Path]:
    """Discovers projection CSV files from explicit path or data/projections/ directory."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("data/projections/fanduel_research_projections.csv"),
        Path("data/projections/projections.csv"),
        Path("data/projections/numberfire_projections.csv"),
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


def apply_forward_projections(optimizer: LineupOptimizer, proj_path: Path) -> Tuple[int, int]:
    """
    Loads weekly forward-looking projections and overwrites backward-looking historical
    FPPG values for all players and team defenses.

    Unprojected bench / inactive players are set to 0.0 to prevent selection anomalies.
    Returns:
        (updated_count, zeroed_count)
    """
    logger.info("Applying forward-looking projections from: %s", proj_path)
    proj_df = pd.read_csv(proj_path)

    # Determine projection score column
    score_col = None
    for candidate_col in ["fantasy", "Projection", "projection", "Points", "points", "FPPG", "fppg"]:
        if candidate_col in proj_df.columns:
            score_col = candidate_col
            break

    if not score_col:
        logger.warning("Could not determine fantasy score column in %s; skipping projection merge.", proj_path)
        return 0, 0

    player_map: Dict[str, float] = {}
    def_map: Dict[str, float] = {}

    player_col = "player" if "player" in proj_df.columns else "Player"
    team_col = "team" if "team" in proj_df.columns else "Team"

    for _, row in proj_df.iterrows():
        p_val = str(row[player_col]).strip() if player_col in row and pd.notna(row[player_col]) else ""
        t_val = str(row[team_col]).strip() if team_col in row and pd.notna(row[team_col]) else ""
        try:
            val = float(row[score_col]) if pd.notna(row[score_col]) else 0.0
        except (ValueError, TypeError):
            val = 0.0

        if p_val.endswith("D/ST") or "D/ST" in p_val:
            def_map[normalize_name(t_val)] = val
            clean_team = p_val.replace("D/ST", "").strip()
            def_map[normalize_name(clean_team)] = val
        else:
            player_map[normalize_name(p_val)] = val

    updated_count = 0
    zeroed_count = 0

    for player in optimizer.player_pool.all_players:
        is_def = "D" in player.positions or player.positions == ["D"]
        if is_def:
            d_key = normalize_name(player.full_name)
            t_key = normalize_name(player.team) if player.team else ""
            # Check full name, team code, and word components (e.g. "chiefs")
            found = False
            for cand in [d_key, t_key] + d_key.split():
                if cand in def_map:
                    player.fppg = def_map[cand]
                    updated_count += 1
                    found = True
                    break
            if not found:
                logger.warning("Defense %s not found in projections.", player.full_name)
        else:
            k = normalize_name(player.full_name)
            if k in player_map:
                player.fppg = player_map[k]
                updated_count += 1
            else:
                player.fppg = 0.0
                zeroed_count += 1

    logger.info(
        "Successfully updated forward-looking projections for %d active players/defenses (%d unprojected set to 0.0).",
        updated_count,
        zeroed_count,
    )
    return updated_count, zeroed_count
