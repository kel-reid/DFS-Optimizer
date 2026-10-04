"""
================================================================================
FanDuel Data Loader Module (src/data/loader.py)
================================================================================
Handles player pool CSV ingestion, inactive/IR pruning, backup QB filtering,
and CSV file discovery for FanDuel slates.
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from pydfs_lineup_optimizer import (
    LineupOptimizer,
    Site,
    Sport,
    get_optimizer,
)

from src.config import SimOptimizerConfig
from src.data.projections import apply_forward_projections

logger = logging.getLogger("FanDuelSimOptimizer")


def find_players_csv(explicit_path: Optional[Path] = None) -> Path:
    """Discovers FanDuel player pool CSV file from explicit path or common search locations."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("data/players/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/templates/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = sorted(Path("data").glob("**/*players-list.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not locate FanDuel players list CSV in data/ directory.")


def find_template_csv(explicit_path: Optional[Path] = None) -> Path:
    """Discovers FanDuel contest entries template CSV from explicit path or common search locations."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("data/templates/FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv"),
        Path("data/FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = sorted(Path("data").glob("**/*entries*template*.csv"))
    if not matches:
        matches = sorted(Path("data").glob("**/*template*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not locate FanDuel entries upload template CSV in data/ directory.")


class FanDuelDataLoader:
    """
    Ingests official FanDuel player pool CSV directly into pydfs-lineup-optimizer
    and executes pre-solve sanitization (injury pruning & backup QB zeroing).
    """

    NON_STARTING_BACKUP_QBS: Set[str] = {
        "Case Keenum",
        "Drew Lock",
        "Josh Johnson",
        "Carson Wentz",
        "Jameis Winston",
        "Shane Buechele",
        "Joe Milton III",
        "Tommy DeVito",
        "Max Brosmer",
        "Stetson Bennett IV",
        "Sean Clifford",
        "Davis Mills",
        "Tanner McKee",
        "Quinn Ewers",
        "Garrett Nussmeier",
        "Jarrett Stidham",
        "Fernando Mendoza",
        "Tyrod Taylor",
        "Kyle McCord",
        "Sam Howell",
        "Trey Lance",
        "Behren Morton",
        "J.J. McCarthy",
        "Gardner Minshew II",
        "Sam Ehlinger",
        "Tyler Huntley",
        "Cade Klubnik",
        "Justin Fields",
        "Kedon Slovis",
    }

    def __init__(self, config: SimOptimizerConfig) -> None:
        self.config = config

    def load_and_sanitize(self) -> LineupOptimizer:
        csv_path = self.config.players_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"Player pool CSV not found: {csv_path.resolve()}")

        optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
        logger.info("Loading player pool from CSV: %s", csv_path)
        optimizer.load_players_from_csv(str(csv_path))
        optimizer.player_pool.with_injured = True

        raw_count = len(optimizer.player_pool.all_players)
        logger.info("Successfully loaded %d raw player entries via optimizer.load_players_from_csv.", raw_count)

        if self.config.exclude_out_injured:
            self._prune_inactive_players(optimizer, csv_path)

        if self.config.projections_csv and self.config.projections_csv.exists():
            self._apply_external_projections(optimizer, self.config.projections_csv)
        self._filter_backup_quarterbacks(optimizer)


        return optimizer

    def _prune_inactive_players(self, optimizer: LineupOptimizer, csv_path: Path) -> None:
        """Prunes confirmed inactive/IR players while preserving Questionable starters."""
        df = pd.read_csv(csv_path)
        if "Injury Indicator" in df.columns and "Id" in df.columns:
            inactive_indicators = {"IR", "O", "D", "PUP"}
            inactive_df = df[df["Injury Indicator"].isin(inactive_indicators)]
            inactive_ids = set(inactive_df["Id"].astype(str))

            removed_count = 0
            for player in list(optimizer.player_pool.all_players):
                if str(player.id) in inactive_ids:
                    optimizer.player_pool.remove_player(player)
                    removed_count += 1

            logger.info(
                "Pruned %d confirmed inactive/IR players (retained active & Questionable pool: %d players).",
                removed_count,
                len(optimizer.player_pool.all_players),
            )

    def _filter_backup_quarterbacks(self, optimizer: LineupOptimizer) -> None:
        """
        Pre-solve filter: Zeroes out projected points / FPPG for non-starting backup
        quarterbacks (specifically Case Keenum, Drew Lock, etc.) so only active starters
        are eligible for selection.
        """
        for player in optimizer.player_pool.all_players:
            if "QB" in player.positions:
                if player.full_name in self.NON_STARTING_BACKUP_QBS or player.full_name == "Case Keenum":
                    player.fppg = 0.0

    def _apply_external_projections(self, optimizer: LineupOptimizer, proj_path: Path) -> None:
        """Applies weekly forward-looking projections to the player pool."""
        apply_forward_projections(optimizer, proj_path)
