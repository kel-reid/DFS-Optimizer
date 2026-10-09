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
import re
from pathlib import Path
from typing import Optional, Set, Union

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


def normalize_week(week: Optional[Union[str, int]]) -> Optional[str]:
    """
    Normalizes week input (e.g. 5, '5', '05', 'week-5', 'week-05') to standard 'week-05' zero-padded format.
    """
    if week is None:
        return None
    s = str(week).strip().lower()
    if not s:
        return None
    m = re.search(r"\d+", s)
    if m:
        return f"week-{int(m.group(0)):02d}"
    return s


def _find_in_slate_dirs(
    slate: str,
    patterns: list[str],
    week: Optional[Union[str, int]] = None,
    slate_date: Optional[str] = None,
    legacy_folder: Optional[str] = None,
) -> Optional[Path]:
    search_dirs: list[Path] = []
    norm_week = normalize_week(week)
    if norm_week:
        search_dirs.append(Path(f"data/{norm_week}/{slate}"))
    if slate_date:
        search_dirs.append(Path(f"data/{slate_date}/{slate}"))

    # Search week-based directories in descending order (e.g. week-18 down to week-01)
    for p in sorted(Path("data").glob(f"week-*/{slate}"), reverse=True):
        if p not in search_dirs:
            search_dirs.append(p)

    # Search any remaining slate directories
    for p in sorted(Path("data").glob(f"week-*/{slate}"), reverse=True):
        if p not in search_dirs:
            search_dirs.append(p)

    search_dirs.append(Path(f"data/{slate}"))
    if legacy_folder:
        search_dirs.append(Path(f"data/{legacy_folder}/{slate}"))

    for s_dir in search_dirs:
        if s_dir.is_dir():
            for pat in patterns:
                matches = sorted(s_dir.glob(pat))
                if matches:
                    return matches[0]
    return None


def find_players_csv(
    explicit_path: Optional[Path] = None,
    slate: Optional[str] = None,
    week: Optional[Union[str, int]] = None,
    slate_date: Optional[str] = None,
) -> Path:
    """Discovers FanDuel player pool CSV file from explicit path, data/{week}/{slate}/, or common search locations."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    if slate:
        found = _find_in_slate_dirs(slate, ["*player*.csv", "players.csv", "*.csv"], week, slate_date, "players")
        if found:
            return found

    candidates = [
        Path("data/week-05/main-slate/players.csv"),
        Path("data/week-05/main-slate/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/week-05/sunday-night/players.csv"),
        Path("data/week-05/monday-night/players.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = sorted(Path("data").glob("**/*players-list.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not locate FanDuel players list CSV in data/ directory.")


def find_template_csv(
    explicit_path: Optional[Path] = None,
    slate: Optional[str] = None,
    week: Optional[Union[str, int]] = None,
    slate_date: Optional[str] = None,
) -> Path:
    """Discovers FanDuel contest entries template CSV from explicit path, data/{week}/{slate}/, or common search locations."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    if slate:
        found = _find_in_slate_dirs(
            slate,
            ["*entries*template*.csv", "*template*.csv", "*entries*.csv", "entries_template.csv"],
            week,
            slate_date,
            "templates",
        )
        if found:
            return found

    candidates = [
        Path("data/week-05/main-slate/entries_template.csv"),
        Path("data/week-05/main-slate/Completed-FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv"),
        Path("data/week-05/sunday-night/entries_template.csv"),
        Path("data/week-05/monday-night/entries_template.csv"),
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

        is_single_game = self._detect_single_game(csv_path)
        self.config.is_single_game = is_single_game
        site = Site.FANDUEL_SINGLE_GAME if is_single_game else Site.FANDUEL

        optimizer = get_optimizer(site, Sport.FOOTBALL)
        logger.info(
            "Loading player pool from CSV (%s): %s",
            "Single Game" if is_single_game else "Classic",
            csv_path,
        )
        optimizer.load_players_from_csv(str(csv_path))
        optimizer.player_pool.with_injured = True
        optimizer.settings.budget = self.config.salary_cap

        raw_count = len(optimizer.player_pool.all_players)
        logger.info("Successfully loaded %d raw player entries via optimizer.load_players_from_csv.", raw_count)

        if self.config.exclude_out_injured:
            self._prune_inactive_players(optimizer, csv_path)

        if self.config.projections_csv and self.config.projections_csv.exists():
            self._apply_external_projections(optimizer, self.config.projections_csv)

        if is_single_game:
            # Enforce 1.5x MVP projection multiplier for Single Game
            for player in optimizer.player_pool.all_players:
                if "MVP" in player.positions:
                    player.fppg *= 1.5
        else:
            self._filter_backup_quarterbacks(optimizer)

        return optimizer

    def _detect_single_game(self, csv_path: Path) -> bool:
        """Detects whether the target contest format is Single Game / Showdown vs Classic."""
        if self.config.is_single_game:
            return True
        if self.config.slate in ("sunday-night", "monday-night", "single-game", "showdown"):
            return True
        if self.config.template_csv and self.config.template_csv.exists():
            try:
                with open(self.config.template_csv, "r", encoding="utf-8-sig") as f:
                    header = f.readline()
                    if "MVP" in header or "AnyFLEX" in header:
                        return True
            except Exception:
                pass
        try:
            df = pd.read_csv(csv_path)
            if "Roster Position" in df.columns:
                roster_pos = df["Roster Position"].dropna().astype(str)
                if any("MVP" in p or "AnyFLEX" in p for p in roster_pos):
                    return True
        except Exception:
            pass
        return False

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
                len(optimizer.player_pool.filtered_players),
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
        """Applies weekly forward-looking projections to the player pool (SaberSim zero_unprojected standard)."""
        apply_forward_projections(optimizer, proj_path, zero_unprojected=self.config.zero_unprojected)
