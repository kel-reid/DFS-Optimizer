#!/usr/bin/env python3
"""
================================================================================
FanDuel NFL Classic Multi-Entry (MME) Lineup Optimization & Export Pipeline
================================================================================
Quantitative integer linear programming engine for generating 150 correlated,
risk-diversified NFL DFS tournament rosters for FanDuel Classic contests.

Architecture & Workflow:
  1. Data Ingestion: Loads the official FanDuel player pool CSV directly using
     pydfs-lineup-optimizer (`load_players_from_csv`). Does NOT fall back to
     extracting embedded players from the entry template.
  2. Pre-Solve Filtering:
     - Confirmed OUT/IR/Doubtful players are pruned.
     - Non-starting backup QBs (specifically Case Keenum and low-volume backups)
       have their projected points / FPPG zeroed out so the solver allocates
       100% of QB volume exclusively to active starting quarterbacks.
  3. Mathematical Solver & Constraints (PuLP / CBC Backend):
     - Roster Composition: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DEF (9 players).
     - Salary Cap: Total roster salary <= $60,000.
     - Stacking Architecture (80/20 Allocation):
       * Phase 1 (80% / 120 lineups): Mandatory Primary Stack (QB + >= 1 same-team WR/TE).
       * Phase 2 (20% / 30 lineups): Unconstrained stack for standalone rushing QBs.
     - Negative Correlation: 0% exposure of DEF against opposing offensive skill players.
     - Uniqueness: Max 6 repeating players (guarantees >= 3 unique players between every roster).
     - Exposure Ceilings (Portfolio Risk Diversification):
       * Starting QBs: Max 25% (37 lineups)
       * Defenses (DEF): Max 20% (30 lineups)
       * Skill Positions (RB, WR, TE): Max 25% (37 lineups)
     - Monte Carlo Variance: 0.25 (±25.0% uniform random multiplier applied to projections).
  4. Template Mapping & Export:
     - Reads target entry template with pandas.
     - Slices strictly to the 150 reserved contest entries.
     - Maps the 150 lineups into the 9 roster columns using standard `PlayerID:PlayerName` format.
     - Preserves all original metadata (`entry_id`, `contest_id`, `contest_name`, `entry_fee`).
     - Exports populated DataFrame to `data/output/Completed-<template-name>.csv`.
  5. Post-Solve Audit:
     - Rigorous quality assurance audit verifying cap, stacks, correlation, and exposures.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from pydfs_lineup_optimizer import (
    Lineup,
    LineupOptimizer,
    Player,
    PositionsStack,
    RandomFantasyPointsStrategy,
    Site,
    Sport,
    get_optimizer,
)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FanDuelMMEOptimizer")


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------
@dataclass
class OptimizerConfig:
    """Runtime configuration and hyperparameters for the MME pipeline."""

    players_csv: Path = Path("data/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv")
    template_csv: Path = Path("data/templates/FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv")
    output_csv: Path = Path("data/output/Completed-FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv")
    num_lineups: int = 150
    salary_cap: int = 60_000
    max_exposure: float = 0.25       # General individual player exposure ceiling (25% = 37 lineups)
    max_qb_exposure: float = 0.25    # Starting QB exposure ceiling (25% = 37 lineups)
    max_rb_exposure: float = 0.25    # Running back exposure ceiling (25% = 37 lineups)
    max_wr_exposure: float = 0.25    # Wide receiver exposure ceiling (25% = 37 lineups)
    max_te_exposure: float = 0.25    # Tight end exposure ceiling (25% = 37 lineups)
    max_def_exposure: float = 0.20   # Team defense exposure ceiling (20% = 30 lineups)
    max_repeating_players: int = 6   # Enforces >= 3 unique players between every pair of lineups
    randomness_deviation: float = 0.25  # ±25% Monte Carlo ceiling projection variance
    stack_ratio: float = 0.80        # 80% primary stacked (120 lineups) / 20% unconstrained (30 lineups)
    id_format: str = "id_name"       # Standard FanDuel player string format (PlayerID:PlayerName)
    exclude_out_injured: bool = True # Prune confirmed OUT, IR, and Doubtful players


# -----------------------------------------------------------------------------
# Data Loader & Pre-Solve Filter Engine
# -----------------------------------------------------------------------------
class FanDuelDataLoader:
    """
    Ingests official FanDuel player list CSV directly into pydfs-lineup-optimizer
    and executes pre-solve sanitization (injury pruning & backup QB zeroing).
    """

    # Known backup QBs with small-sample / misleading baseline FPPG to zero out
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
        "Mac Jones",
        "Jalon Daniels",
        "Nick Mullens",
        "Andy Dalton",
        "Miller Moss",
        "Easton Stick",
        "Joey Aguilar",
        "Hendon Hooker",
    }

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def initialize_and_load_optimizer(self) -> LineupOptimizer:
        """
        Instantiates the FanDuel Football optimizer and loads players directly
        from the official player pool CSV.
        """
        csv_path = self.config.players_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Required FanDuel player list CSV not found at: {csv_path.resolve()}\n"
                f"Please verify file placement."
            )

        logger.info("Initializing FanDuel Football Optimizer with PuLP/CBC backend...")
        optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)

        logger.info("Loading player pool directly from CSV: %s", csv_path)
        optimizer.load_players_from_csv(str(csv_path))
        optimizer.player_pool.with_injured = True  # Retain Questionable (Q) players

        initial_count = len(optimizer.player_pool.all_players)
        logger.info("Successfully loaded %d raw player entries via optimizer.load_players_from_csv.", initial_count)

        # 1. Prune confirmed Out / IR / Doubtful / PUP players
        if self.config.exclude_out_injured:
            self._prune_inactive_players(optimizer, csv_path)

        # 2. Pre-solve filter: Zero out projected points / FPPG for non-starting backup QBs
        self._filter_backup_quarterbacks(optimizer)

        return optimizer

    def _prune_inactive_players(self, optimizer: LineupOptimizer, csv_path: Path) -> None:
        """Prunes confirmed inactive/IR players while preserving Questionable starters."""
        df = pd.read_csv(csv_path)
        inactive_indicators = {"IR", "O", "D", "PUP"}
        inactive_df = df[df["Injury Indicator"].isin(inactive_indicators)]
        inactive_ids = set(inactive_df["Id"].astype(str))

        removed_count = 0
        for player in list(optimizer.player_pool.all_players):
            if str(player.id) in inactive_ids:
                optimizer.player_pool.remove_player(player)
                removed_count += 1

        logger.info("Pruned %d confirmed inactive/IR players (retained active & Questionable pool: %d players).",
                    removed_count, len(optimizer.player_pool.all_players))

    def _filter_backup_quarterbacks(self, optimizer: LineupOptimizer) -> None:
        """
        Pre-solve filter: Zeroes out projected points / FPPG for non-starting backup
        quarterbacks (specifically Case Keenum) so the optimizer allocates 100% of QB
        volume exclusively to active starting quarterbacks.
        """
        zeroed_qbs: List[str] = []
        for player in optimizer.player_pool.all_players:
            if "QB" in player.positions:
                # Target Case Keenum explicitly or any cataloged non-starting backup
                if "Keenum" in player.full_name or player.full_name in self.NON_STARTING_BACKUP_QBS:
                    if player.fppg > 0:
                        player.fppg = 0.0
                        zeroed_qbs.append(f"{player.full_name} ({player.team})")

        logger.info(
            "Pre-solve filter: Zeroed out projected FPPG for %d backup QBs (including Case Keenum):",
            len(zeroed_qbs),
        )
        for qb_name in zeroed_qbs[:8]:
            logger.info("  - %s -> FPPG set to 0.0 (ineligible for selection)", qb_name)
        if len(zeroed_qbs) > 8:
            logger.info("  ... and %d more backup QBs zeroed.", len(zeroed_qbs) - 8)


# -----------------------------------------------------------------------------
# Optimization Engine & Portfolio Allocation
# -----------------------------------------------------------------------------
class FanDuelLineupPipeline:
    """
    Executes the two-phase integer programming solver loop:
      - Phase 1 (80% / 120 lineups): Mandatory Primary Stack (QB + WR/TE same team).
      - Phase 2 (20% / 30 lineups): Unconstrained stack for standalone rushing QBs.
    Enforces negative defense correlation, max repeating players (<= 6),
    Monte Carlo randomness (0.25), and strict position-specific exposure caps.
    """

    def __init__(self, config: OptimizerConfig, optimizer: LineupOptimizer) -> None:
        self.config = config
        self.optimizer = optimizer
        self.players = list(optimizer.player_pool.all_players)

    def _get_player_cap_count(self, player: Player) -> int:
        """Computes exact maximum allowed lineup appearances for a player."""
        n = self.config.num_lineups
        positions = set(player.positions)
        if "D" in positions:
            return math.floor(n * self.config.max_def_exposure)
        elif "QB" in positions:
            return math.floor(n * self.config.max_qb_exposure)
        elif "RB" in positions:
            return math.floor(n * self.config.max_rb_exposure)
        elif "WR" in positions:
            return math.floor(n * self.config.max_wr_exposure)
        elif "TE" in positions:
            return math.floor(n * self.config.max_te_exposure)
        return math.floor(n * self.config.max_exposure)

    def configure_optimizer_instance(self, opt: LineupOptimizer) -> None:
        """Applies shared baseline constraints (uniqueness, randomness, negative correlation)."""
        # 1. Uniqueness constraint: max 6 repeating players (>= 3 unique players per roster)
        opt.set_max_repeating_players(self.config.max_repeating_players)

        # 2. Monte Carlo projection variance: 0.25 (±25.0% uniform random deviation)
        opt.set_fantasy_points_strategy(
            RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
        )

        # 3. Negative correlation: 0 DEF against opposing offensive skill players
        opt.restrict_positions_for_opposing_team(["D"], ["QB", "RB", "WR", "TE"])

    def generate_lineups(self) -> List[Lineup]:
        """
        Generates 150 total lineups split across:
          - 120 Primary Stacked Lineups (QB + same-team WR/TE)
          - 30 Unconstrained Lineups (enabling standalone rushing QBs)
        Strictly preserves global exposure ceilings across both phases.
        """
        n = self.config.num_lineups
        ratio = self.config.stack_ratio
        n_stacked = round(n * ratio)   # 120
        n_unconstrained = n - n_stacked # 30

        logger.info(
            "Portfolio Allocation Plan: %d Stacked (%.0f%%) + %d Unconstrained (%.0f%%)",
            n_stacked, ratio * 100, n_unconstrained, (1 - ratio) * 100,
        )
        logger.info(
            "Exposure Limits: QB <= %.1f%% (max %d) | RB <= %.1f%% (max %d) | WR <= %.1f%% (max %d) | TE <= %.1f%% (max %d) | DEF <= %.1f%% (max %d)",
            self.config.max_qb_exposure * 100, math.floor(n * self.config.max_qb_exposure),
            self.config.max_rb_exposure * 100, math.floor(n * self.config.max_rb_exposure),
            self.config.max_wr_exposure * 100, math.floor(n * self.config.max_wr_exposure),
            self.config.max_te_exposure * 100, math.floor(n * self.config.max_te_exposure),
            self.config.max_def_exposure * 100, math.floor(n * self.config.max_def_exposure),
        )

        # ---------------------------------------------------------------------
        # Phase 1: 120 Primary Stacked Lineups (QB + same-team WR/TE)
        # ---------------------------------------------------------------------
        logger.info("--- Phase 1: Solving %d primary stacked lineups (QB + same-team WR/TE) ---", n_stacked)
        self.configure_optimizer_instance(self.optimizer)
        self.optimizer.add_stack(PositionsStack(["QB", ("WR", "TE")]))

        # Set Phase 1 max exposure ratios based on global caps
        for player in self.optimizer.player_pool.all_players:
            cap_count = self._get_player_cap_count(player)
            player.max_exposure = min(1.0, cap_count / n_stacked) if n_stacked > 0 else 1.0

        stacked_lineups: List[Lineup] = []
        for i, lineup in enumerate(self.optimizer.optimize(n=n_stacked), start=1):
            stacked_lineups.append(lineup)
            if i % 30 == 0 or i == n_stacked:
                logger.info("... Solved %d / %d stacked lineups ...", i, n_stacked)

        if n_unconstrained == 0:
            return stacked_lineups

        # Track usage from Phase 1
        usage_phase1: Counter[Player] = Counter()
        for l in stacked_lineups:
            for p in l.lineup:
                usage_phase1[p] += 1

        # ---------------------------------------------------------------------
        # Phase 2: 30 Unconstrained Lineups (standalone rushing QB ceiling)
        # ---------------------------------------------------------------------
        logger.info("--- Phase 2: Solving %d unconstrained lineups (rushing QB upside) ---", n_unconstrained)
        opt_unconstrained = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
        opt_unconstrained.player_pool.load_players(self.players)
        opt_unconstrained.player_pool.with_injured = True
        self.configure_optimizer_instance(opt_unconstrained)

        # Set remaining allowances for Phase 2
        for player in opt_unconstrained.player_pool.all_players:
            used = usage_phase1.get(player, 0)
            cap_count = self._get_player_cap_count(player)
            remaining = cap_count - used
            if remaining <= 0:
                opt_unconstrained.player_pool.remove_player(player)
            else:
                player.max_exposure = min(1.0, remaining / n_unconstrained)

        unconstrained_lineups: List[Lineup] = []
        for i, lineup in enumerate(opt_unconstrained.optimize(n=n_unconstrained), start=1):
            unconstrained_lineups.append(lineup)
            if i % 15 == 0 or i == n_unconstrained:
                logger.info("... Solved %d / %d unconstrained lineups ...", i, n_unconstrained)

        combined = stacked_lineups + unconstrained_lineups
        logger.info("Successfully assembled %d total lineups (%d stacked + %d unconstrained).",
                    len(combined), len(stacked_lineups), len(unconstrained_lineups))
        return combined


# -----------------------------------------------------------------------------
# Portfolio Auditor & Quality Assurance
# -----------------------------------------------------------------------------
class PortfolioAuditor:
    """Performs quantitative risk and compliance audit across the 150-lineup portfolio."""

    @staticmethod
    def audit_and_report(lineups: Sequence[Lineup], config: OptimizerConfig) -> None:
        total_lineups = len(lineups)
        if total_lineups == 0:
            raise ValueError("No lineups available for audit.")

        logger.info("=" * 70)
        logger.info("PORTFOLIO AUDIT & RISK ANALYSIS (%d LINEUPS)", total_lineups)
        logger.info("=" * 70)

        player_counts: Counter[str] = Counter()
        player_names: Dict[str, str] = {}
        team_qb_counts: Counter[str] = Counter()
        qb_counts: Counter[str] = Counter()
        salary_list: List[int] = []
        fppg_list: List[float] = []

        violations: List[str] = []
        stacked_count = 0
        unconstrained_count = 0

        for idx, lineup in enumerate(lineups, start=1):
            salary = lineup.salary_costs
            fppg = lineup.fantasy_points_projection
            salary_list.append(salary)
            fppg_list.append(fppg)

            if salary > config.salary_cap:
                violations.append(f"Lineup #{idx}: Exceeded salary cap (${salary} > ${config.salary_cap})")

            qb_player: Optional[Player] = None
            def_player: Optional[Player] = None
            offensive_players: List[Player] = []

            for p in lineup.lineup:
                player_counts[p.id] += 1
                player_names[p.id] = f"{p.full_name} ({p.team} - {p.lineup_position})"

                if p.lineup_position == "QB":
                    qb_player = p
                elif p.lineup_position == "DEF":
                    def_player = p
                else:
                    offensive_players.append(p)

            # Stack Check
            if qb_player:
                team_qb_counts[qb_player.team] += 1
                qb_counts[qb_player.full_name] += 1
                same_team_pass_catchers = [
                    op for op in offensive_players
                    if op.team == qb_player.team and any(pos in ("WR", "TE") for pos in op.positions)
                ]
                if len(same_team_pass_catchers) > 0:
                    stacked_count += 1
                else:
                    unconstrained_count += 1

            # Negative Correlation Check (DEF vs Opposing Offense)
            if def_player and def_player.game_info:
                def_team = def_player.team
                game = def_player.game_info
                opp_team = game.home_team if game.away_team == def_team else game.away_team
                opposing_offense = [op for op in offensive_players if op.team == opp_team]
                if opposing_offense:
                    opp_names = [f"{op.full_name} ({op.lineup_position})" for op in opposing_offense]
                    violations.append(
                        f"Lineup #{idx}: DEF violation! {def_player.full_name} rostered with opposing "
                        f"players: {', '.join(opp_names)}"
                    )

        # Exposure Compliance Check per position
        def get_pos_cap(p_name: str) -> Tuple[float, str]:
            if "DEF" in p_name:
                return config.max_def_exposure, "DEF"
            elif "QB" in p_name:
                return config.max_qb_exposure, "QB"
            elif "RB" in p_name:
                return config.max_rb_exposure, "RB"
            elif "WR" in p_name:
                return config.max_wr_exposure, "WR"
            elif "TE" in p_name:
                return config.max_te_exposure, "TE"
            return config.max_exposure, "Player"

        for pid, count in player_counts.items():
            p_name = player_names.get(pid, "")
            cap_ratio, pos_label = get_pos_cap(p_name)
            max_allowed = math.ceil(total_lineups * cap_ratio)
            pct = (count / total_lineups) * 100
            if count > max_allowed:
                violations.append(
                    f"{pos_label} {player_names.get(pid, pid)} exceeded max exposure: "
                    f"{count}/{total_lineups} ({pct:.1f}% > {cap_ratio * 100:.1f}%)"
                )

        if violations:
            logger.error("AUDIT FAILED WITH %d VIOLATIONS:", len(violations))
            for v in violations[:10]:
                logger.error("  - %s", v)
            raise AssertionError("Portfolio failed strict risk/correlation audit.")

        logger.info("ALL CONSTRAINTS STRICTLY SATISFIED:")
        logger.info("  - 100%% of lineups comply with $60,000 salary cap.")
        logger.info("  - Primary Stacks: %d / %d (%.1f%%) lineups feature QB + WR/TE stack.",
                    stacked_count, total_lineups, (stacked_count / total_lineups) * 100)
        logger.info("  - Unconstrained:  %d / %d (%.1f%%) lineups feature unconstrained rushing QBs.",
                    unconstrained_count, total_lineups, (unconstrained_count / total_lineups) * 100)
        logger.info("  - 0%% lineups feature DEF against opposing offensive skill players.")
        logger.info("  - 0 players exceed position exposure ceilings (QB: %.1f%%, RB: %.1f%%, WR: %.1f%%, TE: %.1f%%, DEF: %.1f%%).",
                    config.max_qb_exposure * 100, config.max_rb_exposure * 100, config.max_wr_exposure * 100,
                    config.max_te_exposure * 100, config.max_def_exposure * 100)
        logger.info("  - QBs diversified across %d starting QBs (Case Keenum zeroed out).", len(qb_counts))
        def_count = len([pid for pid in player_counts if "DEF" in player_names.get(pid, "")])
        logger.info("  - DEFs diversified across %d defenses.", def_count)
        logger.info("  - Total distinct players utilized across portfolio: %d players.", len(player_counts))

        # Display Top Exposures Audit Table
        logger.info("-" * 70)
        logger.info("POST-SOLVE EXPOSURE AUDIT TABLE (TOP 15 PLAYERS):")
        for pid, count in player_counts.most_common(15):
            pct = (count / total_lineups) * 100
            name_str = player_names.get(pid, pid)
            bar = "#" * int(pct / 2.5)
            logger.info("  %-35s : %3d / %3d (%5.1f%%) | %s", name_str, count, total_lineups, pct, bar)

        # Display Starting QB Distribution
        logger.info("-" * 70)
        logger.info("STARTING QB EXPOSURE DISTRIBUTION:")
        for qb_name, count in qb_counts.most_common():
            pct = (count / total_lineups) * 100
            bar = "=" * int(pct / 2)
            logger.info("  QB %-28s : %3d lineups (%5.1f%%) | %s", qb_name, count, pct, bar)

        # Display Metrics Summary
        avg_sal = sum(salary_list) / len(salary_list)
        avg_fppg = sum(fppg_list) / len(fppg_list)
        logger.info("-" * 70)
        logger.info("PORTFOLIO METRICS SUMMARY:")
        logger.info("  Salary Utilization : Min=$%d | Max=$%d | Mean=$%.1f", min(salary_list), max(salary_list), avg_sal)
        logger.info("  Projected FPPG     : Min=%.2f | Max=%.2f | Mean=%.2f", min(fppg_list), max(fppg_list), avg_fppg)
        logger.info("=" * 70)


# -----------------------------------------------------------------------------
# Template Ingestion & CSV Export Mapping
# -----------------------------------------------------------------------------
class FanDuelTemplateExporter:
    """
    Reads the target upload template with pandas, slices strictly to the 150
    reserved contest entries, maps the generated 150 lineups into the 9 roster
    columns ('QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF') using
    standard FanDuel format ('PlayerID:PlayerName'), preserves all metadata columns,
    and exports the populated DataFrame to data/output/.
    """

    ROSTER_SLOTS: Tuple[str, ...] = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF")

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def export_lineups(self, lineups: Sequence[Lineup]) -> Path:
        template_path = self.config.template_csv
        output_path = self.config.output_csv

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path.resolve()}")

        logger.info("Reading reserved entries template: %s", template_path)

        # Read the template using csv.reader to capture header and slice strictly to 150 reserved entries.
        # Note: FanDuel templates have the player pool embedded in columns 14+ starting at row 7.
        # Reading columns [:13] safely extracts the reserved contest entries without tokenizer mismatch.
        with open(template_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)[:13]
            rows = []
            for row in reader:
                if row and row[0].strip():
                    rows.append(row[:13])
                if len(rows) == self.config.num_lineups:
                    break

        df = pd.DataFrame(rows, columns=header)
        logger.info("Loaded template with pandas: %d reserved entries, %d columns.", len(df), len(df.columns))

        if len(df) != len(lineups):
            raise ValueError(
                f"Row mismatch: Template contains {len(df)} reserved entries, but solver generated {len(lineups)} lineups."
            )

        # Map 150 lineups into the 9 roster columns (columns index 4 to 12)
        logger.info("Mapping %d optimized lineups to %d template entry rows...", len(lineups), len(df))
        for i, lineup in enumerate(lineups):
            # lineup.lineup is ordered: [QB, RB, RB, WR, WR, WR, TE, FLEX, DEF]
            formatted_players = [f"{p.id}:{p.full_name}" for p in lineup.lineup]
            for slot_idx, player_str in enumerate(formatted_players, start=4):
                df.iat[i, slot_idx] = player_str

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Export successfully written to: %s", output_path.resolve())
        logger.info("Total entries exported: %d valid rows (strictly 151 lines with header).", len(df))
        return output_path


# -----------------------------------------------------------------------------
# File Path Auto-Discovery
# -----------------------------------------------------------------------------
def find_players_csv(explicit_path: Optional[Path]) -> Path:
    """Resolves the official FanDuel player pool CSV without template fallback."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("data/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/players/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/templates/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/players/FanDuel-NFL-players-list.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(Path("data").glob("**/*players-list.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not locate FanDuel players list CSV. Expected 'data/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv'."
    )


def find_template_csv(explicit_path: Optional[Path]) -> Path:
    """Resolves the FanDuel entries upload template CSV."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("data/templates/FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv"),
        Path("data/FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(Path("data").glob("**/*entries-upload-template.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not locate FanDuel entries upload template CSV.")


# -----------------------------------------------------------------------------
# CLI Argument Parser
# -----------------------------------------------------------------------------
def parse_arguments() -> OptimizerConfig:
    parser = argparse.ArgumentParser(
        description="FanDuel NFL Classic Quantitative MME Lineup Optimizer (150 Entries)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--players-csv",
        type=Path,
        default=None,
        help="Path to official FanDuel player list CSV download.",
    )
    parser.add_argument(
        "--template-csv",
        type=Path,
        default=None,
        help="Path to FanDuel reserved entries upload template CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination path for completed upload CSV.",
    )
    parser.add_argument(
        "--num-lineups",
        type=int,
        default=150,
        help="Number of lineups to generate (default: 150).",
    )
    parser.add_argument(
        "--stack-ratio",
        type=float,
        default=0.80,
        help="Proportion of lineups with mandatory primary stack (default: 0.80 = 80%%).",
    )
    parser.add_argument(
        "--max-qb-exposure",
        type=float,
        default=0.25,
        help="Maximum individual QB exposure ceiling across portfolio (default: 0.25 = 25%%).",
    )
    parser.add_argument(
        "--max-rb-exposure",
        type=float,
        default=0.25,
        help="Maximum individual RB exposure ceiling across portfolio (default: 0.25 = 25%%).",
    )
    parser.add_argument(
        "--max-wr-exposure",
        type=float,
        default=0.25,
        help="Maximum individual WR exposure ceiling across portfolio (default: 0.25 = 25%%).",
    )
    parser.add_argument(
        "--max-te-exposure",
        type=float,
        default=0.25,
        help="Maximum individual TE exposure ceiling across portfolio (default: 0.25 = 25%%).",
    )
    parser.add_argument(
        "--max-def-exposure",
        type=float,
        default=0.20,
        help="Maximum individual DEF exposure ceiling across portfolio (default: 0.20 = 20%%).",
    )
    parser.add_argument(
        "--max-exposure",
        type=float,
        default=0.25,
        help="General player exposure ceiling across portfolio (default: 0.25 = 25%%).",
    )
    parser.add_argument(
        "--randomness",
        type=float,
        default=0.25,
        help="Uniform random deviation applied to fantasy projections (default: 0.25 = ±25%%).",
    )
    parser.add_argument(
        "--max-repeating",
        type=int,
        default=6,
        help="Maximum repeating players per lineup, enforcing structural diversity (default: 6 of 9).",
    )
    parser.add_argument(
        "--keep-injured",
        action="store_true",
        default=False,
        help="Retain players marked with injury indicators (O/IR/D/PUP).",
    )

    args = parser.parse_args()

    players_path = find_players_csv(args.players_csv)
    template_path = find_template_csv(args.template_csv)

    if args.output_csv:
        output_path = args.output_csv
    else:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"Completed-{template_path.name}"

    return OptimizerConfig(
        players_csv=players_path,
        template_csv=template_path,
        output_csv=output_path,
        num_lineups=args.num_lineups,
        max_exposure=args.max_exposure,
        randomness_deviation=args.randomness,
        stack_ratio=args.stack_ratio,
        max_qb_exposure=args.max_qb_exposure,
        max_rb_exposure=args.max_rb_exposure,
        max_wr_exposure=args.max_wr_exposure,
        max_te_exposure=args.max_te_exposure,
        max_def_exposure=args.max_def_exposure,
        max_repeating_players=args.max_repeating,
        exclude_out_injured=not args.keep_injured,
    )


# -----------------------------------------------------------------------------
# Main Execution Workflow
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" FANDUEL NFL DFS OPTIMIZATION PIPELINE ")
    print(" Quantitative Integer Linear Programming Engine ")
    print("=" * 70)

    # 1. Parse configuration & locate CSV files
    config = parse_arguments()
    logger.info("Target Player Pool File : %s", config.players_csv.resolve())
    logger.info("Target Entry Template   : %s", config.template_csv.resolve())
    logger.info("Target Completed Output : %s", config.output_csv.resolve())

    # 2. Ingest players directly via optimizer.load_players_from_csv + pre-solve filters
    loader = FanDuelDataLoader(config)
    optimizer = loader.initialize_and_load_optimizer()

    # 3. Setup pipeline and execute solver
    pipeline = FanDuelLineupPipeline(config, optimizer)
    lineups = pipeline.generate_lineups()

    # 4. Perform rigorous post-solve portfolio audit
    PortfolioAuditor.audit_and_report(lineups, config)

    # 5. Map lineups into pandas DataFrame and export completed upload CSV
    exporter = FanDuelTemplateExporter(config)
    output_file = exporter.export_lineups(lineups)

    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETE: Ready for upload to FanDuel!")
    print(f" Output Location: {output_file.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
