#!/usr/bin/env python3
"""
================================================================================
DraftKings NFL Classic Multi-Entry (MME) Lineup Optimization & Export Pipeline
================================================================================
Quantitative integer linear programming engine for generating 150 correlated,
risk-diversified NFL DFS tournament rosters for DraftKings Classic contests.

Architecture & Workflow:
  1. Data Ingestion: Loads official DraftKings `DKSalaries.csv` directly using
     `optimizer.load_players_from_csv(...)`.
  2. Pre-Solve Filtering:
     - Confirmed inactive players pruned.
     - Non-starting backup QBs have their projected points / FPPG zeroed out.
  3. Mathematical Solver & Constraints (PuLP / CBC Backend):
     - Roster Composition: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST (9 players).
     - Salary Cap: Total roster salary <= $50,000.
     - Stacking Architecture (80/20 Allocation):
       * Phase 1 (80% / 120 lineups): Primary Correlation Stack (QB + >= 1 same-team WR/TE).
       * Phase 2 (20% / 30 lineups): Unconstrained stack for standalone rushing QBs.
     - Negative Correlation: 0% exposure of DST against opposing offensive skill players.
     - Uniqueness: Max 6 repeating players (guarantees >= 3 unique players between every roster).
     - Exposure Ceilings:
       * Starting QBs: Max 25% (37 lineups)
       * Defenses (DST): Max 20% (30 lineups)
       * Skill Positions (RB, WR, TE): Max 25% (37 lineups)
     - Monte Carlo Variance: 0.25 (±25.0% uniform random multiplier applied to projections).
  4. Template Mapping & Export:
     - Reads target `DKEntries.csv` template with pandas.
     - Slices strictly to the 150 reserved contest entries.
     - Maps the 150 lineups into the 9 roster columns using standard DraftKings format.
     - Preserves all original contest metadata columns.
     - Exports populated DataFrame to `data/output/Completed-<template-name>.csv`.
  5. Post-Solve Audit:
     - Rigorous quality assurance audit verifying $50k cap, stacks, correlation, and exposures.
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
from typing import Any, Dict, List, Optional, Sequence, Set, Union

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
from pydfs_lineup_optimizer.player import LineupPlayer

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DraftKingsMMEOptimizer")


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------
from src.config import load_yaml_settings


@dataclass
class DKOptimizerConfig:
    """Runtime configuration and hyperparameters for DraftKings MME pipeline."""

    players_csv: Path = Path("data/players/DKSalaries.csv")
    template_csv: Path = Path("data/templates/DKEntries.csv")
    output_csv: Path = Path("data/output/Completed-DKEntries.csv")
    num_lineups: int = 150
    salary_cap: int = 50_000         # DraftKings Classic $50,000 budget
    max_exposure: float = 0.25       # General player exposure ceiling (25% = 37 lineups)
    max_qb_exposure: float = 0.25    # Starting QB exposure ceiling (25% = 37 lineups)
    max_rb_exposure: float = 0.25    # Running back exposure ceiling (25% = 37 lineups)
    max_wr_exposure: float = 0.25    # Wide receiver exposure ceiling (25% = 37 lineups)
    max_te_exposure: float = 0.25    # Tight end exposure ceiling (25% = 37 lineups)
    max_def_exposure: float = 0.20   # Team defense (DST) ceiling (20% = 30 lineups)
    max_repeating_players: int = 6   # Enforces >= 3 unique players between every pair of lineups
    randomness_deviation: float = 0.25  # ±25% Monte Carlo ceiling projection variance
    stack_ratio: float = 0.80        # 80% primary stacked (120 lineups) / 20% unconstrained (30 lineups)
    id_format: str = "name_id"       # "name_id" ("Josh Allen (123456)") or "id_only" ("123456")
    exclude_out_injured: bool = True # Prune confirmed OUT, IR, and Doubtful players

    @classmethod
    def from_settings(cls, settings_path: Optional[Path] = None, **overrides: Any) -> DKOptimizerConfig:
        """Constructs DKOptimizerConfig merging config/settings.yaml, environment variables, and kwargs."""
        cfg_dict = load_yaml_settings(settings_path)
        dk_cfg = cfg_dict.get("draftkings", {})
        sim_cfg = dk_cfg.get("simulation", {})
        exp_cfg = dk_cfg.get("exposure_caps", {})
        solver_cfg = dk_cfg.get("solver", {})

        params: Dict[str, Any] = {}
        if "salary_cap" in dk_cfg:
            params["salary_cap"] = int(dk_cfg["salary_cap"])
        if "num_lineups" in sim_cfg:
            params["num_lineups"] = int(sim_cfg["num_lineups"])
        for k in ["max_qb_exposure", "max_rb_exposure", "max_wr_exposure", "max_te_exposure", "max_def_exposure", "max_exposure"]:
            if k in exp_cfg:
                params[k] = float(exp_cfg[k])
        for k in ["stack_ratio", "randomness_deviation"]:
            if k in solver_cfg:
                params[k] = float(solver_cfg[k])
        for k in ["max_repeating_players"]:
            if k in solver_cfg:
                params[k] = int(solver_cfg[k])
        if "exclude_out_injured" in solver_cfg:
            params["exclude_out_injured"] = bool(solver_cfg["exclude_out_injured"])

        env_map = {
            "DFS_DK_SALARY_CAP": ("salary_cap", int),
            "DFS_DK_NUM_LINEUPS": ("num_lineups", int),
            "DFS_DK_STACK_RATIO": ("stack_ratio", float),
            "DFS_DK_MAX_REPEATING": ("max_repeating_players", int),
            "DFS_DK_RANDOMNESS": ("randomness_deviation", float),
        }
        import os
        for env_var, (attr, cast) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                params[attr] = cast(val)

        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**params)


# -----------------------------------------------------------------------------
# File Path Auto-Discovery
# -----------------------------------------------------------------------------
def find_dk_players_csv(explicit_path: Optional[Path]) -> Path:
    """Resolves the official DraftKings player pool CSV."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("data/players/DKSalaries.csv"),
        Path("data/DKSalaries.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(Path("data").glob("**/*DKSalaries*.csv"))
    if matches:
        return matches[0]

    matches_gen = list(Path("data").glob("**/*salaries*.csv"))
    if matches_gen:
        return matches_gen[0]

    return Path("data/players/DKSalaries.csv")


def find_dk_template_csv(explicit_path: Optional[Path]) -> Path:
    """Resolves the DraftKings entries upload template CSV."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path("data/templates/DKEntries.csv"),
        Path("data/DKEntries.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(Path("data").glob("**/*DKEntries*.csv"))
    if matches:
        return matches[0]

    return Path("data/templates/DKEntries.csv")


# -----------------------------------------------------------------------------
# Data Ingestion & Pre-Solve Filter
# -----------------------------------------------------------------------------
class DraftKingsDataLoader:
    """Ingests official DraftKings player pool CSV directly into optimizer."""

    NON_STARTING_BACKUP_QBS: Set[str] = {
        "Case Keenum", "Drew Lock", "Josh Johnson", "Carson Wentz",
        "Jameis Winston", "Shane Buechele", "Joe Milton III", "Tommy DeVito",
        "Max Brosmer", "Stetson Bennett IV", "Sean Clifford", "Davis Mills",
        "Tanner McKee", "Jarrett Stidham", "Tyrod Taylor", "Sam Howell",
        "Trey Lance", "Gardner Minshew II", "Sam Ehlinger", "Tyler Huntley",
        "Justin Fields", "Mac Jones", "Nick Mullens", "Andy Dalton", "Easton Stick"
    }

    def __init__(self, config: DKOptimizerConfig) -> None:
        self.config = config

    def initialize_and_load_optimizer(self) -> LineupOptimizer:
        csv_path = self.config.players_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"DraftKings player CSV not found at: {csv_path.resolve()}")

        logger.info("Initializing DraftKings Football Optimizer with PuLP/CBC backend...")
        optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)

        logger.info("Loading player pool directly from CSV: %s", csv_path)
        optimizer.load_players_from_csv(str(csv_path))
        optimizer.player_pool.with_injured = True

        logger.info("Successfully loaded %d raw player entries via optimizer.load_players_from_csv.",
                    len(optimizer.player_pool.all_players))

        if self.config.exclude_out_injured:
            self._prune_inactive_players(optimizer, csv_path)

        # Filter backup QBs
        zeroed_count = 0
        for p in optimizer.player_pool.all_players:
            if "QB" in p.positions and (p.full_name in self.NON_STARTING_BACKUP_QBS or "Keenum" in p.full_name):
                if p.fppg > 0:
                    p.fppg = 0.0
                    zeroed_count += 1

        if zeroed_count > 0:
            logger.info("Pre-solve filter: Zeroed out projected FPPG for %d non-starting backup QBs.", zeroed_count)

        return optimizer

    def _prune_inactive_players(self, optimizer: LineupOptimizer, csv_path: Path) -> None:
        """Prunes confirmed inactive/IR players while preserving Questionable starters."""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning("Could not read %s for injury pruning: %s", csv_path, e)
            return

        inactive_indicators = {"IR", "O", "D", "PUP", "Out", "Injured Reserve", "Doubtful"}
        indicator_col = None
        for col in ["Injury Indicator", "Status", "Injury Status"]:
            if col in df.columns:
                indicator_col = col
                break

        if not indicator_col:
            return

        inactive_df = df[df[indicator_col].astype(str).str.strip().isin(inactive_indicators)]
        id_col = "ID" if "ID" in df.columns else "Id" if "Id" in df.columns else None
        if not id_col:
            return

        inactive_ids = set(inactive_df[id_col].astype(str))
        removed_count = 0
        for player in list(optimizer.player_pool.all_players):
            if str(player.id) in inactive_ids:
                optimizer.player_pool.remove_player(player)
                removed_count += 1

        if removed_count > 0:
            logger.info("Pruned %d confirmed inactive/IR players (retained active & Questionable pool: %d players).",
                        removed_count, len(optimizer.player_pool.all_players))


# -----------------------------------------------------------------------------
# Optimization Engine & Portfolio Allocation
# -----------------------------------------------------------------------------
class DraftKingsLineupPipeline:
    """Solves 150 DraftKings tournament lineups with 80/20 stacking and DST negative correlation."""

    def __init__(self, config: DKOptimizerConfig, optimizer: LineupOptimizer) -> None:
        self.config = config
        self.optimizer = optimizer
        self.players = list(optimizer.player_pool.all_players)

    def _get_player_cap_count(self, player: Player) -> int:
        n = self.config.num_lineups
        positions = set(player.positions)
        if "DST" in positions or "D" in positions:
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
        opt.set_max_repeating_players(self.config.max_repeating_players)
        opt.set_fantasy_points_strategy(
            RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
        )
        opt.restrict_positions_for_opposing_team(["DST"], ["QB", "RB", "WR", "TE"])

    def generate_lineups(self) -> List[Lineup]:
        n = self.config.num_lineups
        ratio = self.config.stack_ratio
        n_stacked = round(n * ratio)   # 120
        n_unconstrained = n - n_stacked # 30

        logger.info("DraftKings Portfolio Allocation Plan: %d Stacked (%.0f%%) + %d Unconstrained (%.0f%%)",
                    n_stacked, ratio * 100, n_unconstrained, (1 - ratio) * 100)

        # Phase 1: Stacked
        logger.info("--- Phase 1: Solving %d primary stacked lineups (QB + same-team WR/TE) ---", n_stacked)
        self.configure_optimizer_instance(self.optimizer)
        self.optimizer.add_stack(PositionsStack(["QB", ("WR", "TE")]))

        for player in self.optimizer.player_pool.all_players:
            cap = self._get_player_cap_count(player)
            player.max_exposure = min(1.0, cap / n_stacked) if n_stacked > 0 else 1.0

        stacked_lineups: List[Lineup] = []
        for i, lineup in enumerate(self.optimizer.optimize(n=n_stacked), start=1):
            stacked_lineups.append(lineup)
            if i % 30 == 0 or i == n_stacked:
                logger.info("... Solved %d / %d stacked lineups ...", i, n_stacked)

        if n_unconstrained == 0:
            return stacked_lineups

        usage_phase1: Counter[Any] = Counter()
        for l in stacked_lineups:
            for p in l.lineup:
                usage_phase1[p] += 1

        # Phase 2: Unconstrained
        logger.info("--- Phase 2: Solving %d unconstrained lineups (rushing QB upside) ---", n_unconstrained)
        opt_unconstrained = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
        opt_unconstrained.player_pool.load_players(self.players)
        opt_unconstrained.player_pool.with_injured = True
        self.configure_optimizer_instance(opt_unconstrained)

        for player in opt_unconstrained.player_pool.all_players:
            used = usage_phase1.get(player, 0)
            cap = self._get_player_cap_count(player)
            remaining = cap - used
            if remaining <= 0:
                opt_unconstrained.player_pool.remove_player(player)
            else:
                player.max_exposure = min(1.0, remaining / n_unconstrained)

        stacked_sets = [set(p.id for p in l.lineup) for l in stacked_lineups]
        max_rep = self.config.max_repeating_players

        unconstrained_lineups: List[Lineup] = []
        for lineup in opt_unconstrained.optimize(n=n_unconstrained * 2):
            l_set = set(p.id for p in lineup.lineup)
            if any(len(l_set & s_set) > max_rep for s_set in stacked_sets):
                continue
            unconstrained_lineups.append(lineup)
            if len(unconstrained_lineups) % 15 == 0 or len(unconstrained_lineups) == n_unconstrained:
                logger.info("... Solved %d / %d unconstrained lineups ...", len(unconstrained_lineups), n_unconstrained)
            if len(unconstrained_lineups) == n_unconstrained:
                break

        combined = stacked_lineups + unconstrained_lineups
        logger.info("Successfully assembled %d total lineups (%d stacked + %d unconstrained).",
                    len(combined), len(stacked_lineups), len(unconstrained_lineups))
        return combined


# -----------------------------------------------------------------------------
# Portfolio Auditor
# -----------------------------------------------------------------------------
class DraftKingsPortfolioAuditor:
    """Verifies $50,000 cap adherence, DST correlation, and exposure limits."""

    @staticmethod
    def audit_and_report(lineups: Sequence[Lineup], config: DKOptimizerConfig) -> None:
        total_lineups = len(lineups)
        if total_lineups == 0:
            raise ValueError("No lineups available for audit.")

        logger.info("=" * 70)
        logger.info("DRAFTKINGS PORTFOLIO AUDIT & RISK ANALYSIS (%d LINEUPS)", total_lineups)
        logger.info("=" * 70)

        player_counts: Counter[str] = Counter()
        player_names: Dict[str, str] = {}
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

            qb_player: Optional[LineupPlayer] = None
            dst_player: Optional[LineupPlayer] = None
            offensive_players: List[LineupPlayer] = []

            for p in lineup.lineup:
                player_counts[p.id] += 1
                player_names[p.id] = f"{p.full_name} ({p.team} - {p.lineup_position})"

                if p.lineup_position == "QB":
                    qb_player = p
                elif p.lineup_position == "DST":
                    dst_player = p
                else:
                    offensive_players.append(p)

            # Stack Check
            if qb_player:
                qb_counts[qb_player.full_name] += 1
                same_team_pass_catchers = [
                    op for op in offensive_players
                    if op.team == qb_player.team and any(pos in ("WR", "TE") for pos in op.positions)
                ]
                if len(same_team_pass_catchers) > 0:
                    stacked_count += 1
                else:
                    unconstrained_count += 1

            # Negative Correlation Check (DST vs Opposing Offense)
            if dst_player and dst_player.game_info:
                dst_team = dst_player.team
                game = dst_player.game_info
                opp_team = game.home_team if game.away_team == dst_team else game.away_team
                opposing = [op for op in offensive_players if op.team == opp_team]
                if opposing:
                    opp_names = [f"{op.full_name} ({op.lineup_position})" for op in opposing]
                    violations.append(
                        f"Lineup #{idx}: DST violation! {dst_player.full_name} rostered with opposing: {', '.join(opp_names)}"
                    )

        # Exposure Cap Validation
        def get_cap(player: Union[Player, LineupPlayer]) -> int:
            n = total_lineups
            positions = set(player.positions)
            if "DST" in positions or "D" in positions:
                return math.floor(n * config.max_def_exposure)
            elif "QB" in positions:
                return math.floor(n * config.max_qb_exposure)
            elif "RB" in positions:
                return math.floor(n * config.max_rb_exposure)
            elif "WR" in positions:
                return math.floor(n * config.max_wr_exposure)
            elif "TE" in positions:
                return math.floor(n * config.max_te_exposure)
            return math.floor(n * config.max_exposure)

        player_obj_map = {p.id: p for l in lineups for p in l.lineup}
        for pid, count in player_counts.items():
            p = player_obj_map[pid]
            cap = get_cap(p)
            if count > cap:
                violations.append(
                    f"Exposure cap exceeded for {p.full_name} ({p.lineup_position}): {count} lineups > cap of {cap}"
                )

        if violations:
            logger.error("AUDIT FAILED WITH %d VIOLATIONS:", len(violations))
            for v in violations[:10]:
                logger.error("  - %s", v)
            raise AssertionError("Portfolio failed strict risk/correlation audit.")

        logger.info("ALL CONSTRAINTS STRICTLY SATISFIED:")
        logger.info("  - 100%% of lineups comply with $50,000 salary cap.")
        logger.info("  - 100%% of player exposures comply with position caps.")
        logger.info("  - Primary Stacks: %d / %d (%.1f%%) lineups feature QB + WR/TE stack.",
                    stacked_count, total_lineups, (stacked_count / total_lineups) * 100)
        logger.info("  - Unconstrained:  %d / %d (%.1f%%) lineups feature unconstrained rushing QBs.",
                    unconstrained_count, total_lineups, (unconstrained_count / total_lineups) * 100)
        logger.info("  - 0%% lineups feature DST against opposing offensive skill players.")
        logger.info("  - QBs diversified across %d starting QBs.", len(qb_counts))
        logger.info("  - Total distinct players utilized across portfolio: %d players.", len(player_counts))

        # Top Exposures
        logger.info("-" * 70)
        logger.info("POST-SOLVE EXPOSURE AUDIT TABLE (TOP 15 PLAYERS):")
        for pid, count in player_counts.most_common(15):
            pct = (count / total_lineups) * 100
            name_str = player_names.get(pid, pid)
            logger.info("  %-35s : %3d / %3d (%5.1f%%)", name_str, count, total_lineups, pct)

        # QB Distribution
        logger.info("-" * 70)
        logger.info("STARTING QB EXPOSURE DISTRIBUTION:")
        for qb_name, count in qb_counts.most_common():
            pct = (count / total_lineups) * 100
            logger.info("  QB %-28s : %3d lineups (%5.1f%%)", qb_name, count, pct)

        avg_sal = sum(salary_list) / len(salary_list)
        avg_fppg = sum(fppg_list) / len(fppg_list)
        logger.info("-" * 70)
        logger.info("PORTFOLIO METRICS SUMMARY:")
        logger.info("  Salary Utilization : Min=$%d | Max=$%d | Mean=$%.1f", min(salary_list), max(salary_list), avg_sal)
        logger.info("  Projected FPPG     : Min=%.2f | Max=%.2f | Mean=%.2f", min(fppg_list), max(fppg_list), avg_fppg)
        logger.info("=" * 70)


# -----------------------------------------------------------------------------
# Template Exporter
# -----------------------------------------------------------------------------
class DraftKingsTemplateExporter:
    """Maps 150 optimized rosters into DraftKings CSV upload format without column mangling."""

    ROSTER_SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST")

    def __init__(self, config: DKOptimizerConfig) -> None:
        self.config = config

    def export_lineups(self, lineups: Sequence[Lineup]) -> Path:
        template_path = self.config.template_csv
        output_path = self.config.output_csv

        if not template_path.exists():
            raise FileNotFoundError(f"DraftKings contest template not found: {template_path.resolve()}")

        logger.info("Reading DraftKings reserved entries template: %s", template_path)

        with open(template_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for row in reader:
                if row and any(field.strip() for field in row):
                    rows.append(row)
                if len(rows) == self.config.num_lineups:
                    break

        if len(rows) != len(lineups):
            raise ValueError(
                f"Row mismatch: Template has {len(rows)} reserved entries, but {len(lineups)} lineups were generated."
            )

        # Slot indices 4 to 12
        for i, lineup in enumerate(lineups):
            for slot_idx, p in enumerate(lineup.lineup, start=4):
                if self.config.id_format == "name_id":
                    val = f"{p.full_name} ({p.id})"
                else:
                    val = str(p.id)
                while len(rows[i]) <= slot_idx:
                    rows[i].append("")
                rows[i][slot_idx] = val

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        logger.info("Export successfully written to: %s", output_path.resolve())
        logger.info("Total entries exported: %d valid rows with preserved template headers.", len(rows))
        return output_path


# -----------------------------------------------------------------------------
# CLI Arguments Parser & Entry Point
# -----------------------------------------------------------------------------
def parse_dk_arguments() -> DKOptimizerConfig:
    default_cfg = DKOptimizerConfig.from_settings()

    parser = argparse.ArgumentParser(
        description="DraftKings NFL Classic Quantitative MME Lineup Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--players-csv", type=Path, default=None)
    parser.add_argument("--template-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-lineups", type=int, default=default_cfg.num_lineups)
    parser.add_argument("--stack-ratio", type=float, default=default_cfg.stack_ratio)
    parser.add_argument("--max-qb-exposure", type=float, default=default_cfg.max_qb_exposure)
    parser.add_argument("--max-rb-exposure", type=float, default=default_cfg.max_rb_exposure)
    parser.add_argument("--max-wr-exposure", type=float, default=default_cfg.max_wr_exposure)
    parser.add_argument("--max-te-exposure", type=float, default=default_cfg.max_te_exposure)
    parser.add_argument("--max-def-exposure", type=float, default=default_cfg.max_def_exposure)
    parser.add_argument("--max-exposure", type=float, default=default_cfg.max_exposure)
    parser.add_argument("--randomness", type=float, default=default_cfg.randomness_deviation)
    parser.add_argument("--max-repeating", type=int, default=default_cfg.max_repeating_players)
    parser.add_argument("--keep-injured", action="store_true", help="Keep injured/questionable/out players in player pool")

    args, _ = parser.parse_known_args()

    players_path = find_dk_players_csv(args.players_csv)
    template_path = find_dk_template_csv(args.template_csv)

    if args.output_csv:
        output_path = args.output_csv
    else:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"Completed-{template_path.name}"

    exclude_injured = False if args.keep_injured else default_cfg.exclude_out_injured

    return DKOptimizerConfig.from_settings(
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
        exclude_out_injured=exclude_injured,
    )


def run_draftkings_pipeline(config: Optional[DKOptimizerConfig] = None) -> None:
    if config is None:
        config = parse_dk_arguments()

    logger.info("======================================================================")
    logger.info(" DRAFTKINGS NFL DFS OPTIMIZATION PIPELINE ")
    logger.info(" Quantitative Integer Linear Programming Engine ($50,000 Cap) ")
    logger.info("======================================================================")

    loader = DraftKingsDataLoader(config)
    optimizer = loader.initialize_and_load_optimizer()

    pipeline = DraftKingsLineupPipeline(config, optimizer)
    lineups = pipeline.generate_lineups()

    DraftKingsPortfolioAuditor.audit_and_report(lineups, config)

    exporter = DraftKingsTemplateExporter(config)
    output_file = exporter.export_lineups(lineups)

    print("\n" + "=" * 70)
    print(" DRAFTKINGS PIPELINE COMPLETE: Ready for upload!")
    print(f" Output Location: {output_file.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    run_draftkings_pipeline()
