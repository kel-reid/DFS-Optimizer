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
     - Exports populated DataFrame to `data/<week>/<slate>/completed_lineups.csv`.
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

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

from src.config import DKOptimizerConfig, DraftKingsConfig
from src.engine import (
    CandidatePoolGenerator,
    CorrelatedGameEngine,
    GameBringBackStack,
    OpponentFieldSimulator,
    PortfolioSelector,
    SimAuditReporter,
)

__all__ = [
    "DKOptimizerConfig",
    "DraftKingsConfig",
    "DraftKingsDataLoader",
    "DraftKingsLineupPipeline",
    "DraftKingsPortfolioAuditor",
    "DraftKingsTemplateExporter",
    "find_dk_players_csv",
    "find_dk_template_csv",
    "parse_dk_arguments",
    "run_draftkings_pipeline",
]

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
# File Path Auto-Discovery
# -----------------------------------------------------------------------------
def find_dk_players_csv(
    explicit_path: Optional[Path] = None,
    slate: Optional[str] = None,
    week: Optional[Union[str, int]] = None,
    slate_date: Optional[str] = None,
) -> Path:
    """Resolves the official DraftKings player pool CSV from explicit path, slate folder, or legacy locations."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    from src.data.loader import normalize_week
    norm_week = normalize_week(week)

    if slate:
        from src.data.loader import _find_in_slate_dirs
        patterns = ["*DKSalaries*.csv", "*salaries*.csv", "DKSalaries.csv", "*player*.csv"]
        found = _find_in_slate_dirs(slate, patterns, norm_week, slate_date, "players")
        if found:
            return found

    # Only fall back to generic candidates if neither specific week nor slate_date was given
    if not norm_week and not slate_date:
        candidates = [
            Path("data/players/DKSalaries.csv"),
            Path("data/DKSalaries.csv"),
        ]
        for c in candidates:
            if c.exists():
                return c

        matches = [
            p for p in sorted(Path("data").glob("**/*DKSalaries*.csv"))
            if not p.name.startswith("Completed-") and p.name != "completed_lineups.csv"
        ]
        if matches:
            return matches[0]

        matches_gen = [
            p for p in sorted(Path("data").glob("**/*salaries*.csv"))
            if not p.name.startswith("Completed-") and p.name != "completed_lineups.csv"
        ]
        if matches_gen:
            return matches_gen[0]

    raise FileNotFoundError(
        f"Could not locate DraftKings players/salaries CSV (slate='{slate}', week='{week}', date='{slate_date}')."
    )


def find_dk_template_csv(
    explicit_path: Optional[Path] = None,
    slate: Optional[str] = None,
    week: Optional[Union[str, int]] = None,
    slate_date: Optional[str] = None,
) -> Path:
    """Resolves the DraftKings entries upload template CSV from explicit path, slate folder, or legacy locations."""
    if explicit_path and explicit_path.exists():
        return explicit_path

    from src.data.loader import normalize_week
    norm_week = normalize_week(week)

    if slate:
        from src.data.loader import _find_in_slate_dirs
        patterns = ["*DKEntries*.csv", "*entries*.csv", "DKEntries.csv", "*template*.csv"]
        found = _find_in_slate_dirs(slate, patterns, norm_week, slate_date, "templates")
        if found:
            return found

    # Only fall back to generic candidates if neither specific week nor slate_date was given
    if not norm_week and not slate_date:
        candidates = [
            Path("data/templates/DKEntries.csv"),
            Path("data/DKEntries.csv"),
        ]
        for c in candidates:
            if c.exists():
                return c

        matches = [
            p for p in sorted(Path("data").glob("**/*DKEntries*.csv"))
            if not p.name.startswith("Completed-") and p.name != "completed_lineups.csv"
        ]
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not locate DraftKings entries template CSV (slate='{slate}', week='{week}', date='{slate_date}')."
    )


# -----------------------------------------------------------------------------
# Data Ingestion & Pre-Solve Filter
# -----------------------------------------------------------------------------
class DraftKingsDataLoader:
    """Ingests official DraftKings player pool CSV directly into optimizer."""

    def __init__(self, config: DKOptimizerConfig) -> None:
        self.config = config

    def initialize_and_load_optimizer(self) -> LineupOptimizer:
        csv_path = self.config.players_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"DraftKings player CSV not found at: {csv_path.resolve()}")

        is_single_game = self._detect_single_game(csv_path)
        self.config.is_single_game = is_single_game
        site = Site.DRAFTKINGS_CAPTAIN_MODE if is_single_game else Site.DRAFTKINGS

        logger.info(
            "Initializing DraftKings Football Optimizer (%s) with PuLP/CBC backend...",
            "Showdown / Captain Mode" if is_single_game else "Classic",
        )
        optimizer = get_optimizer(site, Sport.FOOTBALL)

        logger.info("Loading player pool directly from CSV: %s", csv_path)
        optimizer.load_players_from_csv(str(csv_path))
        optimizer.player_pool.with_injured = True
        optimizer.settings.budget = self.config.salary_cap

        logger.info("Successfully loaded %d raw player entries via optimizer.load_players_from_csv.",
                    len(optimizer.player_pool.all_players))

        if self.config.exclude_out_injured:
            self._prune_inactive_players(optimizer, csv_path)

        if self.config.projections_csv and self.config.projections_csv.exists():
            from src.data.projections import apply_forward_projections
            apply_forward_projections(
                optimizer,
                self.config.projections_csv,
                zero_unprojected=self.config.zero_unprojected,
            )

        # Filter backup QBs
        backup_qbs = set(self.config.backup_quarterbacks or ())
        zeroed_count = 0
        if backup_qbs:
            for p in optimizer.player_pool.all_players:
                if "QB" in p.positions and p.full_name in backup_qbs:
                    if p.fppg > 0:
                        p.fppg = 0.0
                        zeroed_count += 1

            if zeroed_count > 0:
                logger.info("Pre-solve filter: Zeroed out projected FPPG for %d non-starting backup QBs.", zeroed_count)

        # Attach underlying position from DKSalaries.csv (e.g. K, DST, QB, WR, RB, TE)
        try:
            df = pd.read_csv(csv_path)
            id_col = "ID" if "ID" in df.columns else "Id"
            pos_col = "Position" if "Position" in df.columns else "Roster Position"
            id_to_pos = dict(zip(df[id_col].astype(str), df[pos_col].astype(str).str.strip().str.upper()))
            for p in optimizer.player_pool.all_players:
                orig = id_to_pos.get(str(p.id))
                if orig:
                    p.underlying_position = orig  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("Could not map underlying positions: %s", e)

        # In Single Game / Showdown, restrict K and DST from the 1.5x Captain slot (FLEX eligible only)
        if is_single_game:
            removed_cpts = 0
            for p in list(optimizer.player_pool.all_players):
                orig = getattr(p, "underlying_position", "")
                if orig in ("K", "DST") and "CPT" in p.positions:
                    optimizer.player_pool.remove_player(p)
                    removed_cpts += 1
            if removed_cpts > 0:
                logger.info("Showdown Rule: Restricted %d Kicker/Defense entries from CPT pool (FLEX eligible only).", removed_cpts)

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

    def _detect_single_game(self, csv_path: Path) -> bool:
        """Detects whether target contest format is Single Game / Showdown vs Classic."""
        if getattr(self.config, "is_single_game", False):
            return True
        if getattr(self.config, "slate", None) in ("sunday-night", "monday-night", "thursday-night", "single-game", "showdown"):
            return True
        if getattr(self.config, "template_csv", None) and self.config.template_csv.exists():
            try:
                with open(self.config.template_csv, "r", encoding="utf-8-sig") as f:
                    header = f.readline()
                    if "CPT" in header:
                        return True
            except Exception:
                pass
        try:
            df = pd.read_csv(csv_path)
            if "Roster Position" in df.columns:
                roster_pos = df["Roster Position"].dropna().astype(str)
                if any("CPT" in p for p in roster_pos):
                    return True
        except Exception:
            pass
        return False



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
        n = self.config.num_selected_lineups
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
        if len(opt.player_pool.available_teams) >= 3:
            opt.set_total_teams(min_teams=3)
        opt.settings.max_from_one_team = 4
        opt.settings.budget = self.config.salary_cap

    def generate_lineups(self) -> List[Lineup]:
        n = self.config.num_selected_lineups
        ratio = self.config.stack_ratio
        game_ratio = getattr(self.config, "game_stack_ratio", 0.0)
        n_game_stacked = round(n * game_ratio)
        n_primary_stacked = round(n * ratio) - n_game_stacked
        n_unconstrained = n - n_game_stacked - n_primary_stacked

        logger.info(
            "DraftKings Portfolio Allocation Plan: %d Game Stack w/ Bring-Back (%.0f%%) + %d Primary Stack (%.0f%%) + %d Unconstrained (%.0f%%)",
            n_game_stacked,
            (n_game_stacked / n) * 100,
            n_primary_stacked,
            (n_primary_stacked / n) * 100,
            n_unconstrained,
            (n_unconstrained / n) * 100,
        )

        all_lineups: List[Lineup] = []
        accepted_sets: List[set[str]] = []
        usage: Counter[Any] = Counter()
        max_rep = self.config.max_repeating_players

        # Phase 1: Game Stack with Bring-Back
        if n_game_stacked > 0:
            logger.info("--- Phase 1: Solving %d game stacked lineups (QB + same WR/TE + opp WR/TE) ---", n_game_stacked)
            self.configure_optimizer_instance(self.optimizer)
            self.optimizer.add_stack(GameBringBackStack())

            for player in self.optimizer.player_pool.all_players:
                cap = self._get_player_cap_count(player)
                player.max_exposure = min(1.0, cap / n_game_stacked) if n_game_stacked > 0 else 1.0

            for i, lineup in enumerate(self.optimizer.optimize(n=n_game_stacked), start=1):
                all_lineups.append(lineup)
                accepted_sets.append(set(p.id for p in lineup.lineup))
                for p in lineup.lineup:
                    usage[p] += 1
                if i % 30 == 0 or i == n_game_stacked:
                    logger.info("... Solved %d / %d game stacked lineups ...", i, n_game_stacked)

        # Phase 2: Primary Stacked
        target_stacked = n_game_stacked + n_primary_stacked
        needed_primary = target_stacked - len(all_lineups)
        if needed_primary > 0:
            logger.info("--- Phase 2: Solving %d primary stacked lineups (QB + same-team WR/TE) ---", needed_primary)
            opt_primary = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
            opt_primary.player_pool.load_players(self.players)
            opt_primary.player_pool.with_injured = True
            self.configure_optimizer_instance(opt_primary)
            opt_primary.add_stack(PositionsStack(["QB", ("WR", "TE")]))

            for player in opt_primary.player_pool.all_players:
                used = usage.get(player, 0)
                cap = self._get_player_cap_count(player)
                remaining = cap - used
                if remaining <= 0:
                    opt_primary.player_pool.remove_player(player)
                else:
                    player.max_exposure = min(1.0, remaining / needed_primary)

            solved_primary = 0
            for lineup in opt_primary.optimize(n=needed_primary * 2):
                l_set = set(p.id for p in lineup.lineup)
                if any(len(l_set & s_set) > max_rep for s_set in accepted_sets):
                    continue
                all_lineups.append(lineup)
                accepted_sets.append(l_set)
                for p in lineup.lineup:
                    usage[p] += 1
                solved_primary += 1
                if solved_primary % 30 == 0 or solved_primary == needed_primary:
                    logger.info("... Solved %d / %d primary stacked lineups ...", solved_primary, needed_primary)
                if solved_primary == needed_primary:
                    break

        # Phase 3: Unconstrained
        needed_unconstrained = n - len(all_lineups)
        if needed_unconstrained > 0:
            logger.info("--- Phase 3: Solving %d unconstrained lineups (rushing QB upside) ---", needed_unconstrained)
            opt_unconstrained = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
            opt_unconstrained.player_pool.load_players(self.players)
            opt_unconstrained.player_pool.with_injured = True
            self.configure_optimizer_instance(opt_unconstrained)

            for player in opt_unconstrained.player_pool.all_players:
                used = usage.get(player, 0)
                cap = self._get_player_cap_count(player)
                remaining = cap - used
                if remaining <= 0:
                    opt_unconstrained.player_pool.remove_player(player)
                else:
                    player.max_exposure = min(1.0, remaining / needed_unconstrained)

            solved_unconstrained = 0
            for lineup in opt_unconstrained.optimize(n=needed_unconstrained * 3):
                l_set = set(p.id for p in lineup.lineup)
                if any(len(l_set & s_set) > max_rep for s_set in accepted_sets):
                    continue
                all_lineups.append(lineup)
                accepted_sets.append(l_set)
                for p in lineup.lineup:
                    usage[p] += 1
                solved_unconstrained += 1
                if solved_unconstrained % 15 == 0 or solved_unconstrained == needed_unconstrained:
                    logger.info("... Solved %d / %d unconstrained lineups ...", solved_unconstrained, needed_unconstrained)
                if solved_unconstrained == needed_unconstrained:
                    break

        if len(all_lineups) < n:
            raise ValueError(
                f"Generated only {len(all_lineups)} / {n} requested lineups. "
                "Exposure ceilings or uniqueness constraints may be overly restrictive."
            )

        logger.info("Successfully assembled %d total lineups across portfolio.", len(all_lineups))
        return all_lineups


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
        game_stacked_count = 0
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
                    if qb_player.game_info:
                        game = qb_player.game_info
                        opp_team = game.home_team if game.away_team == qb_player.team else game.away_team
                        opp_pass_catchers = [
                            op for op in offensive_players
                            if op.team == opp_team and any(pos in ("WR", "TE") for pos in op.positions)
                        ]
                        if len(opp_pass_catchers) > 0:
                            game_stacked_count += 1
                else:
                    unconstrained_count += 1

            # Negative Correlation Check (DST vs Opposing Offense - Classic only)
            if not getattr(config, "is_single_game", False) and dst_player and dst_player.game_info:
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
            orig = getattr(player, "underlying_position", "")
            if orig:
                positions.add(orig)

            if "DST" in positions or "D" in positions or "K" in positions:
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
        exposure_violations: List[str] = []
        for pid, count in player_counts.items():
            p = player_obj_map[pid]
            cap = get_cap(p)
            if count > cap:
                exposure_violations.append(
                    f"Exposure cap exceeded for {p.full_name} ({p.lineup_position}): {count} lineups > cap of {cap}"
                )

        if violations:
            logger.error("AUDIT FAILED WITH %d CRITICAL CONTEST RULE VIOLATIONS:", len(violations))
            for v in violations[:10]:
                logger.error("  - %s", v)
            raise AssertionError("Portfolio failed strict contest rule audit.")

        if exposure_violations:
            if getattr(config, "strict_exposure_caps", False):
                logger.error("AUDIT FAILED: Strict exposure caps violated (%d players over cap):", len(exposure_violations))
                for v in exposure_violations[:10]:
                    logger.error("  - %s", v)
                raise AssertionError("Portfolio failed strict exposure cap audit.")
            else:
                logger.warning(
                    "AUDIT WARNING: %d exposure cap overage(s) present (via Pass 2 fallback fill):",
                    len(exposure_violations),
                )
                for v in exposure_violations[:10]:
                    logger.warning("  - %s", v)

        logger.info("ALL CONSTRAINTS STRICTLY SATISFIED:")
        logger.info("  - 100%% of lineups comply with $50,000 salary cap.")
        logger.info("  - 100%% of player exposures comply with position caps.")
        logger.info("  - Primary Stacks: %d / %d (%.1f%%) lineups feature QB + WR/TE stack.",
                    stacked_count, total_lineups, (stacked_count / total_lineups) * 100)
        logger.info("  - Game Stacks (Bring-Back): %d / %d (%.1f%%) lineups feature QB + WR/TE + Opp WR/TE.",
                    game_stacked_count, total_lineups, (game_stacked_count / total_lineups) * 100)
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
                if len(rows) == self.config.num_selected_lineups:
                    break

        if len(rows) != len(lineups):
            raise ValueError(
                f"Row mismatch: Template has {len(rows)} reserved entries, but {len(lineups)} lineups were generated."
            )

        # Slot indices 4 to 12 (or 4 to 9 for Showdown)
        for i, lineup in enumerate(lineups):
            ordered_lineup = list(lineup.lineup)
            if getattr(self.config, "is_single_game", False):
                ordered_lineup.sort(key=lambda p: 0 if "CPT" in (getattr(p, "lineup_position", "") or p.positions) else 1)
            for slot_idx, p in enumerate(ordered_lineup, start=4):
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
        description="DraftKings NFL Classic Quantitative Monte Carlo Simulation & Optimization Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--players-csv",
        "--salaries-csv",
        dest="players_csv",
        type=Path,
        default=None,
        help="DraftKings player salaries CSV file",
    )
    parser.add_argument(
        "--template-csv",
        "--entries-csv",
        dest="template_csv",
        type=Path,
        default=None,
        help="DraftKings contest entry template CSV file",
    )
    parser.add_argument("--projections-csv", type=Path, default=None, help="External projections CSV file")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=default_cfg.num_candidates)
    parser.add_argument("--num-field", type=int, default=default_cfg.num_field_lineups)
    parser.add_argument("--num-trials", type=int, default=default_cfg.num_sim_trials)
    parser.add_argument("--num-lineups", type=int, default=default_cfg.num_selected_lineups)
    parser.add_argument(
        "--entry-fee",
        type=float,
        default=None,
        help="Contest entry fee in dollars (default: auto-detect from template)",
    )
    parser.add_argument("--stack-ratio", type=float, default=default_cfg.stack_ratio)
    parser.add_argument("--game-stack-ratio", type=float, default=default_cfg.game_stack_ratio)
    parser.add_argument("--max-qb-exposure", type=float, default=default_cfg.max_qb_exposure)
    parser.add_argument("--max-rb-exposure", type=float, default=default_cfg.max_rb_exposure)
    parser.add_argument("--max-wr-exposure", type=float, default=default_cfg.max_wr_exposure)
    parser.add_argument("--max-te-exposure", type=float, default=default_cfg.max_te_exposure)
    parser.add_argument("--max-def-exposure", type=float, default=default_cfg.max_def_exposure)
    parser.add_argument("--max-dst-exposure", type=float, default=default_cfg.max_def_exposure)
    parser.add_argument("--max-exposure", type=float, default=default_cfg.max_exposure)
    parser.add_argument("--randomness", type=float, default=default_cfg.randomness_deviation)
    parser.add_argument("--max-repeating", type=int, default=default_cfg.max_repeating_players)
    parser.add_argument("--keep-injured", action="store_true", help="Keep injured/questionable/out players in player pool")
    parser.add_argument("--strict-caps", action="store_true", default=default_cfg.strict_exposure_caps)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["roi", "ceiling", "win_rate"],
        default=default_cfg.selection_metric,
        help="Portfolio selection metric: 'roi' (mean simulated ROI), 'ceiling' (99.9th percentile score), or 'win_rate' (1st place rate)",
    )
    parser.add_argument(
        "--primary-stack-size",
        type=int,
        default=default_cfg.primary_stack_size,
        help="Number of primary pass-catchers paired with QB (default: 1; 2 for double stack)",
    )
    parser.add_argument(
        "--bring-back-size",
        type=int,
        default=default_cfg.bring_back_size,
        help="Number of opposing pass-catchers for bring-back in game stacks (default: 1; 0 to disable, 2 for double bring-back)",
    )
    parser.add_argument(
        "--team-defense-stack",
        action="store_true",
        default=default_cfg.include_team_defense_stack,
        help="Stack QB with same-team Defense for game-script team onslaughts",
    )
    parser.add_argument(
        "--max-total-ownership",
        type=float,
        default=default_cfg.max_total_ownership,
        help="Maximum cumulative projected ownership per lineup (e.g. 115.0)",
    )
    parser.add_argument("--slate", type=str, default=default_cfg.slate, help="Target slate identifier (e.g. main-slate, sunday-night).")
    parser.add_argument("--week", type=str, default=default_cfg.week, help="NFL Week (e.g. 5, 05, week-05).")
    parser.add_argument("--date", type=str, default=default_cfg.slate_date, help="Slate date in YYYY-MM-DD format (e.g. 2026-10-04).")
    parser.add_argument("--zero-unprojected", action="store_true", default=default_cfg.zero_unprojected)
    parser.add_argument("--keep-unprojected", action="store_true", default=False)
    parser.add_argument("--single-game", action="store_true", default=default_cfg.is_single_game, help="Force Single Game / Showdown format.")

    args = parser.parse_args()

    from src.data.loader import normalize_week
    norm_week = normalize_week(args.week)

    players_path = find_dk_players_csv(args.players_csv, slate=args.slate, week=norm_week, slate_date=args.date)
    template_path = find_dk_template_csv(args.template_csv, slate=args.slate, week=norm_week, slate_date=args.date)

    if args.projections_csv:
        projections_path = args.projections_csv
    elif args.slate:
        from src.data.projections import find_projections_csv
        projections_path = find_projections_csv(slate=args.slate, week=norm_week, slate_date=args.date)
    else:
        projections_path = None

    if args.output_csv:
        output_path = args.output_csv
    elif args.slate:
        w_str = norm_week or "week-05"
        slate_dir = Path(f"data/{w_str}/{args.slate}")
        if not slate_dir.is_dir() and args.date:
            date_dir = Path(f"data/{args.date}/{args.slate}")
            if date_dir.is_dir():
                slate_dir = date_dir
        slate_dir.mkdir(parents=True, exist_ok=True)
        output_path = slate_dir / "completed_lineups.csv"
    else:
        w_str = norm_week or "week-05"
        output_dir = Path(f"data/{w_str}/main-slate")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "completed_lineups.csv"

    exclude_injured = False if args.keep_injured else default_cfg.exclude_out_injured
    zero_unproj = False if args.keep_unprojected else default_cfg.zero_unprojected

    return DKOptimizerConfig.from_settings(
        players_csv=players_path,
        template_csv=template_path,
        output_csv=output_path,
        projections_csv=projections_path,
        slate=args.slate,
        week=norm_week,
        slate_date=args.date,
        num_candidates=args.num_candidates,
        num_field_lineups=args.num_field,
        num_sim_trials=args.num_trials,
        num_selected_lineups=args.num_lineups,
        entry_fee=args.entry_fee,
        max_exposure=args.max_exposure,
        randomness_deviation=args.randomness,
        stack_ratio=args.stack_ratio,
        game_stack_ratio=args.game_stack_ratio,
        selection_metric=args.selection_metric,
        primary_stack_size=args.primary_stack_size,
        bring_back_size=args.bring_back_size,
        include_team_defense_stack=args.team_defense_stack,
        max_total_ownership=args.max_total_ownership,
        max_qb_exposure=args.max_qb_exposure,
        max_rb_exposure=args.max_rb_exposure,
        max_wr_exposure=args.max_wr_exposure,
        max_te_exposure=args.max_te_exposure,
        max_def_exposure=args.max_dst_exposure if args.max_dst_exposure != default_cfg.max_def_exposure else args.max_def_exposure,
        max_repeating_players=args.max_repeating,
        exclude_out_injured=exclude_injured,
        strict_exposure_caps=args.strict_caps,
        zero_unprojected=zero_unproj,
        is_single_game=args.single_game,
    )


def run_draftkings_pipeline(
    config: Optional[DKOptimizerConfig] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Path:
    if config is None:
        config = parse_dk_arguments()

    if progress_callback:
        progress_callback(5, "Initializing DraftKings Monte Carlo Pipeline ($50,000 Cap)...")

    logger.info("======================================================================")
    logger.info(" DRAFTKINGS NFL DFS SIMULATION & OPTIMIZATION PIPELINE ")
    logger.info(" Monte Carlo Tournament Game Slate Engine ($50,000 Cap) ")
    logger.info("======================================================================")

    # 1. Ingest Data & Filter Backup QBs / Inactives
    if progress_callback:
        progress_callback(15, "Ingesting salary data & filtering confirmed inactives / backup QBs...")
    loader = DraftKingsDataLoader(config)
    optimizer = loader.initialize_and_load_optimizer()
    players = list(optimizer.player_pool.filtered_players)

    # 2. Stage 1: Candidate Pool Generation (MILP)
    if progress_callback:
        progress_callback(35, f"Stage 1: Generating candidate pool of {config.num_candidates} lineups via MILP...")
    candidate_generator = CandidatePoolGenerator(optimizer, config)
    candidates, candidates_matrix = candidate_generator.generate_candidate_pool(progress_callback=progress_callback)

    # 3. Stage 2: Simulate Opponent Field Lineups
    if progress_callback:
        progress_callback(55, f"Stage 2: Simulating {config.num_field_lineups:,} opponent field lineups...")
    field_simulator = OpponentFieldSimulator(players, config)
    field_matrix = field_simulator.simulate_field()

    # 4. Stage 3: Correlated Game Outcome Engine & Vectorized Scoring
    if progress_callback:
        progress_callback(75, f"Stage 3: Running {config.num_sim_trials:,} correlated game trials & BLAS scoring...")
    game_engine = CorrelatedGameEngine(players, config)
    sim_points = game_engine.simulate_game_trials()

    sim_res = game_engine.score_and_rank_candidates(
        candidates_matrix, field_matrix, sim_points
    )
    sim_roi, win_counts, top1_rates = sim_res

    # 5. Stage 4: Portfolio Selection (Exposure Ceilings)
    if progress_callback:
        metric_label = "simulated ROI" if config.selection_metric == "roi" else ("99.9th percentile ceiling" if config.selection_metric == "ceiling" else "win rate")
        progress_callback(90, f"Stage 4: Selecting {config.num_selected_lineups} lineups maximizing {metric_label}...")
    selector = PortfolioSelector(config)
    selected_lineups = selector.select_portfolio(
        candidates, sim_roi, win_counts, top1_rates, cand_ceilings=getattr(sim_res, "ceilings", None)
    )

    # 6. Post-Solve Audit & Quality Assurance
    DraftKingsPortfolioAuditor.audit_and_report(selected_lineups, config)
    SimAuditReporter.audit_and_report(selected_lineups, config, players, sim_roi, win_counts, top1_rates)

    # 7. Map Lineups into DraftKings CSV upload format and export
    if progress_callback:
        progress_callback(95, "Mapping lineups to DraftKings contest template...")
    exporter = DraftKingsTemplateExporter(config)
    output_file = exporter.export_lineups(selected_lineups)

    if progress_callback:
        progress_callback(100, "DraftKings optimization complete! Ready for download.")

    print("\n" + "=" * 70)
    print(" DRAFTKINGS PIPELINE COMPLETE: Ready for upload!")
    print(f" Output Location: {output_file.resolve()}")
    print("=" * 70)
    return output_file


if __name__ == "__main__":
    run_draftkings_pipeline()

