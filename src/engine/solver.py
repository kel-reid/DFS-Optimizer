"""
================================================================================
Candidate Pool Generator Module (src/engine/solver.py)
================================================================================
Generates N = 500 valid, diverse, correlated tournament candidate rosters
using CBC integer linear programming solver and encodes into a binary NumPy matrix.
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
from pydfs_lineup_optimizer.stacks import (
    BaseGroup,
    BaseStack,
    NestedPlayersGroup,
    OptimizerStack,
    PlayersGroup,
)

from src.config import BaseOptimizerConfig

logger = logging.getLogger("FanDuelSimOptimizer")


class GameBringBackStack(BaseStack):
    """
    Game Stack with Bring-Back Constraint:
    Requires:
      - Starting/Projected QB from Team A (min 1)
      - >= 1 WR/TE pass-catcher from Team A (Primary Stack)
      - >= 1 WR/TE pass-catcher from Team B (Opposing Bring-Back)
    Iterates across all game matchups present in the player pool.
    """

    def __init__(
        self,
        max_exposure: Optional[float] = None,
        max_exposure_per_team: Optional[Dict[str, float]] = None,
        primary_stack_size: int = 1,
        bring_back_size: int = 1,
        include_team_defense: bool = False,
    ):
        self.max_exposure = max_exposure
        self.max_exposure_per_team = max_exposure_per_team or {}
        self.primary_stack_size = max(1, primary_stack_size)
        self.bring_back_size = max(0, bring_back_size)
        self.include_team_defense = include_team_defense

    def build_stacks(self, players: List[Player], optimizer: LineupOptimizer) -> List[OptimizerStack]:
        players_by_team: Dict[str, List[Player]] = {}
        for p in players:
            if p.team:
                players_by_team.setdefault(p.team, []).append(p)

        all_groups: List[BaseGroup] = []
        for team, t_players in players_by_team.items():
            qbs = [p for p in t_players if "QB" in p.positions and p.fppg > 0.0]
            if not qbs:
                continue

            # Identify opponent team from game_info
            opp_teams = set()
            for p in t_players:
                if p.game_info and p.game_info.home_team and p.game_info.away_team:
                    opp = p.game_info.away_team if p.team == p.game_info.home_team else p.game_info.home_team
                    opp_teams.add(opp)
            if not opp_teams:
                continue

            opp_team = next(iter(opp_teams))
            opp_players = players_by_team.get(opp_team, [])

            same_pass = [p for p in t_players if any(pos in ("WR", "TE") for pos in p.positions) and p.fppg > 0.0]
            opp_pass = [p for p in opp_players if any(pos in ("WR", "TE") for pos in p.positions) and p.fppg > 0.0]

            min_primary = min(len(same_pass), self.primary_stack_size)
            if min_primary < 1:
                continue

            groups: List[PlayersGroup] = [
                PlayersGroup(players=qbs, min_from_group=1),
                PlayersGroup(players=same_pass, min_from_group=min_primary),
            ]

            if self.bring_back_size > 0:
                min_bb = min(len(opp_pass), self.bring_back_size)
                if min_bb < 1:
                    continue
                groups.append(PlayersGroup(players=opp_pass, min_from_group=min_bb))

            if self.include_team_defense:
                same_def = [p for p in t_players if any(pos in ("D", "DEF", "DST") for pos in p.positions) and p.fppg > 0.0]
                if same_def:
                    groups.append(PlayersGroup(players=same_def, min_from_group=1))

            all_groups.append(
                NestedPlayersGroup(
                    groups=groups,
                    max_exposure=self.max_exposure_per_team.get(team, self.max_exposure),
                )
            )

        if not all_groups:
            pass_stack: List[Any] = ["QB"] + [("WR", "TE")] * self.primary_stack_size
            return PositionsStack(pass_stack).build_stacks(players, optimizer)

        return [OptimizerStack(groups=all_groups)]

    def validate(self, optimizer: LineupOptimizer) -> None:
        pass


class CandidatePoolGenerator:
    """
    Generates N = 500 valid, diverse, correlated tournament candidate rosters
    using the PuLP/CBC integer linear programming solver, and encodes them into
    a binary NumPy matrix of shape (500, P).
    """

    def __init__(self, optimizer: LineupOptimizer, config: BaseOptimizerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.players: List[Player] = list(optimizer.player_pool.filtered_players)
        self.player_to_idx: Dict[str, int] = {str(p.id): idx for idx, p in enumerate(self.players)}

    def generate_candidate_pool(self) -> Tuple[List[Lineup], np.ndarray]:
        """
        Solves 500 candidates:
          - 200 Game Stack with Bring-Back (QB + same WR/TE + opp WR/TE) [40%]
          - 200 Primary Stacked (QB + >= 1 same-team WR/TE) [40%]
          - 100 Unconstrained (capturing standalone rushing QB ceiling) [20%]
        Returns:
          lineups: List of 500 Lineup objects
          candidates_matrix: Binary ndarray of shape (500, P)
        """
        n_total = self.config.num_candidates
        ratio = self.config.stack_ratio
        game_ratio = getattr(self.config, "game_stack_ratio", 0.0)

        total_slots = self.optimizer.settings.get_total_players()
        is_single_game = total_slots < 9

        if not is_single_game:
            n_game_stack = round(n_total * game_ratio)
            n_primary_stack = round(n_total * ratio) - n_game_stack
            n_unconstrained = n_total - n_game_stack - n_primary_stack
        else:
            n_game_stack = 0
            n_primary_stack = round(n_total * ratio)
            n_unconstrained = n_total - n_primary_stack

        print()
        logger.info("=" * 70)
        logger.info("STAGE 1: GENERATING CANDIDATE POOL (N = %d LINEUPS)", n_total)
        logger.info("=" * 70)
        logger.info(
            "Candidate Pool Allocation: %d Game Stack w/ Bring-Back (%.0f%%) + %d Primary Stack (%.0f%%) + %d Unconstrained (%.0f%%)",
            n_game_stack,
            (n_game_stack / n_total) * 100,
            n_primary_stack,
            (n_primary_stack / n_total) * 100,
            n_unconstrained,
            (n_unconstrained / n_total) * 100,
        )

        max_rep = min(self.config.max_repeating_players, total_slots - 1)
        site = self.optimizer.settings.site
        def_pos = ["DST"] if site == Site.DRAFTKINGS else ["D"]

        # Baseline constraints for candidates
        self.optimizer.set_max_repeating_players(max_rep)
        self.optimizer.set_fantasy_points_strategy(
            RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
        )
        if not is_single_game:
            opp_restricted = ["QB", "RB"] if getattr(self.config, "include_team_defense_stack", False) else ["QB", "RB", "WR", "TE"]
            self.optimizer.restrict_positions_for_opposing_team(def_pos, opp_restricted)
        if len(self.optimizer.player_pool.available_teams) >= 3:
            self.optimizer.set_total_teams(min_teams=3)
        elif len(self.optimizer.player_pool.available_teams) == 2:
            self.optimizer.set_total_teams(min_teams=2)

        self.optimizer.settings.max_from_one_team = 4
        self.optimizer.settings.budget = self.config.salary_cap
        # Ensure diversity across candidates by setting max exposure during candidate phase
        num_teams = len(self.optimizer.player_pool.available_teams)
        if is_single_game:
            cand_max_exp = 0.65
        elif num_teams <= 4:
            cand_max_exp = 0.60
        else:
            cand_max_exp = 0.35

        max_own = getattr(self.config, "max_total_ownership", None)
        if max_own is not None and max_own > 0:
            has_own = any(p.projected_ownership is not None for p in self.optimizer.player_pool.filtered_players)
            if has_own:
                avg_own = max_own / total_slots
                if avg_own > 1.0:
                    avg_own /= 100.0
                self.optimizer.set_projected_ownership(max_projected_ownership=avg_own)
                logger.info("Enforcing cumulative projected ownership cap: <= %.1f%% (avg <= %.2f%%/slot)", max_own, avg_own * 100)

        primary_size = getattr(self.config, "primary_stack_size", 1)
        bb_size = getattr(self.config, "bring_back_size", 1)
        def_stack = getattr(self.config, "include_team_defense_stack", False)

        candidates: List[Lineup] = []
        t0 = time.time()

        # Phase 1: Game Stack with Bring-Back
        if n_game_stack > 0:
            logger.info(
                "Solving %d game stacked candidate lineups (QB + %d same-team WR/TE + %d opp WR/TE%s)...",
                n_game_stack,
                primary_size,
                bb_size,
                " + DEF" if def_stack else "",
            )
            self.optimizer.add_stack(
                GameBringBackStack(
                    primary_stack_size=primary_size,
                    bring_back_size=bb_size,
                    include_team_defense=def_stack,
                )
            )
            for i, lineup in enumerate(self.optimizer.optimize(n=n_game_stack), start=1):
                candidates.append(lineup)
                if i % 100 == 0 or i == n_game_stack:
                    logger.info("... Solved %d / %d game stacked candidates (%.1fs) ...", i, n_game_stack, time.time() - t0)

        # Phase 2: Primary Stacked Lineups
        target_stacked = n_game_stack + n_primary_stack
        needed_primary = target_stacked - len(candidates)
        if needed_primary > 0:
            logger.info(
                "Solving %d primary stacked candidate lineups (QB + same-team WR/TE)..." if not is_single_game else "Solving %d candidate lineups for Single Game slate...",
                needed_primary,
            )
            opt_primary = get_optimizer(site, Sport.FOOTBALL)
            opt_primary.player_pool.load_players(self.players)
            opt_primary.player_pool.with_injured = True
            opt_primary.set_max_repeating_players(max_rep)
            opt_primary.set_fantasy_points_strategy(
                RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
            )
            if max_own is not None and max_own > 0:
                has_own = any(p.projected_ownership is not None for p in opt_primary.player_pool.filtered_players)
                if has_own:
                    avg_own = max_own / total_slots
                    if avg_own > 1.0:
                        avg_own /= 100.0
                    opt_primary.set_projected_ownership(max_projected_ownership=avg_own)
            if not is_single_game:
                opt_primary.restrict_positions_for_opposing_team(def_pos, ["QB", "RB", "WR", "TE"])
                pass_stack: List[Any] = ["QB"] + [("WR", "TE")] * primary_size
                opt_primary.add_stack(PositionsStack(pass_stack))
            if len(opt_primary.player_pool.available_teams) >= 3:
                opt_primary.set_total_teams(min_teams=3)
            elif len(opt_primary.player_pool.available_teams) == 2:
                opt_primary.set_total_teams(min_teams=2)

            opt_primary.settings.max_from_one_team = 4
            opt_primary.settings.budget = self.config.salary_cap
            for p in opt_primary.player_pool.filtered_players:
                p.max_exposure = cand_max_exp if p.fppg > 0.0 else 0.0

            accepted_sets = [set(p.id for p in c.lineup) for c in candidates]
            t1 = time.time()
            solved_primary = 0
            search_n = max(50, needed_primary * 5)
            for lineup in opt_primary.optimize(n=search_n):
                l_set = set(p.id for p in lineup.lineup)
                if any(len(l_set & prev_set) > max_rep for prev_set in accepted_sets):
                    continue
                candidates.append(lineup)
                accepted_sets.append(l_set)
                solved_primary += 1
                if solved_primary % 100 == 0 or solved_primary == needed_primary:
                    logger.info("... Solved %d / %d primary stacked candidate lineups (%.1fs) ...", solved_primary, needed_primary, time.time() - t1)
                if solved_primary == needed_primary:
                    break

        # Phase 3: Unconstrained Lineups
        needed_unconstrained = n_total - len(candidates)
        if needed_unconstrained > 0:
            logger.info("Solving %d unconstrained candidate lineups...", needed_unconstrained)
            opt_unconstrained = get_optimizer(site, Sport.FOOTBALL)
            opt_unconstrained.player_pool.load_players(self.players)
            opt_unconstrained.player_pool.with_injured = True
            opt_unconstrained.set_max_repeating_players(max_rep)
            opt_unconstrained.set_fantasy_points_strategy(
                RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
            )
            if max_own is not None and max_own > 0:
                has_own = any(p.projected_ownership is not None for p in opt_unconstrained.player_pool.filtered_players)
                if has_own:
                    avg_own = max_own / total_slots
                    if avg_own > 1.0:
                        avg_own /= 100.0
                    opt_unconstrained.set_projected_ownership(max_projected_ownership=avg_own)
            if not is_single_game:
                opt_unconstrained.restrict_positions_for_opposing_team(def_pos, ["QB", "RB", "WR", "TE"])
            if len(opt_unconstrained.player_pool.available_teams) >= 3:
                opt_unconstrained.set_total_teams(min_teams=3)
            elif len(opt_unconstrained.player_pool.available_teams) == 2:
                opt_unconstrained.set_total_teams(min_teams=2)

            opt_unconstrained.settings.max_from_one_team = 4
            opt_unconstrained.settings.budget = self.config.salary_cap
            for p in opt_unconstrained.player_pool.filtered_players:
                p.max_exposure = cand_max_exp if p.fppg > 0.0 else 0.0

            accepted_sets = [set(p.id for p in c.lineup) for c in candidates]

            t2 = time.time()
            solved_unconstrained = 0
            search_n = max(50, needed_unconstrained * 5)
            for lineup in opt_unconstrained.optimize(n=search_n):
                l_set = set(p.id for p in lineup.lineup)
                if any(len(l_set & prev_set) > max_rep for prev_set in accepted_sets):
                    continue
                candidates.append(lineup)
                accepted_sets.append(l_set)
                solved_unconstrained += 1
                if solved_unconstrained % 50 == 0 or solved_unconstrained == needed_unconstrained:
                    logger.info(
                        "... Solved %d / %d unconstrained candidate lineups (%.1fs) ...",
                        solved_unconstrained,
                        needed_unconstrained,
                        time.time() - t2,
                    )
                if solved_unconstrained == needed_unconstrained:
                    break

        logger.info("Successfully generated %d candidate lineups. Building binary matrix...", len(candidates))

        # Encode candidates into binary/weighted NumPy matrix C in {0, 1, 1.5}^(N x P)
        P = len(self.players)
        candidates_matrix = np.zeros((len(candidates), P), dtype=np.float32)

        for row_idx, lineup in enumerate(candidates):
            for player in lineup.lineup:
                col_idx = self.player_to_idx.get(str(player.id))
                if col_idx is not None:
                    is_mvp = is_single_game and "MVP" in (getattr(player, "lineup_position", "") or "")
                    candidates_matrix[row_idx, col_idx] = 1.5 if is_mvp else 1.0

        return candidates, candidates_matrix
