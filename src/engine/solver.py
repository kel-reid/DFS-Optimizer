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
from typing import Dict, List, Tuple

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

from src.config import BaseOptimizerConfig

logger = logging.getLogger("FanDuelSimOptimizer")


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
          - 400 Primary Stacked (QB + >= 1 same-team WR/TE)
          - 100 Unconstrained (capturing standalone rushing QB ceiling)
        Returns:
          lineups: List of 500 Lineup objects
          candidates_matrix: Binary ndarray of shape (500, P)
        """
        n_total = self.config.num_candidates
        ratio = self.config.stack_ratio
        n_stacked = round(n_total * ratio)       # e.g., 400
        n_unconstrained = n_total - n_stacked    # e.g., 100

        print()
        logger.info("=" * 70)
        logger.info("STAGE 1: GENERATING CANDIDATE POOL (N = %d LINEUPS)", n_total)
        logger.info("=" * 70)
        logger.info(
            "Candidate Pool Allocation: %d Stacked (%.0f%%) + %d Unconstrained (%.0f%%)",
            n_stacked,
            ratio * 100,
            n_unconstrained,
            (1 - ratio) * 100,
        )

        total_slots = self.optimizer.settings.get_total_players()
        is_single_game = total_slots < 9
        max_rep = min(self.config.max_repeating_players, total_slots - 1)
        site = self.optimizer.settings.site
        def_pos = ["DST"] if site == Site.DRAFTKINGS else ["D"]

        # Baseline constraints for candidates
        self.optimizer.set_max_repeating_players(max_rep)
        self.optimizer.set_fantasy_points_strategy(
            RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
        )
        if not is_single_game:
            self.optimizer.restrict_positions_for_opposing_team(def_pos, ["QB", "RB", "WR", "TE"])
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

        for p in self.optimizer.player_pool.filtered_players:
            p.max_exposure = cand_max_exp if p.fppg > 0.0 else 0.0

        # Phase 1: Primary Stacked Lineups
        if not is_single_game:
            logger.info("Solving %d primary stacked candidate lineups (QB + same-team WR/TE)...", n_stacked)
            self.optimizer.add_stack(PositionsStack(["QB", ("WR", "TE")]))
        else:
            logger.info("Solving %d candidate lineups for Single Game slate...", n_stacked)

        candidates: List[Lineup] = []
        t0 = time.time()
        for i, lineup in enumerate(self.optimizer.optimize(n=n_stacked), start=1):
            candidates.append(lineup)
            if i % 100 == 0 or i == n_stacked:
                logger.info("... Solved %d / %d candidate lineups (%.1fs) ...", i, n_stacked, time.time() - t0)

        # Phase 2: Unconstrained Lineups
        if n_unconstrained > 0:
            logger.info("Solving %d unconstrained candidate lineups...", n_unconstrained)
            opt_unconstrained = get_optimizer(site, Sport.FOOTBALL)
            opt_unconstrained.player_pool.load_players(self.players)
            opt_unconstrained.player_pool.with_injured = True
            opt_unconstrained.set_max_repeating_players(max_rep)
            opt_unconstrained.set_fantasy_points_strategy(
                RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
            )
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

            t1 = time.time()
            solved_unconstrained = 0
            search_n = max(50, n_unconstrained * 5)
            for lineup in opt_unconstrained.optimize(n=search_n):
                l_set = set(p.id for p in lineup.lineup)
                if any(len(l_set & prev_set) > max_rep for prev_set in accepted_sets):
                    continue
                candidates.append(lineup)
                accepted_sets.append(l_set)
                solved_unconstrained += 1
                if solved_unconstrained % 50 == 0 or solved_unconstrained == n_unconstrained:
                    logger.info(
                        "... Solved %d / %d unconstrained candidate lineups (%.1fs) ...",
                        solved_unconstrained,
                        n_unconstrained,
                        time.time() - t1,
                    )
                if solved_unconstrained == n_unconstrained:
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
