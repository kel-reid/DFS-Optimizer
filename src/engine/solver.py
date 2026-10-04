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

from src.config import SimOptimizerConfig

logger = logging.getLogger("FanDuelSimOptimizer")


class CandidatePoolGenerator:
    """
    Generates N = 500 valid, diverse, correlated tournament candidate rosters
    using the PuLP/CBC integer linear programming solver, and encodes them into
    a binary NumPy matrix of shape (500, P).
    """

    def __init__(self, optimizer: LineupOptimizer, config: SimOptimizerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.players: List[Player] = list(optimizer.player_pool.all_players)
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

        # Baseline constraints for candidates
        self.optimizer.set_max_repeating_players(self.config.max_repeating_players)
        self.optimizer.set_fantasy_points_strategy(
            RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
        )
        self.optimizer.restrict_positions_for_opposing_team(["D"], ["QB", "RB", "WR", "TE"])
        # Ensure diversity across candidates by setting max 35% exposure during candidate phase
        for p in self.optimizer.player_pool.all_players:
            p.max_exposure = 0.35

        # Phase 1: Primary Stacked Lineups
        logger.info("Solving %d primary stacked candidate lineups (QB + same-team WR/TE)...", n_stacked)
        self.optimizer.add_stack(PositionsStack(["QB", ("WR", "TE")]))

        candidates: List[Lineup] = []
        t0 = time.time()
        for i, lineup in enumerate(self.optimizer.optimize(n=n_stacked), start=1):
            candidates.append(lineup)
            if i % 100 == 0 or i == n_stacked:
                logger.info("... Solved %d / %d stacked candidate lineups (%.1fs) ...", i, n_stacked, time.time() - t0)

        # Phase 2: Unconstrained Lineups
        if n_unconstrained > 0:
            logger.info("Solving %d unconstrained candidate lineups (standalone rushing QB ceiling)...", n_unconstrained)
            opt_unconstrained = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
            opt_unconstrained.player_pool.load_players(self.players)
            opt_unconstrained.player_pool.with_injured = True
            opt_unconstrained.set_max_repeating_players(self.config.max_repeating_players)
            opt_unconstrained.set_fantasy_points_strategy(
                RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
            )
            opt_unconstrained.restrict_positions_for_opposing_team(["D"], ["QB", "RB", "WR", "TE"])
            for p in opt_unconstrained.player_pool.all_players:
                p.max_exposure = 0.35

            accepted_sets = [set(p.id for p in c.lineup) for c in candidates]
            max_rep = self.config.max_repeating_players

            t1 = time.time()
            solved_unconstrained = 0
            for lineup in opt_unconstrained.optimize(n=n_unconstrained * 2):
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

        # Encode candidates into binary NumPy matrix C in {0, 1}^(N x P)
        P = len(self.players)
        candidates_matrix = np.zeros((len(candidates), P), dtype=np.float32)

        for row_idx, lineup in enumerate(candidates):
            for player in lineup.lineup:
                col_idx = self.player_to_idx.get(str(player.id))
                if col_idx is not None:
                    candidates_matrix[row_idx, col_idx] = 1.0

        return candidates, candidates_matrix
