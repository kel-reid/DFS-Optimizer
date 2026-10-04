"""
================================================================================
Opponent Field Simulator Module (src/engine/field.py)
================================================================================
Simulates realistic tournament opponent field rosters (M = 10,000) using
power-law efficiency weighting and realistic salary bounds.
================================================================================
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from pydfs_lineup_optimizer import Player

from src.config import SimOptimizerConfig

logger = logging.getLogger("FanDuelSimOptimizer")


class OpponentFieldSimulator:
    """
    Simulates a realistic tournament field of M = 10,000 opponent lineups.
    Entrants construct rosters using power-law ownership/efficiency weighting:
      w_i proportional to (FPPG_i / Salary_i)^alpha, alpha = 2.2
    Lineups are constrained to human contest salary ranges [$58,500, $60,000].
    Encodes the field into binary NumPy matrix F in {0, 1}^(10000 x P).
    """

    def __init__(self, players: List[Player], config: SimOptimizerConfig) -> None:
        self.players = players
        self.config = config
        self.P = len(players)

        # Categorize player indices by position
        self.qb_indices: List[int] = []
        self.rb_indices: List[int] = []
        self.wr_indices: List[int] = []
        self.te_indices: List[int] = []
        self.def_indices: List[int] = []

        for idx, p in enumerate(players):
            pos = set(p.positions)
            if "QB" in pos and p.fppg > 0.0:  # Exclude zeroed backup QBs
                self.qb_indices.append(idx)
            elif "RB" in pos:
                self.rb_indices.append(idx)
            elif "WR" in pos:
                self.wr_indices.append(idx)
            elif "TE" in pos:
                self.te_indices.append(idx)
            elif "D" in pos:
                self.def_indices.append(idx)

        self.flex_indices = self.rb_indices + self.wr_indices + self.te_indices

    def simulate_field(self) -> np.ndarray:
        """
        Generates binary matrix field_matrix of shape (M, P).
        """
        M = self.config.num_field_lineups
        print()
        logger.info("=" * 70)
        logger.info("STAGE 2: SIMULATING OPPONENT FIELD (M = %d LINEUPS)", M)
        logger.info("=" * 70)
        logger.info("Sampled %s realistic tournament rosters with power-law efficiency weighting.", f"{M:,}")

        # Compute power-law selection probabilities based on points-per-dollar efficiency
        alpha = 2.2
        salaries = np.array([p.salary for p in self.players], dtype=np.float32)
        fppgs = np.array([max(0.1, p.fppg) for p in self.players], dtype=np.float32)

        efficiency = fppgs / (salaries / 1000.0)
        raw_weights = np.power(efficiency, alpha)

        def get_probs(indices: List[int]) -> np.ndarray:
            w = raw_weights[indices]
            s = np.sum(w)
            return w / s if s > 0 else np.ones(len(indices)) / len(indices)

        p_qb = get_probs(self.qb_indices)
        p_rb = get_probs(self.rb_indices)
        p_wr = get_probs(self.wr_indices)
        p_te = get_probs(self.te_indices)
        p_def = get_probs(self.def_indices)
        p_flex = get_probs(self.flex_indices)

        rng = np.random.default_rng(self.config.random_seed)

        # Batch sampling of rosters
        batch_size = int(M * 1.25)
        field_matrix_list: List[np.ndarray] = []
        collected = 0
        max_attempts = 100
        consecutive_empty = 0

        while collected < M:
            cur_batch = min(batch_size, (M - collected) * 2)

            # Sample positions
            qbs = rng.choice(self.qb_indices, size=cur_batch, p=p_qb)
            tes = rng.choice(self.te_indices, size=cur_batch, p=p_te)
            defs = rng.choice(self.def_indices, size=cur_batch, p=p_def)

            # Sample 2 distinct RBs per lineup
            rb_sample1 = rng.choice(self.rb_indices, size=cur_batch, p=p_rb)
            rb_sample2 = rng.choice(self.rb_indices, size=cur_batch, p=p_rb)
            collision = rb_sample1 == rb_sample2
            while np.any(collision):
                rb_sample2[collision] = rng.choice(self.rb_indices, size=np.sum(collision), p=p_rb)
                collision = rb_sample1 == rb_sample2

            # Sample 3 distinct WRs per lineup
            wr_sample1 = rng.choice(self.wr_indices, size=cur_batch, p=p_wr)
            wr_sample2 = rng.choice(self.wr_indices, size=cur_batch, p=p_wr)
            coll12 = wr_sample1 == wr_sample2
            while np.any(coll12):
                wr_sample2[coll12] = rng.choice(self.wr_indices, size=np.sum(coll12), p=p_wr)
                coll12 = wr_sample1 == wr_sample2

            wr_sample3 = rng.choice(self.wr_indices, size=cur_batch, p=p_wr)
            coll13 = (wr_sample3 == wr_sample1) | (wr_sample3 == wr_sample2)
            while np.any(coll13):
                wr_sample3[coll13] = rng.choice(self.wr_indices, size=np.sum(coll13), p=p_wr)
                coll13 = (wr_sample3 == wr_sample1) | (wr_sample3 == wr_sample2)

            # Sample 1 FLEX (distinct from rostered RB, WR, TE)
            flex_sample = rng.choice(self.flex_indices, size=cur_batch, p=p_flex)
            rostered_so_far = np.column_stack([rb_sample1, rb_sample2, wr_sample1, wr_sample2, wr_sample3, tes])
            coll_flex = np.any(rostered_so_far == flex_sample[:, None], axis=1)
            while np.any(coll_flex):
                flex_sample[coll_flex] = rng.choice(self.flex_indices, size=np.sum(coll_flex), p=p_flex)
                coll_flex = np.any(rostered_so_far == flex_sample[:, None], axis=1)

            # Assemble full 9-player indices
            rosters = np.column_stack([
                qbs, rb_sample1, rb_sample2, wr_sample1, wr_sample2, wr_sample3, tes, flex_sample, defs
            ])

            # Vectorized salary evaluation: sum salaries across the 9 players
            roster_salaries = np.sum(salaries[rosters], axis=1)
            valid_salary_mask = (roster_salaries >= self.config.min_field_salary) & (roster_salaries <= self.config.salary_cap)

            valid_rosters = rosters[valid_salary_mask]
            if len(valid_rosters) > 0:
                consecutive_empty = 0
                needed = M - collected
                take_rosters = valid_rosters[:needed]

                # Convert to binary rows
                batch_mat = np.zeros((len(take_rosters), self.P), dtype=np.float32)
                np.put_along_axis(batch_mat, take_rosters, 1.0, axis=1)
                field_matrix_list.append(batch_mat)
                collected += len(take_rosters)
            else:
                consecutive_empty += 1
                if consecutive_empty >= max_attempts:
                    raise RuntimeError(
                        f"Could not sample opponent field lineups within salary interval "
                        f"[${self.config.min_field_salary}, ${self.config.salary_cap}]. Check player pool salaries."
                    )

        field_matrix = np.vstack(field_matrix_list)
        return field_matrix
