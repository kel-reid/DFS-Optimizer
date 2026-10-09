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

        self.is_single_game = bool(getattr(self.config, "is_single_game", False))

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

        # Auto-detect Single Game if not explicitly set in config
        unique_teams = {p.team for p in players if p.team}
        if not self.is_single_game and len(unique_teams) <= 2 and (len(self.def_indices) == 0 or len(players) <= 100):
            self.is_single_game = True

    def simulate_field(self) -> np.ndarray:
        """
        Generates binary/weighted matrix field_matrix of shape (M, P).
        Dispatches to Single Game or Classic simulation depending on slate structure.
        """
        if self.is_single_game:
            return self._simulate_single_game_field()
        return self._simulate_classic_field()

    def _simulate_single_game_field(self) -> np.ndarray:
        """
        Simulates tournament field for Single Game / Showdown slates:
          - 5-player rosters: 1 MVP (1.5x score multiplier) + 4 AnyFLEX
          - Enforces team diversity (players from both teams) and salary interval
          - Encodes into weighted NumPy matrix F in {0, 1.0, 1.5}^(M x P)
        """
        M = self.config.num_field_lineups
        print()
        logger.info("=" * 70)
        logger.info("STAGE 2: SIMULATING OPPONENT FIELD - SINGLE GAME (M = %d LINEUPS)", M)
        logger.info("=" * 70)
        logger.info("Sampled %s realistic Single Game rosters (1 MVP + 4 AnyFLEX).", f"{M:,}")

        valid_indices = [idx for idx, p in enumerate(self.players) if p.fppg > 0.0]
        if len(valid_indices) < 5:
            raise RuntimeError(
                f"Insufficient active players ({len(valid_indices)}) to sample 5-player Single Game rosters."
            )

        salaries = np.array([p.salary for p in self.players], dtype=np.float32)
        fppgs = np.array([max(0.1, p.fppg) for p in self.players], dtype=np.float32)
        player_teams = np.array([p.team for p in self.players])
        avail_teams = np.unique(player_teams[valid_indices])
        check_two_teams = len(avail_teams) >= 2

        alpha = 2.2
        efficiency = fppgs / (salaries / 1000.0)

        # FLEX selection probabilities (power-law on points-per-dollar efficiency)
        raw_flex = np.power(efficiency[valid_indices], alpha)
        s_flex = np.sum(raw_flex)
        p_flex = raw_flex / s_flex if s_flex > 0 else np.ones(len(valid_indices)) / len(valid_indices)

        # MVP selection probabilities (stars dominate MVP ownership in GPP tournaments)
        raw_mvp = np.power(fppgs[valid_indices] * 1.5, 2.0) * efficiency[valid_indices]
        s_mvp = np.sum(raw_mvp)
        p_mvp = raw_mvp / s_mvp if s_mvp > 0 else np.ones(len(valid_indices)) / len(valid_indices)

        min_salary = min(self.config.min_field_salary, int(self.config.salary_cap * 0.88))
        max_salary = self.config.salary_cap

        rng = np.random.default_rng(self.config.random_seed)
        batch_size = int(M * 1.5)
        field_matrix_list: List[np.ndarray] = []
        collected = 0
        max_attempts = 100
        consecutive_empty = 0

        while collected < M:
            cur_batch = min(batch_size, (M - collected) * 3)

            # Sample 1 MVP
            mvp_sample = rng.choice(valid_indices, size=cur_batch, p=p_mvp)

            # Sample 4 distinct AnyFLEX players distinct from MVP
            f1 = rng.choice(valid_indices, size=cur_batch, p=p_flex)
            while np.any(f1 == mvp_sample):
                f1[f1 == mvp_sample] = rng.choice(valid_indices, size=np.sum(f1 == mvp_sample), p=p_flex)

            f2 = rng.choice(valid_indices, size=cur_batch, p=p_flex)
            coll2 = (f2 == mvp_sample) | (f2 == f1)
            while np.any(coll2):
                f2[coll2] = rng.choice(valid_indices, size=np.sum(coll2), p=p_flex)
                coll2 = (f2 == mvp_sample) | (f2 == f1)

            f3 = rng.choice(valid_indices, size=cur_batch, p=p_flex)
            coll3 = (f3 == mvp_sample) | (f3 == f1) | (f3 == f2)
            while np.any(coll3):
                f3[coll3] = rng.choice(valid_indices, size=np.sum(coll3), p=p_flex)
                coll3 = (f3 == mvp_sample) | (f3 == f1) | (f3 == f2)

            f4 = rng.choice(valid_indices, size=cur_batch, p=p_flex)
            coll4 = (f4 == mvp_sample) | (f4 == f1) | (f4 == f2) | (f4 == f3)
            while np.any(coll4):
                f4[coll4] = rng.choice(valid_indices, size=np.sum(coll4), p=p_flex)
                coll4 = (f4 == mvp_sample) | (f4 == f1) | (f4 == f2) | (f4 == f3)

            rosters = np.column_stack([mvp_sample, f1, f2, f3, f4])

            # Team diversity check
            if check_two_teams:
                first_team = avail_teams[0]
                r_teams = player_teams[rosters]
                team1_count = np.sum(r_teams == first_team, axis=1)
                valid_team_mask = (team1_count >= 1) & (team1_count <= 4)
            else:
                valid_team_mask = np.ones(cur_batch, dtype=bool)

            # Salary evaluation
            roster_salaries = np.sum(salaries[rosters], axis=1)
            valid_salary_mask = (roster_salaries >= min_salary) & (roster_salaries <= max_salary)
            valid_mask = valid_team_mask & valid_salary_mask
            valid_rosters = rosters[valid_mask]

            if len(valid_rosters) > 0:
                consecutive_empty = 0
                needed = M - collected
                take_rosters = valid_rosters[:needed]

                # Convert to weighted row vectors: MVP = 1.5, FLEX = 1.0
                batch_mat = np.zeros((len(take_rosters), self.P), dtype=np.float32)
                batch_mat[np.arange(len(take_rosters)), take_rosters[:, 0]] = 1.5
                for col_slot in range(1, 5):
                    batch_mat[np.arange(len(take_rosters)), take_rosters[:, col_slot]] = 1.0

                field_matrix_list.append(batch_mat)
                collected += len(take_rosters)
            else:
                consecutive_empty += 1
                if consecutive_empty >= max_attempts:
                    raise RuntimeError(
                        f"Could not sample Single Game field lineups within salary interval "
                        f"[${min_salary}, ${max_salary}]. Check player pool salaries."
                    )

        return np.vstack(field_matrix_list)

    def _simulate_classic_field(self) -> np.ndarray:
        """
        Simulates tournament field for Classic slates:
          - 9-player rosters: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 DEF
          - Encodes into binary NumPy matrix F in {0, 1}^(M x P)
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
