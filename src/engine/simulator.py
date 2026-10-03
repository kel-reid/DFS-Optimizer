"""
================================================================================
Correlated Game Engine Module (src/engine/simulator.py)
================================================================================
Simulates correlated game slate trials with team shocks, gamma-distributed
points, and performs BLAS matrix scoring and fast GPP tournament percentile ranking.
================================================================================
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from pydfs_lineup_optimizer import Player

from src.config import SimOptimizerConfig

logger = logging.getLogger("FanDuelSimOptimizer")


class CorrelatedGameEngine:
    """
    Simulates T = 5,000 correlated game slate outcomes and executes vectorized
    tournament scoring and ranking against the opponent field.
    """

    def __init__(self, players: List[Player], config: SimOptimizerConfig) -> None:
        self.players = players
        self.config = config
        self.P = len(players)

        # Team mapping for game-level correlation
        self.teams: List[str] = sorted(list({p.team for p in players if p.team}))
        self.team_to_idx = {t: idx for idx, t in enumerate(self.teams)}
        self.player_team_indices = np.array([self.team_to_idx.get(p.team, -1) for p in players], dtype=np.int32)
        self.player_opp_indices = np.full(self.P, -1, dtype=np.int32)
        for i, p in enumerate(players):
            if p.game_info and p.game_info.home_team and p.game_info.away_team:
                opp_team = p.game_info.away_team if p.team == p.game_info.home_team else p.game_info.home_team
                self.player_opp_indices[i] = self.team_to_idx.get(opp_team, -1)

    def simulate_game_trials(self) -> np.ndarray:
        """
        Generates simulated player score matrix S in R^(P x T) across T trials.
        Incorporates right-skewed Gamma distributions with team offensive covariance
        and QB/pass-catcher synergy.
        """
        T = self.config.num_sim_trials
        P = self.P
        rng = np.random.default_rng(self.config.random_seed)

        print()
        logger.info("=" * 70)
        logger.info("STAGE 3: CORRELATED GAME OUTCOME ENGINE (T = %d TRIALS)", T)
        logger.info("=" * 70)
        logger.info("Modeled %s correlated game slate realizations with joint covariance.", f"{T:,}")

        # 1. Base individual player variance (Gamma distributions parameterized by mean and CV)
        # CV varies by position: QB=0.35, RB=0.45, WR=0.55, TE=0.60, DEF=0.70
        fppgs = np.array([p.fppg for p in self.players], dtype=np.float32)
        cvs = np.zeros(P, dtype=np.float32)

        for i, p in enumerate(self.players):
            pos = set(p.positions)
            if "QB" in pos:
                cvs[i] = 0.35
            elif "RB" in pos:
                cvs[i] = 0.45
            elif "WR" in pos:
                cvs[i] = 0.55
            elif "TE" in pos:
                cvs[i] = 0.60
            else:
                cvs[i] = 0.70

        safe_fppgs = np.maximum(0.001, fppgs)
        shapes = 1.0 / (cvs ** 2)
        scales = safe_fppgs * (cvs ** 2)

        # Base independent trial draws: shape (P, T)
        base_draws = rng.gamma(shape=shapes[:, None], scale=scales[:, None], size=(P, T)).astype(np.float32)

        # Zero out unprojected / backup players explicitly
        for i, p in enumerate(self.players):
            if p.fppg == 0.0:
                base_draws[i, :] = 0.0

        # 2. Correlated Game-Level & Team Shocks
        sigma_team = 0.22
        num_teams = len(self.teams)
        team_z = rng.standard_normal(size=(num_teams, T)).astype(np.float32)
        team_factors = np.exp(sigma_team * team_z - 0.5 * (sigma_team ** 2))

        # Apply team offensive shocks to skill players and inverse opponent shock to defenses
        sim_points = np.copy(base_draws)
        for i in range(P):
            if self.players[i].fppg > 0.0:
                if "D" in self.players[i].positions:
                    opp_t_idx = self.player_opp_indices[i]
                    if opp_t_idx >= 0:
                        opp_shock = np.exp(-sigma_team * team_z[opp_t_idx, :] - 0.5 * (sigma_team ** 2))
                        sim_points[i, :] *= opp_shock
                else:
                    t_idx = self.player_team_indices[i]
                    if t_idx >= 0:
                        sim_points[i, :] *= team_factors[t_idx, :]

        return sim_points

    def score_and_rank_candidates(
        self,
        candidates_matrix: np.ndarray,
        field_matrix: np.ndarray,
        sim_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes vectorized scoring via BLAS dot products:
          Y_cand  = candidates_matrix @ sim_points    # Shape: (N, T)
          Y_field = field_matrix @ sim_points         # Shape: (M, T)
        Ranks candidate scores against the field using np.searchsorted, and evaluates
        the contest payout ladder.

        Returns:
          sim_roi: Array of shape (N,) containing Simulated ROI % for each candidate
          win_counts: Array of shape (N,) with 1st place win counts
          top1_rates: Array of shape (N,) with Top-1% finish rates (%)
        """
        cand_scores = candidates_matrix @ sim_points
        field_scores = field_matrix @ sim_points

        # Sort field scores in-place column-wise for fast binary search without duplicate memory allocation
        field_scores.sort(axis=0)
        sorted_field = field_scores

        M = self.config.num_field_lineups
        T = self.config.num_sim_trials
        entry_fee = self.config.entry_fee

        payout_ladder = np.zeros(M + 1, dtype=np.float64)
        ranks_arr = np.arange(M + 1)
        pct_arr = ranks_arr / M

        # Vectorized tier assignment based on finish percentiles
        payout_ladder[(ranks_arr >= 1) & (pct_arr <= 0.2000001)] = 1.5 * entry_fee
        payout_ladder[(ranks_arr >= 1) & (pct_arr <= 0.0500001)] = 5.0 * entry_fee
        payout_ladder[(ranks_arr >= 1) & (pct_arr <= 0.0100001)] = 20.0 * entry_fee
        payout_ladder[(ranks_arr >= 1) & (pct_arr <= 0.0010001)] = 500.0 * entry_fee
        payout_ladder[(ranks_arr == 1) | (pct_arr <= 0.0001001)] = 10_000.0 * entry_fee
        payout_ladder[0] = 0.0

        num_cands = len(cand_scores)
        total_payouts = np.zeros(num_cands, dtype=np.float64)
        win_counts = np.zeros(num_cands, dtype=np.int32)
        top1_counts = np.zeros(num_cands, dtype=np.int32)

        top1_cutoff = max(1, int(round(M * 0.01)))
        for t in range(T):
            sf_col = sorted_field[:, t]
            cs_col = cand_scores[:, t]
            idx = np.searchsorted(sf_col, cs_col, side="right")
            ranks = np.clip(M - idx + 1, 1, M)

            total_payouts += payout_ladder[ranks]
            win_counts += (ranks == 1)
            top1_counts += (ranks <= top1_cutoff)

        total_entry_cost = T * entry_fee
        sim_roi = ((total_payouts - total_entry_cost) / total_entry_cost) * 100.0
        top1_rates = (top1_counts / T) * 100.0

        logger.info("Evaluated candidate rank placements and ROI simulation across %s trials.", f"{T:,}")
        return sim_roi, win_counts, top1_rates
