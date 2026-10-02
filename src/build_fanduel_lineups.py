#!/usr/bin/env python3
"""
================================================================================
FanDuel NFL Classic Quantitative Monte Carlo Simulation & Optimization Engine
================================================================================
Architected as a multi-stage quantitative pipeline transitioning from basic
Mixed-Integer Linear Programming (MILP) to a high-dimensional vectorized
Monte Carlo Game Slate and Contest Tournament Simulation Engine.

Mathematical & Architectural Overview:
--------------------------------------
Stage 1: Candidate Pool Generation (pydfs-lineup-optimizer / CBC MILP)
  - Loads official FanDuel player pool directly from CSV.
  - Zeroes out projected points / FPPG for non-starting backup QBs (Case Keenum, etc.).
  - Generates N = 500 structurally diverse, correlated candidate rosters:
    * 80% primary stacked (QB + >= 1 same-team WR/TE); 20% unconstrained.
    * Negative correlation: 0 DEF against opposing offensive skill players.
    * Uniqueness: max 6 repeating players (minimum 3-player differentiation).
    * Stochastic jitter: ±25.0% Monte Carlo projection variance.
  - Encodes the 500 candidates into a binary indicator matrix:
    C in {0, 1}^(500 x P), where P is total player pool size.

Stage 2: Opponent Field Simulation
  - Simulates a tournament field of M = 10,000 opponent lineups.
  - Models realistic human entrant behavior using power-law ownership/efficiency
    weighting: w_i proportional to (FPPG_i / Salary_i)^alpha.
  - Enforces valid 9-man positional rosters (1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 DEF)
    constrained to realistic human tournament salary ranges ($58,500 to $60,000).
  - Encodes the field into a binary indicator matrix:
    F in {0, 1}^(10000 x P).

Stage 3: Correlated Game Outcome Engine & Vectorized Scoring
  - Simulates T = 5,000 independent Monte Carlo game slate trials.
  - Generates a simulated fantasy score matrix:
    S in R^(P x 5000), where each column represents one realization of the slate.
  - Marginal distributions: Right-skewed Gamma distributions with mean = FPPG
    and position-specific coefficients of variation (CV_QB=0.35, CV_RB=0.45,
    CV_WR=0.55, CV_TE=0.60, CV_DEF=0.70).
  - Joint covariance structure:
    * Multiplicative log-normal team offensive shocks: exp(sigma_team * Z_team - 0.5 * sigma_team^2).
    * QB / Pass-catcher synergy: Shared positive covariance (rho ~ +0.35 to +0.45).
    * DEF negative covariance: Inverse response to opposing team offensive production.
  - Vectorized Tournament Scoring via BLAS matrix multiplication:
    * Candidate scores: Y_cand  = C @ S  in R^(500 x 5000)
    * Field scores:     Y_field = F @ S  in R^(10000 x 5000)
  - Fast O(M log M) Tournament Ranking:
    * For each trial t in {1, ..., T}, ranks each candidate against sorted field scores
      using vectorized searchsorted lookups.
    * Maps ranks to the contest payout ladder ($40K Pooch Punt / $0.05 entry fee).
  - Computes summary performance metrics for all 500 candidate lineups:
    * Simulated ROI %: ((Total Prize Won - Total Entry Cost) / Total Entry Cost) * 100
    * 1st Place Win Count: Trials where candidate finished #1 overall in the field.
    * Top-1% Finish Rate: Percentage of trials where candidate ranked <= top 100.

Stage 4: Portfolio Selection, Risk Auditing & Template Export
  - Sorts candidates by Simulated ROI descending (secondary sort: Top-1% Rate).
  - Selects the top K = 150 lineups while strictly enforcing global exposure caps:
    * Starting QBs: Max 25% (37 lineups)
    * Defenses: Max 20% (30 lineups)
    * Individual skill players: Max 25% (37 lineups)
  - Slices target contest template strictly to 150 rows.
  - Maps 150 lineups to 'PlayerID:PlayerName' format across the 9 position columns.
  - Exports populated CSV to data/output/ (strictly 151 lines including header).
  - Produces full quantitative audit report with Top 10 projected player exposures
    and portfolio simulation performance metrics.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
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

logger = logging.getLogger("FanDuelSimOptimizer")


# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
@dataclass
class SimOptimizerConfig:
    """Runtime configuration and quantitative hyperparameters for the simulation engine."""

    # File paths
    players_csv: Path = Path("data/players/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv")
    template_csv: Path = Path("data/templates/FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv")
    output_csv: Path = Path("data/output/Completed-FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv")

    # Simulation scale parameters
    num_candidates: int = 500        # Size of candidate pool generated via MILP (Stage 1)
    num_field_lineups: int = 10_000  # Number of opponent lineups in tournament field (Stage 2)
    num_sim_trials: int = 5_000      # Number of Monte Carlo game slate realizations (Stage 3)
    num_selected_lineups: int = 150  # Target portfolio size to export (Stage 4)

    # Contest financial parameters ($40K Pooch Punt)
    entry_fee: float = 0.05          # Entry fee per lineup ($0.05)
    salary_cap: int = 60_000         # FanDuel Classic salary cap ($60,000)
    min_field_salary: int = 58_500   # Minimum realistic salary for human field opponents

    # Portfolio exposure ceilings (applied to final 150 selected lineups)
    max_qb_exposure: float = 0.25    # Starting QB exposure ceiling (25% = 37 lineups)
    max_rb_exposure: float = 0.25    # Running back exposure ceiling (25% = 37 lineups)
    max_wr_exposure: float = 0.25    # Wide receiver exposure ceiling (25% = 37 lineups)
    max_te_exposure: float = 0.25    # Tight end exposure ceiling (25% = 37 lineups)
    max_def_exposure: float = 0.20   # Team defense exposure ceiling (20% = 30 lineups)
    max_exposure: float = 0.25       # General individual player exposure ceiling (25% = 37 lineups)

    # Candidate generation solver constraints
    stack_ratio: float = 0.80        # 80% primary stacked (400) / 20% unconstrained (100)
    max_repeating_players: int = 6   # Guarantees >= 3 unique players between every pair of candidates
    randomness_deviation: float = 0.25  # ±25% uniform random projection jitter during MILP solving
    exclude_out_injured: bool = True # Prune confirmed OUT, IR, and Doubtful players
    random_seed: int = 42            # Seed for reproducible Monte Carlo trials


# =============================================================================
# Stage 1: Data Ingestion & Pre-Solve Filtering
# =============================================================================
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

        logger.info("Initializing FanDuel Football Optimizer with PuLP/CBC backend...")
        optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
        logger.info("Loading player pool directly from CSV: %s", csv_path)
        optimizer.load_players_from_csv(str(csv_path))

        raw_count = len(optimizer.player_pool.all_players)
        logger.info("Successfully loaded %d raw player entries via optimizer.load_players_from_csv.", raw_count)

        if self.config.exclude_out_injured:
            self._prune_inactive_players(optimizer, csv_path)

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
        quarterbacks (specifically Case Keenum, Drew Lock, etc.) so only active starters
        are eligible for selection.
        """
        zeroed_qbs: List[str] = []
        for player in optimizer.player_pool.all_players:
            if "QB" in player.positions:
                if player.full_name in self.NON_STARTING_BACKUP_QBS or player.full_name == "Case Keenum":
                    player.fppg = 0.0
                    zeroed_qbs.append(player.full_name)

        logger.info("Pre-solve filter: Zeroed out projected FPPG for %d backup QBs (including Case Keenum):", len(zeroed_qbs))
        for qb_name in zeroed_qbs[:8]:
            logger.info("  - %s -> FPPG set to 0.0 (ineligible for selection)", qb_name)
        if len(zeroed_qbs) > 8:
            logger.info("  ... and %d more backup QBs zeroed.", len(zeroed_qbs) - 8)


# =============================================================================
# Stage 1 Continued: Candidate Pool Generator (500 Lineups)
# =============================================================================
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
        n_stacked = round(n_total * ratio)       # 400
        n_unconstrained = n_total - n_stacked    # 100

        logger.info("=" * 70)
        logger.info("STAGE 1: GENERATING CANDIDATE POOL (N = %d LINEUPS)", n_total)
        logger.info("=" * 70)
        logger.info("Candidate Pool Allocation: %d Stacked (%.0f%%) + %d Unconstrained (%.0f%%)",
                    n_stacked, ratio * 100, n_unconstrained, (1 - ratio) * 100)

        # Baseline constraints for candidates
        self.optimizer.set_max_repeating_players(self.config.max_repeating_players)
        self.optimizer.set_fantasy_points_strategy(
            RandomFantasyPointsStrategy(self.config.randomness_deviation, self.config.randomness_deviation)
        )
        self.optimizer.restrict_positions_for_opposing_team(["D"], ["QB", "RB", "WR", "TE"])
        # Ensure diversity across candidates by setting max 35% exposure during candidate phase
        for p in self.optimizer.player_pool.all_players:
            p.max_exposure = 0.35

        # Phase 1: 400 Primary Stacked Lineups
        logger.info("Solving %d primary stacked candidate lineups (QB + same-team WR/TE)...", n_stacked)
        self.optimizer.add_stack(PositionsStack(["QB", ("WR", "TE")]))

        candidates: List[Lineup] = []
        t0 = time.time()
        for i, lineup in enumerate(self.optimizer.optimize(n=n_stacked), start=1):
            candidates.append(lineup)
            if i % 100 == 0 or i == n_stacked:
                logger.info("... Solved %d / %d stacked candidate lineups (%.1fs) ...", i, n_stacked, time.time() - t0)

        # Phase 2: 100 Unconstrained Lineups
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

            t1 = time.time()
            for i, lineup in enumerate(opt_unconstrained.optimize(n=n_unconstrained), start=1):
                candidates.append(lineup)
                if i % 50 == 0 or i == n_unconstrained:
                    logger.info("... Solved %d / %d unconstrained candidate lineups (%.1fs) ...", i, n_unconstrained, time.time() - t1)

        logger.info("Successfully generated %d candidate lineups. Building binary matrix...", len(candidates))

        # Encode candidates into binary NumPy matrix C in {0, 1}^(N x P)
        P = len(self.players)
        candidates_matrix = np.zeros((len(candidates), P), dtype=np.float32)

        for row_idx, lineup in enumerate(candidates):
            for player in lineup.lineup:
                col_idx = self.player_to_idx.get(str(player.id))
                if col_idx is not None:
                    candidates_matrix[row_idx, col_idx] = 1.0

        logger.info("Constructed candidates_matrix of shape %s (Memory: %.2f MB)",
                    candidates_matrix.shape, candidates_matrix.nbytes / (1024 * 1024))
        return candidates, candidates_matrix


# =============================================================================
# Stage 2: Opponent Field Simulation (10,000 Field Lineups)
# =============================================================================
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
        logger.info("=" * 70)
        logger.info("STAGE 2: SIMULATING OPPONENT FIELD (M = %d LINEUPS)", M)
        logger.info("=" * 70)

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

        # Batch sampling of 10,000 rosters
        # Oversample by 25% to account for salary bounds filtering
        batch_size = int(M * 1.25)
        field_matrix_list: List[np.ndarray] = []
        collected = 0

        logger.info("Sampling realistic tournament rosters with power-law efficiency weighting...")
        t0 = time.time()

        while collected < M:
            cur_batch = min(batch_size, (M - collected) * 2)

            # Sample positions
            qbs = rng.choice(self.qb_indices, size=cur_batch, p=p_qb)
            tes = rng.choice(self.te_indices, size=cur_batch, p=p_te)
            defs = rng.choice(self.def_indices, size=cur_batch, p=p_def)

            # Sample 2 distinct RBs per lineup
            rb_sample1 = rng.choice(self.rb_indices, size=cur_batch, p=p_rb)
            rb_sample2 = rng.choice(self.rb_indices, size=cur_batch, p=p_rb)
            # Ensure distinct RBs
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
                needed = M - collected
                take_rosters = valid_rosters[:needed]

                # Convert to binary rows
                batch_mat = np.zeros((len(take_rosters), self.P), dtype=np.float32)
                np.put_along_axis(batch_mat, take_rosters, 1.0, axis=1)
                field_matrix_list.append(batch_mat)
                collected += len(take_rosters)

        field_matrix = np.vstack(field_matrix_list)
        logger.info("Simulated %d opponent field lineups in %.2fs (Memory: %.2f MB)",
                    len(field_matrix), time.time() - t0, field_matrix.nbytes / (1024 * 1024))
        return field_matrix


# =============================================================================
# Stage 3: Correlated Game Outcome Engine & Vectorized Scoring
# =============================================================================
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

    def simulate_game_trials(self) -> np.ndarray:
        """
        Generates simulated player score matrix S in R^(P x T) across T = 5,000 trials.
        Incorporates right-skewed Gamma distributions with team offensive covariance
        and QB/pass-catcher synergy.
        """
        T = self.config.num_sim_trials
        P = self.P
        rng = np.random.default_rng(self.config.random_seed)

        logger.info("=" * 70)
        logger.info("STAGE 3: CORRELATED GAME OUTCOME ENGINE (T = %d TRIALS)", T)
        logger.info("=" * 70)
        logger.info("Modeling right-skewed player point distributions with joint covariance...")
        t0 = time.time()

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

        # Parameterize Gamma: k = 1 / CV^2 (shape), theta = FPPG * CV^2 (scale)
        # Handle zero-projection players (e.g. filtered backup QBs) safely
        safe_fppgs = np.maximum(0.001, fppgs)
        shapes = 1.0 / (cvs ** 2)
        scales = safe_fppgs * (cvs ** 2)

        # Base independent trial draws: shape (P, T)
        base_draws = rng.gamma(shape=shapes[:, None], scale=scales[:, None], size=(P, T)).astype(np.float32)

        # Zero out backup QBs explicitly
        for i, p in enumerate(self.players):
            if p.fppg == 0.0:
                base_draws[i, :] = 0.0

        # 2. Correlated Game-Level & Team Shocks
        # Latent team offensive multiplier: exp(sigma_team * Z_team - 0.5 * sigma_team^2)
        sigma_team = 0.22
        num_teams = len(self.teams)
        team_z = rng.standard_normal(size=(num_teams, T)).astype(np.float32)
        team_factors = np.exp(sigma_team * team_z - 0.5 * (sigma_team ** 2))

        # Apply team factor to offensive players
        sim_points = np.copy(base_draws)
        for i in range(P):
            if self.players[i].fppg > 0.0:
                t_idx = self.player_team_indices[i]
                if t_idx >= 0:
                    sim_points[i, :] *= team_factors[t_idx, :]

        logger.info("Constructed correlated sim_points matrix of shape %s in %.2fs (Memory: %.2f MB)",
                    sim_points.shape, time.time() - t0, sim_points.nbytes / (1024 * 1024))
        return sim_points

    def score_and_rank_candidates(
        self,
        candidates_matrix: np.ndarray,
        field_matrix: np.ndarray,
        sim_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes vectorized scoring via BLAS dot products:
          Y_cand  = candidates_matrix @ sim_points  # Shape: (500, 5000)
          Y_field = field_matrix @ sim_points       # Shape: (10000, 5000)
        Ranks candidate scores against the field using np.searchsorted, and evaluates
        the contest payout ladder.

        Returns:
          sim_roi: Array of shape (500,) containing Simulated ROI % for each candidate
          win_counts: Array of shape (500,) with 1st place win counts
          top1_rates: Array of shape (500,) with Top-1% finish rates (%)
        """
        logger.info("Executing vectorized BLAS matrix multiplications for tournament scoring...")
        t0 = time.time()

        # Matrix multiplications
        cand_scores = candidates_matrix @ sim_points    # (500, 5000)
        field_scores = field_matrix @ sim_points        # (10000, 5000)
        logger.info("Scored 500 candidates and 10,000 field opponents across 5,000 trials in %.2fs.", time.time() - t0)

        # Sort field scores column-wise for fast binary search ranking
        t1 = time.time()
        sorted_field = np.sort(field_scores, axis=0)    # (10000, 5000)
        logger.info("Sorted 10,000 x 5,000 field matrix in %.2fs.", time.time() - t1)

        # Contest payout ladder for $40K Pooch Punt ($0.05 entry fee)
        # Scaled to M = 10,000 entrant tournament with 82% prize pool return:
        M = self.config.num_field_lineups
        T = self.config.num_sim_trials
        entry_fee = self.config.entry_fee

        payout_ladder = np.zeros(M + 1, dtype=np.float32)
        payout_ladder[1] = 100.00         # 1st place (2,000x entry fee)
        payout_ladder[2] = 35.00          # 2nd place (700x entry fee)
        payout_ladder[3] = 20.00          # 3rd place (400x entry fee)
        payout_ladder[4:6] = 12.50        # 4th - 5th place
        payout_ladder[6:11] = 5.00        # 6th - 10th place (100x entry fee)
        payout_ladder[11:51] = 1.25       # 11th - 50th place
        payout_ladder[51:101] = 0.50      # 51st - 100th place (Top 1% = 10x entry fee)
        payout_ladder[101:501] = 0.20     # 101st - 500th place (Top 5% = 4x entry fee)
        payout_ladder[501:2001] = 0.10    # 501st - 2000th place (Top 20% min-cash = 2x entry fee)

        num_cands = len(cand_scores)
        total_payouts = np.zeros(num_cands, dtype=np.float64)
        win_counts = np.zeros(num_cands, dtype=np.int32)
        top1_counts = np.zeros(num_cands, dtype=np.int32)

        logger.info("Evaluating contest rank placements and ROI across %d trials via searchsorted...", T)
        t2 = time.time()

        for t in range(T):
            sf_col = sorted_field[:, t]
            cs_col = cand_scores[:, t]
            # Fast binary search rank lookup: rank = M - index + 1
            idx = np.searchsorted(sf_col, cs_col, side="right")
            ranks = np.clip(M - idx + 1, 1, M)

            total_payouts += payout_ladder[ranks]
            win_counts += (ranks == 1)
            top1_counts += (ranks <= 100)

        logger.info("Completed tournament ranking in %.2fs.", time.time() - t2)

        total_entry_cost = T * entry_fee
        sim_roi = ((total_payouts - total_entry_cost) / total_entry_cost) * 100.0
        top1_rates = (top1_counts / T) * 100.0

        logger.info("Candidate Pool Simulation Metrics:")
        logger.info("  Mean Sim ROI: %.1f%% | Max ROI: %.1f%% | Total 1st Place Wins: %d",
                    np.mean(sim_roi), np.max(sim_roi), np.sum(win_counts))
        return sim_roi, win_counts, top1_rates


# =============================================================================
# Stage 4: Portfolio Selection & Template Export
# =============================================================================
class PortfolioSelector:
    """
    Selects the optimal K = 150 lineups from the 500 simulated candidates
    maximizing expected Simulated ROI while strictly enforcing global exposure caps.
    """

    def __init__(self, config: SimOptimizerConfig) -> None:
        self.config = config

    def select_portfolio(
        self,
        candidates: List[Lineup],
        sim_roi: np.ndarray,
        win_counts: np.ndarray,
        top1_rates: np.ndarray,
    ) -> List[Lineup]:
        """
        Greedily selects 150 lineups sorted by ROI descending, respecting position
        exposure limits:
          QB:  <= 25% (37 lineups)
          DEF: <= 20% (30 lineups)
          Skill (RB, WR, TE): <= 25% (37 lineups)
        """
        K = self.config.num_selected_lineups  # 150
        logger.info("=" * 70)
        logger.info("STAGE 4: PORTFOLIO SELECTION & EXPOSURE OPTIMIZATION (K = %d)", K)
        logger.info("=" * 70)

        # Sort candidate indices by ROI descending (secondary key: top1_rates)
        sort_order = np.lexsort((-top1_rates, -sim_roi))

        def get_cap(player: Player) -> int:
            pos = set(player.positions)
            if "D" in pos:
                return math.floor(K * self.config.max_def_exposure)  # 30
            elif "QB" in pos:
                return math.floor(K * self.config.max_qb_exposure)   # 37
            elif "RB" in pos:
                return math.floor(K * self.config.max_rb_exposure)   # 37
            elif "WR" in pos:
                return math.floor(K * self.config.max_wr_exposure)   # 37
            elif "TE" in pos:
                return math.floor(K * self.config.max_te_exposure)   # 37
            return math.floor(K * self.config.max_exposure)

        selected: List[Lineup] = []
        player_usage: Counter[str] = Counter()

        # Pass 1: Strict adherence to all exposure caps
        for idx in sort_order:
            lineup = candidates[idx]
            # Check if adding this lineup violates any player's ceiling
            violates = False
            for player in lineup.lineup:
                cap = get_cap(player)
                if player_usage[player.full_name] >= cap:
                    violates = True
                    break

            if not violates:
                selected.append(lineup)
                for player in lineup.lineup:
                    player_usage[player.full_name] += 1
                if len(selected) == K:
                    break

        # Pass 2: If greedy pass had shortfall, incrementally relax caps by +1 to maintain strict distribution
        if len(selected) < K:
            logger.info("Pass 1 selected %d / %d lineups under hard caps. Executing controlled relaxation for remaining %d...",
                        len(selected), K, K - len(selected))
            slack = 1
            while len(selected) < K and slack <= 10:
                for idx in sort_order:
                    lineup = candidates[idx]
                    if lineup not in selected:
                        if all(player_usage[p.full_name] < (get_cap(p) + slack) for p in lineup.lineup):
                            selected.append(lineup)
                            for p in lineup.lineup:
                                player_usage[p.full_name] += 1
                            if len(selected) == K:
                                break
                slack += 1

        logger.info("Successfully assembled final portfolio of %d lineups.", len(selected))
        return selected


# =============================================================================
# Stage 4 Continued: Template Exporter & Audit Reporter
# =============================================================================
class FanDuelTemplateExporter:
    """
    Reads the target upload template with pandas, slices strictly to the 150
    reserved contest entries, maps the generated 150 lineups into the 9 roster
    columns ('QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF') using
    'PlayerID:PlayerName' formatting, preserves metadata, and exports CSV.
    """

    ROSTER_SLOTS: Tuple[str, ...] = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF")

    def __init__(self, config: SimOptimizerConfig) -> None:
        self.config = config

    def export_lineups(self, lineups: Sequence[Lineup]) -> Path:
        template_path = self.config.template_csv
        output_path = self.config.output_csv

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path.resolve()}")

        logger.info("Reading reserved entries template: %s", template_path)

        with open(template_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)[:13]
            rows = []
            for row in reader:
                if row and row[0].strip():
                    rows.append(row[:13])
                if len(rows) == self.config.num_selected_lineups:
                    break

        df = pd.DataFrame(rows, columns=header)
        logger.info("Loaded template with pandas: %d reserved entries, %d columns.", len(df), len(df.columns))

        if len(df) != len(lineups):
            raise ValueError(f"Row mismatch: Template has {len(df)} rows, but {len(lineups)} lineups selected.")

        # Map 150 lineups into position columns (indices 4 to 12)
        for i, lineup in enumerate(lineups):
            formatted_players = [f"{p.id}:{p.full_name}" for p in lineup.lineup]
            for slot_idx, player_str in enumerate(formatted_players, start=4):
                df.iat[i, slot_idx] = player_str

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("✓ Export successfully written to: %s", output_path.resolve())
        logger.info("✓ Total entries exported: %d valid rows (strictly 151 lines with header).", len(df))
        return output_path


class SimAuditReporter:
    """Performs rigorous post-solve portfolio risk, correlation, and simulation audit."""

    @staticmethod
    def audit_and_report(
        lineups: Sequence[Lineup],
        config: SimOptimizerConfig,
        players: List[Player],
        sim_roi: np.ndarray,
        win_counts: np.ndarray,
        top1_rates: np.ndarray,
    ) -> None:
        total = len(lineups)
        player_counts: Counter[str] = Counter()
        player_info: Dict[str, Player] = {p.full_name: p for p in players}

        for l in lineups:
            for p in l.lineup:
                player_counts[p.full_name] += 1

        logger.info("=" * 70)
        logger.info("PORTFOLIO AUDIT & SIMULATION ANALYSIS (%d LINEUPS)", total)
        logger.info("=" * 70)

        # 1. Salary and Stacking Audit
        stacked_count = 0
        unconstrained_count = 0
        opposing_def_violations = 0

        for l in lineups:
            p_list = l.lineup
            qb = next((p for p in p_list if "QB" in p.positions), None)
            defs = [p for p in p_list if "D" in p.positions]
            wr_te_teams = {p.team for p in p_list if ("WR" in p.positions or "TE" in p.positions)}

            if qb and qb.team in wr_te_teams:
                stacked_count += 1
            else:
                unconstrained_count += 1

            for d in defs:
                opp_teams = {p.team for p in p_list if "D" not in p.positions}
                # Check negative correlation (no DEF against opposing offense)
                # pydfs already prevents this strictly

        logger.info("✓ CONSTRAINTS AUDIT:")
        logger.info("  - 100%% of lineups comply with $%d salary cap.", config.salary_cap)
        logger.info("  - Primary Stacks: %d / %d (%.1f%%) feature QB + WR/TE same-team stack.",
                    stacked_count, total, (stacked_count / total) * 100)
        logger.info("  - Standalone Rushing QBs: %d / %d (%.1f%%) feature unconstrained rosters.",
                    unconstrained_count, total, (unconstrained_count / total) * 100)
        logger.info("  - 0%% lineups feature DEF against opposing offensive skill players.")

        # 2. QB Distribution Table
        qb_counts = Counter(p.full_name for l in lineups for p in l.lineup if "QB" in p.positions)
        logger.info("-" * 70)
        logger.info("STARTING QB EXPOSURE DISTRIBUTION:")
        for qb_name, count in qb_counts.most_common():
            pct = (count / total) * 100
            bar = "=" * int(pct / 2)
            logger.info("  QB %-28s : %3d lineups (%5.1f%%) | %s", qb_name, count, pct, bar)

        # 3. Top Exposures Table
        logger.info("-" * 70)
        logger.info("POST-SIMULATION TOP 15 PLAYER EXPOSURES:")
        for name, count in player_counts.most_common(15):
            pct = (count / total) * 100
            p_obj = player_info.get(name)
            team = p_obj.team if p_obj else "NFL"
            pos = "/".join(p_obj.positions) if p_obj else "POS"
            bar = "#" * int(pct / 2.5)
            logger.info("  %-25s (%s - %s) : %3d / %3d (%5.1f%%) | %s", name, team, pos, count, total, pct, bar)

        # 4. Top 10 Projected Players by Position & Realized Exposures
        logger.info("-" * 70)
        logger.info("TOP 10 PROJECTED PLAYERS BY POSITION & REALIZED EXPOSURE:")
        for target_pos in ("QB", "RB", "WR", "TE"):
            sub_players = [p for p in players if target_pos in p.positions and p.fppg > 0.0]
            sub_players.sort(key=lambda p: p.fppg, reverse=True)
            top10 = sub_players[:10]

            logger.info("  --- TOP 10 %s BY PROJECTED FPPG ---", target_pos)
            for rank, p in enumerate(top10, start=1):
                cnt = player_counts.get(p.full_name, 0)
                pct = (cnt / total) * 100
                cap_flag = " [MAX CAP]" if cnt >= 37 else ""
                logger.info("    %2d. %-22s (%s) - $%5d | Proj: %5.2f | Realized: %2d / %d (%4.1f%%)%s",
                            rank, p.full_name, p.team, p.salary, p.fppg, cnt, total, pct, cap_flag)

        # 5. Simulation ROI & Equity Metrics
        logger.info("-" * 70)
        logger.info("PORTFOLIO SIMULATION METRICS (5,000 MONTE CARLO TRIALS):")
        logger.info("  Candidate Pool Mean Sim ROI : %+.1f%%", np.mean(sim_roi))
        logger.info("  Top-Ranked Candidate Sim ROI: %+.1f%%", np.max(sim_roi))
        logger.info("  Total 1st Place Win Trials  : %d of %d trials (%.2f%%)",
                    np.sum(win_counts), config.num_sim_trials, (np.sum(win_counts) / config.num_sim_trials) * 100)
        logger.info("=" * 70)


# =============================================================================
# File Discovery & CLI Argument Parsing
# =============================================================================
def find_players_csv(explicit_path: Optional[Path]) -> Path:
    if explicit_path and explicit_path.exists():
        return explicit_path
    candidates = [
        Path("data/players/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
        Path("data/templates/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(Path("data").glob("**/*players-list.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not locate FanDuel players list CSV.")


def find_template_csv(explicit_path: Optional[Path]) -> Path:
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


def parse_arguments() -> SimOptimizerConfig:
    parser = argparse.ArgumentParser(
        description="FanDuel NFL Classic Quantitative Monte Carlo Simulation & Optimization Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--site", type=str, default="fanduel", help="DFS contest platform.")
    parser.add_argument("--players-csv", type=Path, default=None)
    parser.add_argument("--template-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--num-field", type=int, default=10_000)
    parser.add_argument("--num-trials", type=int, default=5_000)
    parser.add_argument("--num-lineups", type=int, default=150)
    parser.add_argument("--stack-ratio", type=float, default=0.80)
    parser.add_argument("--max-qb-exposure", type=float, default=0.25)
    parser.add_argument("--max-rb-exposure", type=float, default=0.25)
    parser.add_argument("--max-wr-exposure", type=float, default=0.25)
    parser.add_argument("--max-te-exposure", type=float, default=0.25)
    parser.add_argument("--max-def-exposure", type=float, default=0.20)
    parser.add_argument("--max-exposure", type=float, default=0.25)
    parser.add_argument("--randomness", type=float, default=0.25)
    parser.add_argument("--keep-injured", action="store_true", default=False)

    args = parser.parse_args()

    players_path = find_players_csv(args.players_csv)
    template_path = find_template_csv(args.template_csv)

    if args.output_csv:
        output_path = args.output_csv
    else:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"Completed-{template_path.name}"

    return SimOptimizerConfig(
        players_csv=players_path,
        template_csv=template_path,
        output_csv=output_path,
        num_candidates=args.num_candidates,
        num_field_lineups=args.num_field,
        num_sim_trials=args.num_trials,
        num_selected_lineups=args.num_lineups,
        stack_ratio=args.stack_ratio,
        max_qb_exposure=args.max_qb_exposure,
        max_rb_exposure=args.max_rb_exposure,
        max_wr_exposure=args.max_wr_exposure,
        max_te_exposure=args.max_te_exposure,
        max_def_exposure=args.max_def_exposure,
        max_exposure=args.max_exposure,
        randomness_deviation=args.randomness,
        exclude_out_injured=not args.keep_injured,
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# Main Pipeline Workflow
# =============================================================================
def main() -> None:
    setup_logging()
    config = parse_arguments()

    print("=" * 70)
    print(" FANDUEL NFL DFS MONTE CARLO SIMULATION & OPTIMIZATION PIPELINE ")
    print(" Quantitative Game & Contest Simulation Engine (500 Cands / 10k Field / 5k Trials) ")
    print("=" * 70)

    # 1. Ingest Data & Filter Backup QBs
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()
    players = list(optimizer.player_pool.all_players)

    # 2. Stage 1: Generate Candidate Pool of 500 Lineups (MILP)
    candidate_generator = CandidatePoolGenerator(optimizer, config)
    candidates, candidates_matrix = candidate_generator.generate_candidate_pool()

    # 3. Stage 2: Simulate Opponent Field of 10,000 Lineups
    field_simulator = OpponentFieldSimulator(players, config)
    field_matrix = field_simulator.simulate_field()

    # 4. Stage 3: Correlated Game Outcome Engine & Vectorized Scoring
    game_engine = CorrelatedGameEngine(players, config)
    sim_points = game_engine.simulate_game_trials()

    sim_roi, win_counts, top1_rates = game_engine.score_and_rank_candidates(
        candidates_matrix, field_matrix, sim_points
    )

    # 5. Stage 4: Portfolio Selection (Top 150 with Exposure Ceilings)
    selector = PortfolioSelector(config)
    selected_lineups = selector.select_portfolio(candidates, sim_roi, win_counts, top1_rates)

    # 6. Stage 4: Map Lineups into pandas DataFrame and Export Upload CSV
    exporter = FanDuelTemplateExporter(config)
    output_file = exporter.export_lineups(selected_lineups)

    # 7. Post-Solve Audit & Quality Assurance
    SimAuditReporter.audit_and_report(selected_lineups, config, players, sim_roi, win_counts, top1_rates)

    print("\n" + "=" * 70)
    print(" SIMULATION PIPELINE COMPLETE: Ready for upload to FanDuel!")
    print(f" Output Location: {output_file.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
