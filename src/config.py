"""
================================================================================
NFL DFS Simulation & Optimization Configurations (src/config.py)
================================================================================
Defines typed dataclasses for quantitative parameters, solver constraints,
exposure ceilings, and simulation hyperparameters across DFS platforms.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SimOptimizerConfig:
    """Runtime configuration and quantitative hyperparameters for the FanDuel simulation engine."""

    # File paths
    players_csv: Path = Path("data/players/FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv")
    template_csv: Path = Path("data/templates/FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv")
    output_csv: Path = Path("data/output/Completed-FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv")
    projections_csv: Optional[Path] = None

    # Simulation scale parameters
    num_candidates: int = 500        # Size of candidate pool generated via MILP (Stage 1)
    num_field_lineups: int = 10_000  # Number of opponent lineups in tournament field (Stage 2)
    num_sim_trials: int = 5_000      # Number of Monte Carlo game slate realizations (Stage 3)
    num_selected_lineups: int = 150  # Target portfolio size to export (Stage 4)

    # Contest financial parameters
    entry_fee: float = 0.05          # Entry fee per lineup ($0.05 default, CLI configurable)
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
    strict_exposure_caps: bool = False # If True, fail if candidate pool cannot fulfill K lineups under hard caps
    random_seed: int = 42            # Seed for reproducible Monte Carlo trials


@dataclass
class DraftKingsConfig:
    """Runtime configuration for DraftKings NFL Classic optimizer."""

    salaries_csv: Path = Path("data/players/DKSalaries.csv")
    entries_csv: Path = Path("data/templates/DKEntries.csv")
    output_csv: Path = Path("data/output/Completed-DKEntries.csv")
    num_lineups: int = 150
    salary_cap: int = 50_000
    stack_ratio: float = 0.80
    max_repeating_players: int = 6
    randomness_deviation: float = 0.25
    max_qb_exposure: float = 0.25
    max_rb_exposure: float = 0.25
    max_wr_exposure: float = 0.25
    max_te_exposure: float = 0.25
    max_dst_exposure: float = 0.20
    max_exposure: float = 0.25
    exclude_out_injured: bool = True
