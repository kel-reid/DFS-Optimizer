#!/usr/bin/env python3
"""
================================================================================
FanDuel NFL Classic Quantitative Monte Carlo Simulation & Optimization Engine
================================================================================
Architected as a multi-stage quantitative pipeline transitioning from basic
Mixed-Integer Linear Programming (MILP) to a high-dimensional vectorized
Monte Carlo Game Slate and Contest Tournament Simulation Engine.

This entry module acts as a facade orchestrating:
  - Configuration: src.config.SimOptimizerConfig
  - Data Ingestion & Sanitization: src.data.FanDuelDataLoader
  - Candidate Pool Generation (MILP): src.engine.CandidatePoolGenerator
  - Opponent Field Simulation: src.engine.OpponentFieldSimulator
  - Correlated Slate Simulation & GPP Ranking: src.engine.CorrelatedGameEngine
  - Portfolio Selection & Risk Auditing: src.engine.PortfolioSelector
  - Contest Upload Export: src.data.FanDuelTemplateExporter
================================================================================
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Re-export core classes for backward compatibility
from src.config import SimOptimizerConfig
from src.data import (
    FanDuelDataLoader,
    FanDuelTemplateExporter,
    apply_forward_projections,
    find_players_csv,
    find_projections_csv,
    find_template_csv,
    normalize_name,
    normalize_week,
)
from src.engine import (
    CandidatePoolGenerator,
    CorrelatedGameEngine,
    OpponentFieldSimulator,
    PortfolioSelector,
    SimAuditReporter,
)

logger = logging.getLogger("FanDuelSimOptimizer")

__all__ = [
    "CandidatePoolGenerator",
    "CorrelatedGameEngine",
    "FanDuelDataLoader",
    "FanDuelTemplateExporter",
    "OpponentFieldSimulator",
    "PortfolioSelector",
    "SimAuditReporter",
    "SimOptimizerConfig",
    "apply_forward_projections",
    "find_players_csv",
    "find_projections_csv",
    "find_template_csv",
    "main",
    "normalize_week",
    "normalize_name",
    "parse_arguments",
    "setup_logging",
]


def parse_arguments() -> SimOptimizerConfig:
    """Parses CLI arguments into a strongly-typed SimOptimizerConfig instance, merging config/settings.yaml."""
    default_cfg = SimOptimizerConfig.from_settings()

    parser = argparse.ArgumentParser(
        description="FanDuel NFL Classic Quantitative Monte Carlo Simulation & Optimization Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--site", type=str, default="fanduel", help="DFS contest platform.")
    parser.add_argument("--players-csv", type=Path, default=None)
    parser.add_argument("--template-csv", type=Path, default=None)
    parser.add_argument(
        "--projections-csv",
        type=Path,
        default=None,
        help="Path to external projections CSV (e.g. data/projections/fanduel_research_projections.csv).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=default_cfg.num_candidates)
    parser.add_argument("--num-field", type=int, default=default_cfg.num_field_lineups)
    parser.add_argument("--num-trials", type=int, default=default_cfg.num_sim_trials)
    parser.add_argument("--num-lineups", type=int, default=default_cfg.num_selected_lineups)
    parser.add_argument(
        "--entry-fee",
        type=float,
        default=None,
        help="Contest entry fee in dollars (default: auto-detect from template).",
    )
    parser.add_argument("--stack-ratio", type=float, default=default_cfg.stack_ratio)
    parser.add_argument("--max-qb-exposure", type=float, default=default_cfg.max_qb_exposure)
    parser.add_argument("--max-rb-exposure", type=float, default=default_cfg.max_rb_exposure)
    parser.add_argument("--max-wr-exposure", type=float, default=default_cfg.max_wr_exposure)
    parser.add_argument("--max-te-exposure", type=float, default=default_cfg.max_te_exposure)
    parser.add_argument("--max-def-exposure", type=float, default=default_cfg.max_def_exposure)
    parser.add_argument("--max-exposure", type=float, default=default_cfg.max_exposure)
    parser.add_argument(
        "--max-repeating",
        type=int,
        default=default_cfg.max_repeating_players,
        help="Enforces >= 3 unique players between every pair of lineups",
    )
    parser.add_argument("--randomness", type=float, default=default_cfg.randomness_deviation)
    parser.add_argument("--keep-injured", action="store_true", default=False)
    parser.add_argument("--zero-unprojected", action="store_true", default=default_cfg.zero_unprojected, help="Zero out unprojected / N/A players (SaberSim standard)")
    parser.add_argument("--keep-unprojected", action="store_true", default=False, help="Retain baseline FPPG for unprojected players")
    parser.add_argument(
        "--strict-caps",
        action="store_true",
        default=default_cfg.strict_exposure_caps,
        help="Strictly enforce exposure caps; fail if candidate pool cannot fulfill K lineups without cap overage",
    )
    parser.add_argument("--slate", type=str, default=default_cfg.slate, help="Target slate identifier (e.g. sunday-night, main-slate).")
    parser.add_argument("--week", type=str, default=default_cfg.week, help="NFL Week (e.g. 5, 05, week-05).")
    parser.add_argument("--date", type=str, default=default_cfg.slate_date, help="Slate date in YYYY-MM-DD format (e.g. 2026-10-04).")
    parser.add_argument("--single-game", action="store_true", default=default_cfg.is_single_game, help="Force Single Game (Showdown) optimizer format.")

    args = parser.parse_args()

    norm_week = normalize_week(args.week)

    players_path = find_players_csv(args.players_csv, slate=args.slate, week=norm_week, slate_date=args.date)
    template_path = find_template_csv(args.template_csv, slate=args.slate, week=norm_week, slate_date=args.date)
    projections_path = find_projections_csv(args.projections_csv, slate=args.slate, week=norm_week, slate_date=args.date)

    if args.output_csv:
        output_path = args.output_csv
    elif args.slate:
        w_str = norm_week or "week-05"
        slate_dir = Path(f"data/{w_str}/{args.slate}")
        if not slate_dir.is_dir() and args.date:
            date_dir = Path(f"data/{args.date}/{args.slate}")
            if date_dir.is_dir():
                slate_dir = date_dir
        if slate_dir.is_dir():
            output_path = slate_dir / "completed_lineups.csv"
        else:
            output_dir = Path(f"data/output/{args.slate}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"Completed-{template_path.name}"
    else:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"Completed-{template_path.name}"

    exclude_injured = False if args.keep_injured else default_cfg.exclude_out_injured
    zero_unproj = False if args.keep_unprojected else default_cfg.zero_unprojected

    return SimOptimizerConfig.from_settings(
        players_csv=players_path,
        template_csv=template_path,
        output_csv=output_path,
        projections_csv=projections_path,
        num_candidates=args.num_candidates,
        num_field_lineups=args.num_field,
        num_sim_trials=args.num_trials,
        num_selected_lineups=args.num_lineups,
        entry_fee=args.entry_fee,
        stack_ratio=args.stack_ratio,
        max_qb_exposure=args.max_qb_exposure,
        max_rb_exposure=args.max_rb_exposure,
        max_wr_exposure=args.max_wr_exposure,
        max_te_exposure=args.max_te_exposure,
        max_def_exposure=args.max_def_exposure,
        max_exposure=args.max_exposure,
        max_repeating_players=args.max_repeating,
        randomness_deviation=args.randomness,
        exclude_out_injured=exclude_injured,
        strict_exposure_caps=args.strict_caps,
        zero_unprojected=zero_unproj,
        slate=args.slate,
        week=norm_week,
        slate_date=args.date,
        is_single_game=args.single_game,
    )


def setup_logging() -> None:
    """Configures structured console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Main execution orchestrator for the FanDuel simulation pipeline."""
    print("[INFO] Initializing FanDuel Football Optimizer...")
    setup_logging()
    config = parse_arguments()

    # 1. Ingest Data & Filter Backup QBs / Apply Forward Projections
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()
    players = list(optimizer.player_pool.filtered_players)

    # 2. Stage 1: Generate Candidate Pool of Lineups (MILP)
    candidate_generator = CandidatePoolGenerator(optimizer, config)
    candidates, candidates_matrix = candidate_generator.generate_candidate_pool()

    # 3. Stage 2: Simulate Opponent Field Lineups
    field_simulator = OpponentFieldSimulator(players, config)
    field_matrix = field_simulator.simulate_field()

    # 4. Stage 3: Correlated Game Outcome Engine & Vectorized Scoring
    game_engine = CorrelatedGameEngine(players, config)
    sim_points = game_engine.simulate_game_trials()

    sim_roi, win_counts, top1_rates = game_engine.score_and_rank_candidates(
        candidates_matrix, field_matrix, sim_points
    )

    # 5. Stage 4: Portfolio Selection (Exposure Ceilings)
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
