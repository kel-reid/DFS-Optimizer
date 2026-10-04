#!/usr/bin/env python3
"""
Root backward-compatibility wrapper for the FanDuel simulation pipeline.
Re-exports public classes, helpers, and orchestrator from src.build_fanduel_lineups.
"""
from src.build_fanduel_lineups import (
    CandidatePoolGenerator,
    CorrelatedGameEngine,
    FanDuelDataLoader,
    FanDuelTemplateExporter,
    OpponentFieldSimulator,
    PortfolioSelector,
    SimAuditReporter,
    SimOptimizerConfig,
    apply_forward_projections,
    find_players_csv,
    find_projections_csv,
    find_template_csv,
    main,
    normalize_name,
    parse_arguments,
    setup_logging,
)

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
    "normalize_name",
    "parse_arguments",
    "setup_logging",
]

if __name__ == "__main__":
    main()
