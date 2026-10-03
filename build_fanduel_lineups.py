#!/usr/bin/env python3
"""
================================================================================
FanDuel NFL DFS Optimization Pipeline (Compatibility Forwarder)
================================================================================
Forwards execution to src.build_fanduel_lineups for centralized maintenance.
================================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.build_fanduel_lineups import (
    SimOptimizerConfig,
    FanDuelDataLoader,
    CandidatePoolGenerator,
    OpponentFieldSimulator,
    CorrelatedGameEngine,
    PortfolioSelector,
    FanDuelTemplateExporter,
    SimAuditReporter,
    parse_arguments,
    main,
)

if __name__ == "__main__":
    main()
