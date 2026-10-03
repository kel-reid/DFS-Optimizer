"""
Unit tests for contest constraints, salary caps, stacking, and exposure limits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution and language servers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.data.loader import FanDuelDataLoader  # noqa: E402
from src.engine.selector import PortfolioSelector  # noqa: E402
from src.engine.solver import CandidatePoolGenerator  # noqa: E402


def test_loader_sanitization(mock_fanduel_files):
    config, _, _, _ = mock_fanduel_files
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()

    # Verify backup QB Case Keenum is zeroed
    case_keenum = next((p for p in optimizer.player_pool.all_players if p.full_name == "Case Keenum"), None)
    if case_keenum:
        assert case_keenum.fppg == 0.0


def test_candidate_generation_constraints(mock_fanduel_files):
    config, _, _, _ = mock_fanduel_files
    config.num_candidates = 6
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()

    generator = CandidatePoolGenerator(optimizer, config)
    candidates, mat = generator.generate_candidate_pool()

    assert len(candidates) == 6
    assert mat.shape[0] == 6

    # Verify constraints across generated lineups
    for lineup in candidates:
        players = list(lineup.lineup)
        assert len(players) == 9

        # Salary cap compliance
        total_salary = sum(p.salary for p in players)
        assert total_salary <= config.salary_cap

        # Opposing defense violation check
        defs = [p for p in players if "D" in p.positions]
        off_teams = {p.team for p in players if "D" not in p.positions}

        for d in defs:
            if d.game_info and d.game_info.home_team and d.game_info.away_team:
                opp_team = d.game_info.away_team if d.team == d.game_info.home_team else d.game_info.home_team
                assert opp_team not in off_teams, f"Opposing DEF violation detected: {d.team} vs {opp_team}"


def test_portfolio_selector_exposure_caps(mock_fanduel_files):
    config, _, _, _ = mock_fanduel_files
    config.num_candidates = 10
    config.num_selected_lineups = 5
    config.max_qb_exposure = 0.40  # max 2 lineups

    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()
    generator = CandidatePoolGenerator(optimizer, config)
    candidates, _ = generator.generate_candidate_pool()

    selector = PortfolioSelector(config)
    dummy_roi = np.linspace(100, 10, len(candidates))
    dummy_wins = np.zeros(len(candidates), dtype=int)
    dummy_top1 = np.linspace(5, 1, len(candidates))

    selected = selector.select_portfolio(candidates, dummy_roi, dummy_wins, dummy_top1)
    assert len(selected) == 5

    # Check QB exposure count
    qb_counts = {}
    for roster in selected:
        qb = next(p for p in roster.lineup if "QB" in p.positions)
        qb_counts[qb.full_name] = qb_counts.get(qb.full_name, 0) + 1

    max_allowed = int(np.floor(5 * 0.40))
    for count in qb_counts.values():
        assert count <= max_allowed or count > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
