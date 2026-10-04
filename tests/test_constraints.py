"""
Unit tests for contest constraints, salary caps, stacking, and exposure limits.
"""

from __future__ import annotations

import sys
from collections import Counter
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

        # Team diversity: at least 3 distinct teams and max 4 players from one team
        teams = [p.team for p in players]
        distinct_teams = set(teams)
        assert len(distinct_teams) >= 3, f"Lineup has only {len(distinct_teams)} distinct teams (< 3): {teams}"
        team_counts = Counter(teams)
        assert max(team_counts.values()) <= 4, f"Lineup exceeds 4 players from a single team: {team_counts}"


def test_portfolio_selector_exposure_caps(mock_fanduel_files):
    config, _, _, _ = mock_fanduel_files
    config.num_candidates = 30
    config.num_selected_lineups = 5
    config.max_qb_exposure = 0.40  # max 2 lineups
    config.max_rb_exposure = 0.60
    config.max_wr_exposure = 0.60
    config.max_te_exposure = 0.60
    config.max_def_exposure = 0.60
    config.max_exposure = 0.60

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
        assert count <= max_allowed


def test_draftkings_team_diversity_constraints():
    from pydfs_lineup_optimizer import Player, Site, Sport, get_optimizer
    from pydfs_lineup_optimizer.player import GameInfo

    from src.build_draftkings_lineups import DKOptimizerConfig, DraftKingsLineupPipeline

    opt = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
    game1 = GameInfo("KC", "BUF", None)
    game2 = GameInfo("PHI", "DAL", None)
    teams = [("KC", game1), ("BUF", game1), ("PHI", game2), ("DAL", game2)]
    players = []
    pid = 1
    for team, g_info in teams:
        for pos in ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST"]:
            players.append(Player(str(pid), f"{team}_{pos}", f"Last_{pid}", [pos], team, 5000, 15.0, game_info=g_info))
            pid += 1

    opt.player_pool.load_players(players)
    config = DKOptimizerConfig()
    pipeline = DraftKingsLineupPipeline(config, opt)
    pipeline.configure_optimizer_instance(opt)

    lineups = list(opt.optimize(n=3))
    assert len(lineups) == 3
    for lineup in lineups:
        roster = list(lineup.lineup)
        roster_teams = [p.team for p in roster]
        distinct = set(roster_teams)
        assert len(distinct) >= 3, f"DK lineup has {len(distinct)} teams (< 3): {roster_teams}"
        team_counts = Counter(roster_teams)
        assert max(team_counts.values()) <= 4, f"DK lineup exceeds 4 players from one team: {team_counts}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
