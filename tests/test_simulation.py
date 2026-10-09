"""
Unit tests for simulation engine, opponent field sampler, and template export.
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

from src.data.exporter import FanDuelTemplateExporter  # noqa: E402
from src.data.loader import FanDuelDataLoader  # noqa: E402
from src.engine.field import OpponentFieldSimulator  # noqa: E402
from src.engine.simulator import CorrelatedGameEngine  # noqa: E402
from src.engine.solver import CandidatePoolGenerator  # noqa: E402


def test_opponent_field_simulation(mock_fanduel_files):
    config, _, _, _ = mock_fanduel_files
    config.num_field_lineups = 50
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()
    players = list(optimizer.player_pool.all_players)

    field_sim = OpponentFieldSimulator(players, config)
    field_mat = field_sim.simulate_field()

    assert field_mat.shape == (50, len(players))
    # Each row must sum to 9 (9 rostered players)
    row_sums = np.sum(field_mat, axis=1)
    assert np.all(row_sums == 9.0)


def test_correlated_game_engine(mock_fanduel_files):
    config, _, _, _ = mock_fanduel_files
    config.num_candidates = 5
    config.num_field_lineups = 20
    config.num_sim_trials = 25

    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()
    players = list(optimizer.player_pool.all_players)

    # Candidates & field
    cand_gen = CandidatePoolGenerator(optimizer, config)
    _, c_mat = cand_gen.generate_candidate_pool()

    field_sim = OpponentFieldSimulator(players, config)
    f_mat = field_sim.simulate_field()

    engine = CorrelatedGameEngine(players, config)
    sim_points = engine.simulate_game_trials()

    assert sim_points.shape == (len(players), 25)
    # Ensure all simulated points are non-negative
    assert np.all(sim_points >= 0.0)

    sim_roi, win_counts, top1_rates = engine.score_and_rank_candidates(c_mat, f_mat, sim_points)

    assert len(sim_roi) == 5
    assert len(win_counts) == 5
    assert len(top1_rates) == 5
    assert isinstance(float(sim_roi[0]), float)


def test_template_exporter(mock_fanduel_files):
    config, _, _, output_csv = mock_fanduel_files
    config.num_candidates = 5
    config.num_selected_lineups = 5

    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()
    cand_gen = CandidatePoolGenerator(optimizer, config)
    candidates, _ = cand_gen.generate_candidate_pool()

    exporter = FanDuelTemplateExporter(config)
    out_path = exporter.export_lineups(candidates[:5])

    assert out_path.exists()
    assert out_path == output_csv

    # Verify line count (header + 5 lineups)
    lines = [line.strip() for line in out_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 6


def test_single_game_simulation_and_mvp_weighting(tmp_path: Path):
    """
    Verifies that Single Game / Showdown slates:
      1. Correctly apply 1.5x score weighting for MVP players in candidate matrix.
      2. Correctly sample 5-player Single Game rosters with 1.5x MVP in opponent field matrix.
      3. Maintain shape and mathematical consistency through BLAS scoring and portfolio selection.
    """
    import csv

    from src.config import SimOptimizerConfig
    from src.engine.selector import PortfolioSelector

    sg_dir = tmp_path / "data" / "week-05" / "sunday-night"
    sg_dir.mkdir(parents=True)
    players_csv = sg_dir / "players.csv"
    template_csv = sg_dir / "entries_template.csv"
    output_csv = sg_dir / "completed.csv"

    fieldnames = [
        "Id", "Position", "First Name", "Nickname", "Last Name",
        "FPPG", "Team", "Opponent", "Game", "Injury Indicator",
        "Injury Details", "Tier", "Probable Pitcher", "Batting Order",
        "Roster Position", "Salary"
    ]
    rows = [
        {"Id": "101", "Position": "QB", "First Name": "Jared", "Nickname": "Jared Goff", "Last Name": "Goff", "FPPG": "20.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "15000"},
        {"Id": "102", "Position": "RB", "First Name": "Jahmyr", "Nickname": "Jahmyr Gibbs", "Last Name": "Gibbs", "FPPG": "19.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "14500"},
        {"Id": "103", "Position": "WR", "First Name": "Amon-Ra", "Nickname": "Amon-Ra St. Brown", "Last Name": "St. Brown", "FPPG": "18.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "14000"},
        {"Id": "104", "Position": "QB", "First Name": "Bryce", "Nickname": "Bryce Young", "Last Name": "Young", "FPPG": "17.0", "Team": "CAR", "Opponent": "DET", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "13000"},
        {"Id": "105", "Position": "RB", "First Name": "Chuba", "Nickname": "Chuba Hubbard", "Last Name": "Hubbard", "FPPG": "15.0", "Team": "CAR", "Opponent": "DET", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "12000"},
        {"Id": "106", "Position": "WR", "First Name": "Tetairoa", "Nickname": "Tetairoa McMillan", "Last Name": "McMillan", "FPPG": "13.0", "Team": "CAR", "Opponent": "DET", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "10000"},
        {"Id": "107", "Position": "TE", "First Name": "Sam", "Nickname": "Sam LaPorta", "Last Name": "LaPorta", "FPPG": "11.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "9000"},
        {"Id": "108", "Position": "K", "First Name": "Jake", "Nickname": "Jake Bates", "Last Name": "Bates", "FPPG": "8.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP - 1.5X Points/AnyFLEX", "Salary": "8000"},
    ]
    with open(players_csv, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(rows)

    with open(template_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["entry_id", "contest_id", "contest_name", "entry_fee", "MVP - 1.5X Points", "AnyFLEX", "AnyFLEX", "AnyFLEX", "AnyFLEX"])
        csv_writer.writerow(["1", "99", "Single Game", "$0.05", "", "", "", "", ""])
        csv_writer.writerow(["2", "99", "Single Game", "$0.05", "", "", "", "", ""])

    config = SimOptimizerConfig(
        players_csv=players_csv,
        template_csv=template_csv,
        output_csv=output_csv,
        num_candidates=4,
        num_field_lineups=20,
        num_sim_trials=30,
        num_selected_lineups=2,
        is_single_game=True,
    )

    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()
    assert config.is_single_game is True
    assert optimizer.settings.get_total_players() == 5

    # 1. Candidate matrix generation
    cand_gen = CandidatePoolGenerator(optimizer, config)
    candidates, c_mat = cand_gen.generate_candidate_pool()
    assert len(candidates) == 4
    assert c_mat.shape == (4, len(optimizer.player_pool.filtered_players))
    # Each candidate row must sum to exactly 5.5 (1 MVP @ 1.5 + 4 AnyFLEX @ 1.0)
    c_row_sums = np.sum(c_mat, axis=1)
    np.testing.assert_allclose(c_row_sums, 5.5)

    # 2. Opponent field simulation
    players = list(optimizer.player_pool.filtered_players)
    field_sim = OpponentFieldSimulator(players, config)
    assert field_sim.is_single_game is True
    f_mat = field_sim.simulate_field()
    assert f_mat.shape == (20, len(players))
    # Each field row must sum to exactly 5.5 (1 MVP @ 1.5 + 4 AnyFLEX @ 1.0)
    f_row_sums = np.sum(f_mat, axis=1)
    np.testing.assert_allclose(f_row_sums, 5.5)

    # 3. Vectorized simulation scoring
    engine = CorrelatedGameEngine(players, config)
    sim_points = engine.simulate_game_trials()
    assert sim_points.shape == (len(players), 30)

    sim_roi, win_counts, top1_rates = engine.score_and_rank_candidates(c_mat, f_mat, sim_points)
    assert len(sim_roi) == 4
    assert len(win_counts) == 4
    assert len(top1_rates) == 4

    # 4. Portfolio selection under Single Game exposures
    selector = PortfolioSelector(config)
    selected = selector.select_portfolio(candidates, sim_roi, win_counts, top1_rates)
    assert len(selected) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
