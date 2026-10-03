"""
Unit tests for simulation engine, opponent field sampler, and template export.
"""

import numpy as np

from src.data.exporter import FanDuelTemplateExporter
from src.data.loader import FanDuelDataLoader
from src.engine.field import OpponentFieldSimulator
from src.engine.simulator import CorrelatedGameEngine
from src.engine.solver import CandidatePoolGenerator


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
    candidates, c_mat = cand_gen.generate_candidate_pool()

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
    config, _, template_csv, output_csv = mock_fanduel_files
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
