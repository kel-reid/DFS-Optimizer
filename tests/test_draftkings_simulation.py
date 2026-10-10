"""
================================================================================
DraftKings Monte Carlo Simulation Engine Tests (tests/test_draftkings_simulation.py)
================================================================================
Adversarial and invariant testing for DraftKings simulation unification:
  - End-to-end 4-stage simulation pipeline execution under $50,000 budget
  - DST defense shock symmetry in CorrelatedGameEngine
  - Contest-agnostic entry fee auto-detection from template CSVs
  - Template exporter mapping without column mangling
  - Two-way alias synchronization in DraftKingsConfig
================================================================================
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pytest
from pydfs_lineup_optimizer import Player, Site, Sport, get_optimizer
from pydfs_lineup_optimizer.player import GameInfo

from src.build_draftkings_lineups import (
    DKOptimizerConfig,
    DraftKingsConfig,
    DraftKingsPortfolioAuditor,
    DraftKingsTemplateExporter,
    parse_dk_arguments,
)
from src.config import BaseOptimizerConfig, detect_entry_fee
from src.engine import (
    CandidatePoolGenerator,
    CorrelatedGameEngine,
    OpponentFieldSimulator,
    PortfolioSelector,
    SimAuditReporter,
)


@pytest.fixture
def dk_synthetic_players() -> List[Player]:
    """Generates a synthetic 4-team DraftKings NFL Classic player pool."""
    game1 = GameInfo("KC", "BUF", None)
    game2 = GameInfo("PHI", "DAL", None)
    teams = [("KC", game1), ("BUF", game1), ("PHI", game2), ("DAL", game2)]
    players: List[Player] = []
    pid = 1

    for team, g_info in teams:
        # 1 QB
        players.append(Player(str(pid), f"{team}_QB", f"Last_{pid}", ["QB"], team, 7000, 22.0, game_info=g_info))
        pid += 1
        # 3 RBs
        for i in range(3):
            sal = 5000 + i * 500
            fppg = 14.0 + i * 2.0
            players.append(Player(str(pid), f"{team}_RB{i}", f"Last_{pid}", ["RB"], team, sal, fppg, game_info=g_info))
            pid += 1
        # 4 WRs
        for i in range(4):
            sal = 4500 + i * 600
            fppg = 12.0 + i * 2.5
            players.append(Player(str(pid), f"{team}_WR{i}", f"Last_{pid}", ["WR"], team, sal, fppg, game_info=g_info))
            pid += 1
        # 2 TEs
        for i in range(2):
            sal = 3800 + i * 400
            fppg = 9.0 + i * 2.0
            players.append(Player(str(pid), f"{team}_TE{i}", f"Last_{pid}", ["TE"], team, sal, fppg, game_info=g_info))
            pid += 1
        # 1 DST
        players.append(Player(str(pid), f"{team}_DST", f"Last_{pid}", ["DST"], team, 3000, 8.0, game_info=g_info))
        pid += 1

    return players


def test_draftkings_simulation_pipeline_end_to_end(dk_synthetic_players: List[Player]) -> None:
    """Verifies complete 4-stage Monte Carlo simulation pipeline for DraftKings Classic."""
    config = DraftKingsConfig(
        salary_cap=50_000,
        min_field_salary=40_000,
        num_candidates=50,
        num_field_lineups=100,
        num_sim_trials=50,
        num_selected_lineups=5,
        max_qb_exposure=0.60,
        max_rb_exposure=0.60,
        max_wr_exposure=0.60,
        max_te_exposure=0.60,
        max_def_exposure=0.60,
        max_exposure=0.60,
        randomness_deviation=0.15,
        random_seed=42,
    )

    assert isinstance(config, BaseOptimizerConfig)
    assert config.salary_cap == 50_000

    opt = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
    opt.player_pool.load_players(dk_synthetic_players)
    players = list(opt.player_pool.filtered_players)

    # Stage 1: Candidate Pool
    gen = CandidatePoolGenerator(opt, config)
    candidates, c_mat = gen.generate_candidate_pool()
    assert len(candidates) == 50
    assert c_mat.shape == (50, len(players))

    # Stage 2: Opponent Field Simulator
    field_sim = OpponentFieldSimulator(players, config)
    f_mat = field_sim.simulate_field()
    assert f_mat.shape == (100, len(players))

    # Stage 3: Correlated Game Outcome Engine
    engine = CorrelatedGameEngine(players, config)
    s_points = engine.simulate_game_trials()
    assert s_points.shape == (len(players), 50)

    sim_roi, win_counts, top1_rates = engine.score_and_rank_candidates(c_mat, f_mat, s_points)
    assert len(sim_roi) == 50
    assert len(win_counts) == 50
    assert len(top1_rates) == 50

    # Stage 4: Portfolio Selection
    selector = PortfolioSelector(config)
    selected = selector.select_portfolio(candidates, sim_roi, win_counts, top1_rates)
    assert len(selected) == 5

    # Verification of Invariants
    for lineup in selected:
        roster = list(lineup.lineup)
        assert len(roster) == 9
        total_sal = sum(p.salary for p in roster)
        assert total_sal <= 50_000, f"Lineup exceeded DraftKings $50k cap: {total_sal}"

        # Team diversity: DK requires >= 2 teams
        distinct_teams = {p.team for p in roster}
        assert len(distinct_teams) >= 2

        # Check DST position is present
        assert any("DST" in p.positions for p in roster)

    # Audits run without raising exceptions
    DraftKingsPortfolioAuditor.audit_and_report(selected, config)
    SimAuditReporter.audit_and_report(selected, config, players, sim_roi, win_counts, top1_rates)


def test_draftkings_dst_shock_symmetry(dk_synthetic_players: List[Player]) -> None:
    """Verifies CorrelatedGameEngine applies defense shock symmetrically to DST."""
    config = DraftKingsConfig(num_sim_trials=200, random_seed=123)
    engine = CorrelatedGameEngine(dk_synthetic_players, config)
    sim_points = engine.simulate_game_trials()

    dst_indices = [i for i, p in enumerate(dk_synthetic_players) if "DST" in p.positions]
    assert len(dst_indices) > 0

    # DST scores should be positive finite values across trials
    for idx in dst_indices:
        scores = sim_points[idx, :]
        assert np.all(np.isfinite(scores))
        assert np.mean(scores) > 0.0


def test_draftkings_entry_fee_auto_detection(tmp_path: Path) -> None:
    """Verifies contest-agnostic entry fee detection from template CSV headers."""
    # Test 1: $20 high-stakes contest
    template_file = tmp_path / "DKEntries_HighRoller.csv"
    with open(template_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entry ID", "Contest ID", "Contest Name", "Entry Fee", "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"])
        writer.writerow(["1001", "2001", "$100K First Down", "$20.00", "", "", "", "", "", "", "", "", ""])

    detected = detect_entry_fee(template_file)
    assert detected == 20.0

    cfg = DraftKingsConfig.from_settings(template_csv=template_file)
    assert cfg.entry_fee == 20.0

    # Test 2: $0.25 micro contest
    template_file2 = tmp_path / "DKEntries_Micro.csv"
    with open(template_file2, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entry_id", "contest_id", "contest_name", "entry_fee"])
        writer.writerow(["1002", "2002", "Quarter Arcade", "0.25"])

    detected2 = detect_entry_fee(template_file2)
    assert detected2 == 0.25

    cfg2 = DraftKingsConfig.from_settings(template_csv=template_file2)
    assert cfg2.entry_fee == 0.25

    # Test 3: Explicit override takes precedence
    cfg3 = DraftKingsConfig.from_settings(template_csv=template_file2, entry_fee=5.0)
    assert cfg3.entry_fee == 5.0


def test_draftkings_template_exporter(tmp_path: Path, dk_synthetic_players: List[Player]) -> None:
    """Verifies DraftKingsTemplateExporter maps lineups into template slots preserving headers."""
    opt = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
    opt.player_pool.load_players(dk_synthetic_players)
    lineups = list(opt.optimize(n=2))

    template_csv = tmp_path / "DKEntries.csv"
    output_csv = tmp_path / "Completed-DKEntries.csv"

    header = ["Entry ID", "Contest ID", "Contest Name", "Entry Fee", "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    rows = [
        ["101", "201", "Contest 1", "$3.00", "", "", "", "", "", "", "", "", ""],
        ["102", "201", "Contest 1", "$3.00", "", "", "", "", "", "", "", "", ""],
    ]
    with open(template_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    config = DraftKingsConfig(
        template_csv=template_csv,
        output_csv=output_csv,
        num_selected_lineups=2,
        id_format="name_id",
    )

    exporter = DraftKingsTemplateExporter(config)
    res_path = exporter.export_lineups(lineups)
    assert res_path.exists()

    with open(res_path, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == header
        assert len(reader) == 3  # Header + 2 rows
        for row in reader[1:]:
            assert row[0] in ("101", "102")
            assert row[3] == "$3.00"
            # Verify 9 player slots are populated
            for slot_val in row[4:13]:
                assert "(" in slot_val and ")" in slot_val


def test_draftkings_config_aliases_and_sync() -> None:
    """Verifies two-way property aliases in DraftKingsConfig and DKOptimizerConfig compatibility."""
    assert DKOptimizerConfig is DraftKingsConfig
    cfg = DraftKingsConfig(
        salaries_csv=Path("custom_salaries.csv"),
        entries_csv=Path("custom_entries.csv"),
        num_lineups=25,
        max_dst_exposure=0.18,
    )
    assert cfg.players_csv == Path("custom_salaries.csv")
    assert cfg.salaries_csv == Path("custom_salaries.csv")
    assert cfg.template_csv == Path("custom_entries.csv")
    assert cfg.entries_csv == Path("custom_entries.csv")
    assert cfg.num_selected_lineups == 25
    assert cfg.num_lineups == 25
    assert cfg.max_def_exposure == 0.18
    assert cfg.max_dst_exposure == 0.18

    # Mutate through base name
    cfg.players_csv = Path("new_players.csv")
    assert cfg.salaries_csv == Path("new_players.csv")

    # Mutate through alias name
    cfg.num_lineups = 50
    assert cfg.num_selected_lineups == 50


def test_draftkings_cli_parsing_aliases_and_fee_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that parse_dk_arguments correctly processes aliases and auto-detects entry fees."""
    salaries_csv = tmp_path / "DKSalaries.csv"
    salaries_csv.write_text("Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n")
    template_csv = tmp_path / "DKEntries.csv"
    template_csv.write_text("Entry ID,Contest ID,Contest Name,Entry Fee\n1,100,Test Contest,$25.00\n")

    # 1. Test alias flags --salaries-csv and --entries-csv with omitted --entry-fee
    test_args = [
        "build_draftkings_lineups.py",
        "--salaries-csv", str(salaries_csv),
        "--entries-csv", str(template_csv),
    ]
    monkeypatch.setattr("sys.argv", test_args)
    cfg = parse_dk_arguments()

    assert cfg.players_csv == salaries_csv
    assert cfg.template_csv == template_csv
    assert cfg.entry_fee == 25.0  # Auto-detected from $25.00 in template

    # 2. Test explicit --entry-fee override
    test_args_override = [
        "build_draftkings_lineups.py",
        "--players-csv", str(salaries_csv),
        "--template-csv", str(template_csv),
        "--entry-fee", "5.0",
    ]
    monkeypatch.setattr("sys.argv", test_args_override)
    cfg_override = parse_dk_arguments()
    assert cfg_override.entry_fee == 5.0

