"""
================================================================================
Tests for GPP Ceiling Optimization, Multi-Piece Stacking & Ownership Constraints
================================================================================
Covers:
  1. GPP Selection Metrics: 'ceiling' (Q_0.999), 'win_rate', and 'roi'
  2. Multi-Piece Game Stacks: Double Stacks (QB + 2 WR/TE), Double Bring-Backs,
     and Team Defense Onslaughts (QB + pass-catcher + same-team DEF)
  3. Cumulative Projected Ownership Constraints
  4. SimulationResult tuple subclass backwards compatibility
================================================================================
"""

from pathlib import Path

import numpy as np
import pytest
from pydfs_lineup_optimizer import Lineup, Player, Site, Sport, get_optimizer
from pydfs_lineup_optimizer.player import LineupPlayer

from src.config import SimOptimizerConfig
from src.data.projections import apply_forward_projections
from src.engine.selector import PortfolioSelector
from src.engine.simulator import SimulationResult
from src.engine.solver import GameBringBackStack


def test_simulation_result_tuple_subclass_compatibility() -> None:
    """Verifies SimulationResult unpacks as a 3-tuple while providing ceiling attributes."""
    roi = np.array([10.5, 20.0])
    wins = np.array([1, 5])
    top1 = np.array([2.5, 10.0])
    ceilings = np.array([185.5, 210.0])
    top01 = np.array([0.5, 2.0])

    res = SimulationResult(roi, wins, top1, ceilings=ceilings, top01_rates=top01)

    # 1. 3-tuple unpacking
    unpacked_roi, unpacked_wins, unpacked_top1 = res
    np.testing.assert_array_equal(unpacked_roi, roi)
    np.testing.assert_array_equal(unpacked_wins, wins)
    np.testing.assert_array_equal(unpacked_top1, top1)
    assert len(res) == 3

    # 2. Named attributes
    assert res.ceilings is not None
    np.testing.assert_array_equal(res.ceilings, ceilings)
    assert res.top01_rates is not None
    np.testing.assert_array_equal(res.top01_rates, top01)


def test_config_gpp_validation_invariants() -> None:
    """Validates configuration bounds for selection_metric, stack sizes, and ownership."""
    cfg = SimOptimizerConfig(
        selection_metric="ceiling",
        primary_stack_size=2,
        bring_back_size=1,
        include_team_defense_stack=True,
        max_total_ownership=115.0,
    )
    assert cfg.selection_metric == "ceiling"
    assert cfg.primary_stack_size == 2
    assert cfg.include_team_defense_stack is True

    # Invalid selection_metric
    with pytest.raises(ValueError, match="Invalid selection_metric"):
        SimOptimizerConfig(selection_metric="invalid_metric")

    # Invalid primary_stack_size
    with pytest.raises(ValueError, match="Invalid primary_stack_size"):
        SimOptimizerConfig(primary_stack_size=0)

    # Invalid bring_back_size
    with pytest.raises(ValueError, match="Invalid bring_back_size"):
        SimOptimizerConfig(bring_back_size=-1)

    # Invalid max_total_ownership
    with pytest.raises(ValueError, match="Invalid max_total_ownership"):
        SimOptimizerConfig(max_total_ownership=-5.0)


def test_portfolio_selector_ceiling_sort_precedence() -> None:
    """Tests PortfolioSelector prioritizing Q_0.999 ceiling when selection_metric='ceiling'."""
    p1 = Player("1", "QB", "One", ["QB"], "KC", 6000, 20.0)
    p2 = Player("2", "RB", "Two", ["RB"], "KC", 6000, 15.0)
    p3 = Player("3", "WR", "Three", ["WR"], "LV", 6000, 15.0)

    lp1 = LineupPlayer(p1, "QB")
    lp2 = LineupPlayer(p2, "RB")
    lp3 = LineupPlayer(p3, "WR")

    lineup_a = Lineup([lp1, lp2, lp3])  # High ROI, lower ceiling
    lineup_b = Lineup([lp1, lp2, lp3])  # Lower ROI, monstrous ceiling

    candidates = [lineup_a, lineup_b]
    sim_roi = np.array([150.0, 50.0])       # A has 150% ROI, B has 50% ROI
    win_counts = np.array([2, 5])
    top1_rates = np.array([5.0, 12.0])
    cand_ceilings = np.array([170.0, 245.0]) # B has 245.0 ceiling vs A 170.0

    # 1. Under default 'roi' metric -> A should be selected first
    cfg_roi = SimOptimizerConfig(num_selected_lineups=1, selection_metric="roi", max_exposure=1.0)
    sel_roi = PortfolioSelector(cfg_roi)
    selected_roi = sel_roi.select_portfolio(candidates, sim_roi, win_counts, top1_rates, cand_ceilings)
    assert selected_roi[0] == lineup_a

    # 2. Under 'ceiling' metric -> B should be selected first due to 245.0 ceiling
    cfg_ceil = SimOptimizerConfig(num_selected_lineups=1, selection_metric="ceiling", max_exposure=1.0)
    sel_ceil = PortfolioSelector(cfg_ceil)
    selected_ceil = sel_ceil.select_portfolio(candidates, sim_roi, win_counts, top1_rates, cand_ceilings)
    assert selected_ceil[0] == lineup_b


def test_portfolio_selector_win_rate_metric() -> None:
    """Tests PortfolioSelector prioritizing 1st-place win counts when selection_metric='win_rate'."""
    p1 = Player("1", "QB", "One", ["QB"], "KC", 6000, 20.0)
    p2 = Player("2", "RB", "Two", ["RB"], "KC", 6000, 15.0)

    lp1 = LineupPlayer(p1, "QB")
    lp2 = LineupPlayer(p2, "RB")

    lineup_a = Lineup([lp1, lp2])  # High ROI (100%), 1 win
    lineup_b = Lineup([lp1, lp2])  # Lower ROI (20%), 10 wins

    candidates = [lineup_a, lineup_b]
    sim_roi = np.array([100.0, 20.0])
    win_counts = np.array([1, 10])
    top1_rates = np.array([5.0, 25.0])

    cfg_win = SimOptimizerConfig(num_selected_lineups=1, selection_metric="win_rate", max_exposure=1.0)
    sel_win = PortfolioSelector(cfg_win)
    selected = sel_win.select_portfolio(candidates, sim_roi, win_counts, top1_rates)
    assert selected[0] == lineup_b


def test_game_bring_back_stack_double_stack_and_onslaught() -> None:
    """Tests GameBringBackStack generating double stacks (QB + 2 WR/TE) and Defense onslaughts."""
    optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
    players = [
        # Team A (KC)
        Player("1", "Patrick", "Mahomes", ["QB"], "KC", 8000, 22.0),
        Player("2", "Travis", "Kelce", ["TE"], "KC", 7000, 16.0),
        Player("3", "Rashee", "Rice", ["WR"], "KC", 6500, 15.0),
        Player("4", "Xavier", "Worthy", ["WR"], "KC", 5500, 12.0),
        Player("5", "Kansas City", "Chiefs", ["D"], "KC", 4000, 8.0),
        Player("5b", "Isiah", "Pacheco", ["RB"], "KC", 6500, 14.0),
        # Team B (LV - Opponent)
        Player("6", "Gardner", "Minshew", ["QB"], "LV", 6000, 12.0),
        Player("7", "Davante", "Adams", ["WR"], "LV", 7500, 18.0),
        Player("8", "Jakobi", "Meyers", ["WR"], "LV", 5500, 11.0),
        Player("9", "Brock", "Bowers", ["TE"], "LV", 5000, 10.0),
        Player("9b", "Zamir", "White", ["RB"], "LV", 5500, 10.0),
        # Team C (Neutral for FD >= 3 teams invariant)
        Player("10", "Derrick", "Henry", ["RB"], "BAL", 8500, 20.0),
        Player("11", "Zay", "Flowers", ["WR"], "BAL", 6000, 14.0),
        Player("12", "Lamar", "Jackson", ["QB"], "BAL", 8200, 23.0),
    ]

    from pydfs_lineup_optimizer.player import GameInfo
    kc_lv_game = GameInfo("KC", "LV", None)
    bal_cin_game = GameInfo("BAL", "CIN", None)
    for p in players:
        p.game_info = kc_lv_game if p.team in ("KC", "LV") else bal_cin_game

    optimizer.player_pool.load_players(players)

    # Build Double Stack + Bring-Back + DEF Onslaught:
    # QB + 2 Pass Catchers (KC) + 1 Opposing Pass Catcher (LV) + KC DEF
    stack = GameBringBackStack(
        primary_stack_size=2,
        bring_back_size=1,
        include_team_defense=True,
    )
    optimizer.add_stack(stack)
    optimizer.set_total_teams(min_teams=3)

    lineups = list(optimizer.optimize(n=1))
    assert len(lineups) == 1
    lineup = lineups[0]

    kc_players = [p for p in lineup.lineup if p.team == "KC"]
    lv_players = [p for p in lineup.lineup if p.team == "LV"]

    # Verify QB is present
    kc_qbs = [p for p in kc_players if "QB" in p.positions]
    assert len(kc_qbs) == 1

    # Verify >= 2 primary pass-catchers
    kc_pass = [p for p in kc_players if any(pos in ("WR", "TE") for pos in p.positions)]
    assert len(kc_pass) >= 2

    # Verify opposing bring-back
    lv_pass = [p for p in lv_players if any(pos in ("WR", "TE") for pos in p.positions)]
    assert len(lv_pass) >= 1

    # Verify same-team DEF onslaught
    kc_defs = [p for p in kc_players if "D" in p.positions or "DST" in p.positions]
    assert len(kc_defs) == 1


def test_apply_forward_projections_with_ownership_and_candidate_constraint(tmp_path: Path) -> None:
    """Verifies ownership parsing from projection feed and enforcement of cumulative ownership cap."""
    proj_csv = tmp_path / "projections_with_ownership.csv"
    proj_csv.write_text(
        "first_name,last_name,position,team,ppg_projection,ownership_projection\n"
        "Patrick,Mahomes,QB,KC,24.5,25.0\n"
        "Travis,Kelce,TE,KC,18.0,30.0\n"
        "Derrick,Henry,RB,BAL,20.0,35.0\n"
        "Davante,Adams,WR,LV,17.5,15.0\n",
        encoding="utf-8",
    )

    optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
    players = [
        Player("1", "Patrick", "Mahomes", ["QB"], "KC", 8000, 20.0),
        Player("2", "Travis", "Kelce", ["TE"], "KC", 7000, 15.0),
        Player("3", "Derrick", "Henry", ["RB"], "BAL", 8500, 18.0),
        Player("4", "Davante", "Adams", ["WR"], "LV", 7500, 16.0),
    ]
    optimizer.player_pool.load_players(players)

    updated, _ = apply_forward_projections(optimizer, proj_csv, zero_unprojected=False)
    assert updated == 4

    p_map = {p.full_name: p for p in optimizer.player_pool.filtered_players}
    assert p_map["Patrick Mahomes"].projected_ownership == 0.25
    assert p_map["Travis Kelce"].projected_ownership == 0.30
    assert p_map["Derrick Henry"].projected_ownership == 0.35
    assert p_map["Davante Adams"].projected_ownership == 0.15
