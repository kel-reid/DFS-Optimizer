"""
Unit and integration tests for Slate-First architecture and Single Game detection across NFL weeks.
"""

import csv
from pathlib import Path

import pytest
from pydfs_lineup_optimizer import Site

from src.config import SimOptimizerConfig
from src.data.exporter import FanDuelTemplateExporter
from src.data.loader import (
    FanDuelDataLoader,
    find_players_csv,
    find_template_csv,
    normalize_week,
)
from src.data.projections import find_projections_csv


@pytest.fixture(autouse=True)
def setup_mock_slate_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Sets up isolated mock data/week-05 directory tree in tmp_path so slate loader tests
    run deterministically and independently of repository data files.
    """
    monkeypatch.chdir(tmp_path)

    # 1. sunday-night (Single Game format)
    sn_dir = tmp_path / "data" / "week-05" / "sunday-night"
    sn_dir.mkdir(parents=True)

    sn_players = sn_dir / "players.csv"
    fieldnames = [
        "Id", "Position", "First Name", "Nickname", "Last Name",
        "FPPG", "Team", "Opponent", "Game", "Injury Indicator",
        "Injury Details", "Tier", "Probable Pitcher", "Batting Order",
        "Roster Position", "Salary"
    ]
    sn_rows = [
        {"Id": "1001", "Position": "RB", "First Name": "Jahmyr", "Nickname": "Jahmyr Gibbs", "Last Name": "Gibbs", "FPPG": "24.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP/AnyFLEX", "Salary": "15000"},
        {"Id": "1002", "Position": "QB", "First Name": "Jared", "Nickname": "Jared Goff", "Last Name": "Goff", "FPPG": "18.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP/AnyFLEX", "Salary": "14000"},
        {"Id": "1003", "Position": "WR", "First Name": "Amon-Ra", "Nickname": "Amon-Ra St. Brown", "Last Name": "St. Brown", "FPPG": "17.0", "Team": "DET", "Opponent": "CAR", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP/AnyFLEX", "Salary": "13500"},
        {"Id": "1004", "Position": "QB", "First Name": "Bryce", "Nickname": "Bryce Young", "Last Name": "Young", "FPPG": "15.0", "Team": "CAR", "Opponent": "DET", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP/AnyFLEX", "Salary": "12000"},
        {"Id": "1005", "Position": "RB", "First Name": "Chuba", "Nickname": "Chuba Hubbard", "Last Name": "Hubbard", "FPPG": "14.0", "Team": "CAR", "Opponent": "DET", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP/AnyFLEX", "Salary": "11000"},
        {"Id": "1006", "Position": "WR", "First Name": "Tetairoa", "Nickname": "Tetairoa McMillan", "Last Name": "McMillan", "FPPG": "12.0", "Team": "CAR", "Opponent": "DET", "Game": "DET@CAR", "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "", "Batting Order": "", "Roster Position": "MVP/AnyFLEX", "Salary": "9000"},
    ]
    with open(sn_players, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sn_rows)

    sn_template = sn_dir / "entries_template.csv"
    tmpl_headers = ["entry_id", "contest_id", "contest_name", "entry_fee", "MVP - 1.5X Points", "AnyFLEX", "AnyFLEX", "AnyFLEX", "AnyFLEX"]
    with open(sn_template, "w", newline="", encoding="utf-8") as f:
        tmpl_writer = csv.writer(f)
        tmpl_writer.writerow(tmpl_headers)
        tmpl_writer.writerow(["12345", "9999", "NFL Single Game", "$0.05", "", "", "", "", ""])
        tmpl_writer.writerow(["12346", "9999", "NFL Single Game", "$0.05", "", "", "", "", ""])

    sn_proj = sn_dir / "projections.csv"
    with open(sn_proj, "w", newline="", encoding="utf-8") as f:
        proj_writer = csv.writer(f)
        proj_writer.writerow(["player", "team", "pos", "fantasy"])
        proj_writer.writerow(["Jahmyr Gibbs", "DET", "RB", "24.0"])
        proj_writer.writerow(["Jared Goff", "DET", "QB", "18.0"])
        proj_writer.writerow(["Amon-Ra St. Brown", "DET", "WR", "17.0"])
        proj_writer.writerow(["Bryce Young", "CAR", "QB", "15.0"])
        proj_writer.writerow(["Chuba Hubbard", "CAR", "RB", "14.0"])
        proj_writer.writerow(["Tetairoa McMillan", "CAR", "WR", "12.0"])

    # 2. main-slate
    main_dir = tmp_path / "data" / "week-05" / "main-slate"
    main_dir.mkdir(parents=True)
    main_players = main_dir / "players.csv"
    main_players.write_text("Id,Position,First Name,Nickname,Last Name,FPPG,Team,Opponent,Game,Injury Indicator,Injury Details,Tier,Probable Pitcher,Batting Order,Roster Position,Salary\n")
    main_proj = main_dir / "projections.csv"
    main_proj.write_text("player,team,pos,fantasy\n")

    # 3. monday-night
    mn_dir = tmp_path / "data" / "week-05" / "monday-night"
    mn_dir.mkdir(parents=True)
    mn_players = mn_dir / "players.csv"
    mn_players.write_text("Id,Position,First Name,Nickname,Last Name,FPPG,Team,Opponent,Game,Injury Indicator,Injury Details,Tier,Probable Pitcher,Batting Order,Roster Position,Salary\n")


def test_normalize_week():
    assert normalize_week(5) == "week-05"
    assert normalize_week("5") == "week-05"
    assert normalize_week("05") == "week-05"
    assert normalize_week("week-5") == "week-05"
    assert normalize_week("week-05") == "week-05"
    assert normalize_week("week-18") == "week-18"
    assert normalize_week(None) is None
    assert normalize_week("") is None


def test_slate_file_discovery_with_week():
    # Test with string '5'
    players_p = find_players_csv(slate="sunday-night", week="5")
    assert players_p.exists()
    assert "week-05" in str(players_p)
    assert "sunday-night" in str(players_p)

    template_p = find_template_csv(slate="sunday-night", week=5)
    assert template_p.exists()
    assert "week-05" in str(template_p)
    assert "sunday-night" in str(template_p)

    proj_p = find_projections_csv(slate="sunday-night", week="week-05")
    assert proj_p is not None and proj_p.exists()
    assert "week-05" in str(proj_p)
    assert "sunday-night" in str(proj_p)


def test_slate_file_discovery_main_slate():
    players_p = find_players_csv(slate="main-slate", week="week-05")
    assert players_p.exists()
    assert "week-05" in str(players_p)
    assert "main-slate" in str(players_p)

    proj_p = find_projections_csv(slate="main-slate", week=5)
    assert proj_p is not None and proj_p.exists()
    assert "week-05" in str(proj_p)
    assert "main-slate" in str(proj_p)


def test_slate_file_discovery_monday_night():
    players_p = find_players_csv(slate="monday-night", week="5")
    assert players_p.exists()
    assert "week-05" in str(players_p)
    assert "monday-night" in str(players_p)


def test_config_from_settings_week_resolution():
    config = SimOptimizerConfig.from_settings(slate="sunday-night", week="5")
    assert config.slate == "sunday-night"
    assert config.week == "week-05"
    assert "week-05/sunday-night" in str(config.players_csv)
    assert "week-05/sunday-night" in str(config.template_csv)
    assert "week-05/sunday-night" in str(config.output_csv)


def test_single_game_auto_detection():
    config = SimOptimizerConfig.from_settings(slate="sunday-night", week="week-05")
    assert config.slate == "sunday-night"

    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()

    # Must be Single Game format (5 players)
    assert optimizer.settings.get_total_players() == 5
    assert optimizer.settings.site == Site.FANDUEL_SINGLE_GAME

    # Verify MVP multiplier was applied
    mvp_players = [p for p in optimizer.player_pool.all_players if "MVP" in p.positions]
    assert len(mvp_players) > 0
    gibbs_mvp = next((p for p in mvp_players if "Gibbs" in p.full_name), None)
    if gibbs_mvp:
        assert gibbs_mvp.fppg > 30.0


def test_single_game_exporter_schema(tmp_path: Path):
    config = SimOptimizerConfig.from_settings(slate="sunday-night", week="week-05")
    config.num_selected_lineups = 2
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()

    lineups = list(optimizer.optimize(2))
    assert len(lineups) == 2

    out_file = tmp_path / "test_completed.csv"
    config.output_csv = out_file
    exporter = FanDuelTemplateExporter(config)
    result = exporter.export_lineups(lineups)

    assert result.exists()
    with open(result, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Exactly 9 columns: 4 metadata + 1 MVP + 4 AnyFLEX
        assert len(header) == 9
        assert header[4] == "MVP - 1.5X Points"
        assert header[5] == "AnyFLEX"
        assert header[8] == "AnyFLEX"

        row1 = next(reader)
        assert len(row1) == 9
        assert ":" in row1[4]
        assert ":" in row1[5]


def test_sim_optimizer_config_exposure_bounds_validation():
    import pytest

    # Direct instantiation with invalid bounds
    with pytest.raises(ValueError, match="Invalid single_game_max_exposure"):
        SimOptimizerConfig(single_game_max_exposure=-0.1)

    with pytest.raises(ValueError, match="Invalid single_game_max_exposure"):
        SimOptimizerConfig(single_game_max_exposure=1.05)

    with pytest.raises(ValueError, match="Invalid max_qb_exposure"):
        SimOptimizerConfig(max_qb_exposure=-0.01)

    with pytest.raises(ValueError, match="Invalid max_exposure"):
        SimOptimizerConfig(max_exposure=1.5)

    # from_settings with overrides
    with pytest.raises(ValueError, match="Invalid single_game_max_exposure"):
        SimOptimizerConfig.from_settings(single_game_max_exposure=-0.5)

    # Valid boundaries: 0.0 and 1.0 must succeed
    cfg_zero = SimOptimizerConfig(single_game_max_exposure=0.0, max_exposure=0.0)
    assert cfg_zero.single_game_max_exposure == 0.0
    cfg_one = SimOptimizerConfig(single_game_max_exposure=1.0, max_exposure=1.0)
    assert cfg_one.single_game_max_exposure == 1.0
