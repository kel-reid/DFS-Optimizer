"""
Pytest configuration and synthetic fixtures for DFS Optimizer tests.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution and language servers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from src.config import SimOptimizerConfig  # noqa: E402


@pytest.fixture
def mock_fanduel_files(tmp_path: Path):
    """Generates synthetic FanDuel player pool and template CSVs for rapid testing."""
    players_csv = tmp_path / "FanDuel-NFL-players-list.csv"
    template_csv = tmp_path / "FanDuel-NFL-entries-upload-template.csv"
    output_csv = tmp_path / "Completed-FanDuel-NFL-entries-upload-template.csv"

    # Minimal 4-team slate (KC @ BUF, PHI @ DAL)
    teams = [
        ("KC", "BUF", "Chiefs", "Bills"),
        ("PHI", "DAL", "Eagles", "Cowboys"),
    ]

    fieldnames = [
        "Id", "Position", "First Name", "Nickname", "Last Name",
        "FPPG", "Team", "Opponent", "Game", "Injury Indicator",
        "Injury Details", "Tier", "Probable Pitcher", "Batting Order",
        "Roster Position", "Salary"
    ]

    rows = []
    pid = 1000

    for away, home, away_nick, home_nick in teams:
        game_str = f"{away}@{home}"
        for team, opp, nick in [(away, home, away_nick), (home, away, home_nick)]:
            # 1 Starter QB
            pid += 1
            rows.append({
                "Id": f"fd-{pid}", "Position": "QB", "First Name": f"{team}", "Nickname": f"{team} QB1",
                "Last Name": "QB1", "FPPG": "20.0", "Team": team, "Opponent": opp, "Game": game_str,
                "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "",
                "Batting Order": "", "Roster Position": "QB", "Salary": "7500"
            })
            # 1 Backup QB (should be zeroed)
            pid += 1
            rows.append({
                "Id": f"fd-{pid}", "Position": "QB", "First Name": "Case", "Nickname": "Case Keenum",
                "Last Name": "Keenum", "FPPG": "15.0", "Team": team, "Opponent": opp, "Game": game_str,
                "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "",
                "Batting Order": "", "Roster Position": "QB", "Salary": "5000"
            })
            # 2 RBs
            for rbi in range(1, 3):
                pid += 1
                rows.append({
                    "Id": f"fd-{pid}", "Position": "RB", "First Name": f"{team}", "Nickname": f"{team} RB{rbi}",
                    "Last Name": f"RB{rbi}", "FPPG": "14.0", "Team": team, "Opponent": opp, "Game": game_str,
                    "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "",
                    "Batting Order": "", "Roster Position": "RB", "Salary": "6500"
                })
            # 3 WRs
            for wri in range(1, 4):
                pid += 1
                rows.append({
                    "Id": f"fd-{pid}", "Position": "WR", "First Name": f"{team}", "Nickname": f"{team} WR{wri}",
                    "Last Name": f"WR{wri}", "FPPG": "13.0", "Team": team, "Opponent": opp, "Game": game_str,
                    "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "",
                    "Batting Order": "", "Roster Position": "WR", "Salary": "6000"
                })
            # 1 TE
            pid += 1
            rows.append({
                "Id": f"fd-{pid}", "Position": "TE", "First Name": f"{team}", "Nickname": f"{team} TE1",
                "Last Name": "TE1", "FPPG": "11.0", "Team": team, "Opponent": opp, "Game": game_str,
                "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "",
                "Batting Order": "", "Roster Position": "TE", "Salary": "5500"
            })
            # 1 DEF
            pid += 1
            rows.append({
                "Id": f"fd-{pid}", "Position": "D", "First Name": team, "Nickname": f"{nick}",
                "Last Name": nick, "FPPG": "8.0", "Team": team, "Opponent": opp, "Game": game_str,
                "Injury Indicator": "", "Injury Details": "", "Tier": "", "Probable Pitcher": "",
                "Batting Order": "", "Roster Position": "DEF", "Salary": "4000"
            })

    with open(players_csv, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(rows)

    # 10 Entry Template
    template_header = [
        "entry_id", "contest_id", "contest_name", "entry_fee",
        "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"
    ]
    template_rows = []
    for i in range(10):
        template_rows.append([f"E-{1000 + i}", "C-500", "NFL $100K", "$0.05", "", "", "", "", "", "", "", "", ""])

    with open(template_csv, "w", newline="", encoding="utf-8") as f:
        template_writer = csv.writer(f)
        template_writer.writerow(template_header)
        template_writer.writerows(template_rows)

    config = SimOptimizerConfig(
        players_csv=players_csv,
        template_csv=template_csv,
        output_csv=output_csv,
        num_candidates=20,
        num_field_lineups=100,
        num_sim_trials=50,
        num_selected_lineups=10,
        entry_fee=0.05,
        salary_cap=60_000,
        min_field_salary=48_000,
    )
    return config, players_csv, template_csv, output_csv


@pytest.fixture
def mock_projections_csv(tmp_path: Path):
    """Creates a sample projection CSV file."""
    proj_path = tmp_path / "mock_projections.csv"
    data = [
        ["player", "team", "pos", "fantasy"],
        ["KC QB1", "KC", "QB", "24.5"],
        ["KC WR1", "KC", "WR", "18.2"],
        ["BUF QB1", "BUF", "QB", "22.0"],
        ["Chiefs D/ST", "KC", "D/ST", "9.5"],
    ]
    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        proj_writer = csv.writer(f)
        proj_writer.writerows(data)
    return proj_path
