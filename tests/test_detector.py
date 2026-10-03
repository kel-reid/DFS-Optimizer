"""
Unit tests for src/site_detector.py.
"""

from pathlib import Path

from src.site_detector import (
    detect_site_from_content,
    detect_site_from_name,
    resolve_site,
)


def test_detect_site_from_name_draftkings():
    dk_names = [
        "DKSalaries.csv",
        "DKSalaries (1).csv",
        "dksalaries.csv",
        "DKEntries.csv",
        "dkentries.csv",
        "DraftKings_NFL_Classic.csv",
        "dk_main_slate.csv",
        "dk-week4.csv",
        "dk contest.csv",
        "dk.slates.csv",
    ]
    for fn in dk_names:
        assert detect_site_from_name(fn) == "draftkings"


def test_detect_site_from_name_fanduel():
    fd_names = [
        "FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv",
        "FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv",
        "fanduel_salaries.csv",
        "players-list.csv",
        "entries-upload-template.csv",
        "fd_salaries.csv",
        "fd-main.csv",
        "fd slate.csv",
    ]
    for fn in fd_names:
        assert detect_site_from_name(fn) == "fanduel"


def test_detect_site_from_name_unknown():
    assert detect_site_from_name("random_unrelated_file.csv") is None


def test_detect_site_from_content_draftkings(tmp_path: Path):
    f = tmp_path / "sample.csv"
    f.write_text("Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n")
    assert detect_site_from_content(f) == "draftkings"


def test_detect_site_from_content_fanduel(tmp_path: Path):
    f = tmp_path / "sample.csv"
    f.write_text("Id,Position,First Name,Nickname,Last Name,FPPG,Team,Opponent,Game,Injury Indicator\n")
    assert detect_site_from_content(f) == "fanduel"


def test_resolve_site_explicit_paths(tmp_path: Path):
    dk_f = tmp_path / "DKSalaries.csv"
    dk_f.write_text("Position,Name + ID,TeamAbbrev,AvgPointsPerGame\n")
    assert resolve_site(players_path=dk_f) == "draftkings"

    fd_f = tmp_path / "FanDuel-NFL-players-list.csv"
    fd_f.write_text("Id,Position,FPPG,Nickname\n")
    assert resolve_site(players_path=fd_f) == "fanduel"
