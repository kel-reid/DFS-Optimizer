"""
Unit tests for src/data/projections.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution and language servers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from pydfs_lineup_optimizer import Site, Sport, get_optimizer  # noqa: E402

from src.data.projections import (  # noqa: E402
    apply_forward_projections,
    find_projections_csv,
    normalize_name,
)


def test_normalize_name():
    assert normalize_name("Patrick Mahomes II") == "patrick mahomes"
    assert normalize_name("Travis Etienne Jr.") == "travis etienne"
    assert normalize_name("Marvin Harrison Jr") == "marvin harrison"
    assert normalize_name("Kenneth Walker III") == "kenneth walker"
    assert normalize_name("D'Andre Swift") == "dandre swift"
    assert normalize_name("A.J. Brown") == "aj brown"


def test_find_projections_csv(tmp_path: Path, monkeypatch):
    explicit = tmp_path / "custom_proj.csv"
    explicit.write_text("player,fantasy\n")
    assert find_projections_csv(explicit) == explicit

    # Test non-existent fallback in isolated directory
    monkeypatch.chdir(tmp_path)
    assert find_projections_csv(Path("non_existent_file.csv")) is None
    assert find_projections_csv(None) is None

    # Test discovery from data/projections/ directory
    proj_dir = tmp_path / "data" / "projections"
    proj_dir.mkdir(parents=True)
    fallback_file = proj_dir / "test_projections.csv"
    fallback_file.write_text("player,fppg\n")
    res = find_projections_csv()
    assert res is not None and res.resolve() == fallback_file.resolve()


def test_apply_forward_projections(mock_fanduel_files, mock_projections_csv):
    _, players_csv, _, _ = mock_fanduel_files
    optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
    optimizer.load_players_from_csv(str(players_csv))

    updated, unprojected = apply_forward_projections(optimizer, mock_projections_csv)

    assert updated > 0
    assert unprojected > 0

    player_dict = {p.full_name: p.fppg for p in optimizer.player_pool.all_players}

    # Verified updated projections
    assert player_dict.get("KC QB1") == 24.5
    assert player_dict.get("KC WR1") == 18.2
    assert player_dict.get("BUF QB1") == 22.0
    # Defense updated (p.full_name for defense is 'KC Chiefs')
    assert player_dict.get("KC Chiefs") == 9.5
    # Unprojected players should retain baseline FPPG by default (PHI QB1 has baseline 20.0)
    assert player_dict.get("PHI QB1") == 20.0

    # With zero_unprojected=True, unprojected players should be zeroed
    apply_forward_projections(optimizer, mock_projections_csv, zero_unprojected=True)
    player_dict_zeroed = {p.full_name: p.fppg for p in optimizer.player_pool.all_players}
    assert player_dict_zeroed.get("PHI QB1") == 0.0


def test_apply_forward_projections_draftkings_dst(tmp_path: Path):
    """Verifies DraftKings defenses with position DST correctly match team defense projections."""
    proj_path = tmp_path / "dk_projections.csv"
    proj_path.write_text(
        "player,team,pos,fantasy\n"
        "Josh Allen,BUF,QB,26.0\n"
        "Bills,BUF,DST,11.5\n"
    )

    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
    salaries_csv = tmp_path / "DKSalaries.csv"
    salaries_csv.write_text(
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
        "QB,Josh Allen (1001),Josh Allen,1001,QB,8000,BUF@KC,BUF,22.0\n"
        "DST,Bills  (1002),Bills ,1002,DST,3500,BUF@KC,BUF,6.0\n"
        "DST,Chiefs  (1003),Chiefs ,1003,DST,3000,BUF@KC,KC,5.0\n"
    )
    optimizer.load_players_from_csv(str(salaries_csv))

    updated, unprojected = apply_forward_projections(optimizer, proj_path, zero_unprojected=True)

    assert updated == 2
    assert unprojected == 1

    player_dict = {p.full_name.strip(): p.fppg for p in optimizer.player_pool.all_players}
    assert player_dict.get("Josh Allen") == 26.0
    assert player_dict.get("Bills") == 11.5
    # Chiefs was not in projections, so with zero_unprojected=True it should be 0.0
    assert player_dict.get("Chiefs") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

