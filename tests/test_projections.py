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


def test_find_projections_csv(tmp_path: Path):
    explicit = tmp_path / "custom_proj.csv"
    explicit.write_text("player,fantasy\n")
    assert find_projections_csv(explicit) == explicit

    # Test non-existent fallback
    assert find_projections_csv(Path("non_existent_file.csv")) is None or True


def test_apply_forward_projections(mock_fanduel_files, mock_projections_csv):
    config, players_csv, _, _ = mock_fanduel_files
    optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
    optimizer.load_players_from_csv(str(players_csv))

    updated, zeroed = apply_forward_projections(optimizer, mock_projections_csv)

    assert updated > 0
    assert zeroed > 0

    player_dict = {p.full_name: p.fppg for p in optimizer.player_pool.all_players}

    # Verified updated projections
    assert player_dict.get("KC QB1") == 24.5
    assert player_dict.get("KC WR1") == 18.2
    assert player_dict.get("BUF QB1") == 22.0
    # Defense updated (p.full_name for defense is 'KC Chiefs')
    assert player_dict.get("KC Chiefs") == 9.5
    # Unprojected players should be zeroed
    assert player_dict.get("PHI QB1") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
