"""Unit tests for externalized backup quarterback configuration and filtering."""

from src.build_draftkings_lineups import DKOptimizerConfig, DraftKingsDataLoader
from src.config import DraftKingsConfig, SimOptimizerConfig
from src.data.loader import FanDuelDataLoader


def test_backup_quarterbacks_loaded_from_settings_yaml():
    """Verify SimOptimizerConfig and DraftKingsConfig load backup QBs from settings.yaml by default."""
    fd_cfg = SimOptimizerConfig.from_settings()
    assert fd_cfg.backup_quarterbacks is not None
    assert "Case Keenum" in fd_cfg.backup_quarterbacks
    assert "Drew Lock" in fd_cfg.backup_quarterbacks
    assert len(fd_cfg.backup_quarterbacks) >= 25

    dk_cfg = DraftKingsConfig.from_settings()
    assert dk_cfg.backup_quarterbacks is not None
    assert "Case Keenum" in dk_cfg.backup_quarterbacks


def test_fanduel_loader_filters_configured_backup_qbs(mock_fanduel_files):
    """Verify FanDuelDataLoader zeroes FPPG only for QBs in config.backup_quarterbacks."""
    config, _, _, _ = mock_fanduel_files
    # Explicitly configure Case Keenum as backup, but NOT Drew Lock
    config.backup_quarterbacks = ("Case Keenum",)
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()

    case_keenum = next((p for p in optimizer.player_pool.all_players if p.full_name == "Case Keenum"), None)
    assert case_keenum is not None
    assert case_keenum.fppg == 0.0


def test_fanduel_loader_empty_backup_qbs_leaves_all_qbs_active(mock_fanduel_files):
    """Verify setting backup_quarterbacks=() does not zero out any QBs."""
    config, _, _, _ = mock_fanduel_files
    config.backup_quarterbacks = ()
    loader = FanDuelDataLoader(config)
    optimizer = loader.load_and_sanitize()

    case_keenum = next((p for p in optimizer.player_pool.all_players if p.full_name == "Case Keenum"), None)
    assert case_keenum is not None
    # In mock_fanduel_files, Case Keenum had 15.0 FPPG initially
    assert case_keenum.fppg == 15.0


def test_backup_quarterbacks_environment_variable(monkeypatch):
    """Verify DFS_BACKUP_QUARTERBACKS environment variable sets backup_quarterbacks."""
    monkeypatch.setenv("DFS_BACKUP_QUARTERBACKS", "Custom QB 1, Custom QB 2")
    cfg = SimOptimizerConfig.from_settings()
    assert cfg.backup_quarterbacks == ("Custom QB 1", "Custom QB 2")


def test_draftkings_loader_filters_configured_backup_qbs(tmp_path):
    """Verify DraftKingsDataLoader zeroes FPPG based on config.backup_quarterbacks."""
    dk_csv = tmp_path / "DKSalaries.csv"
    dk_csv.write_text(
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
        "QB,Starter QB (101),Starter QB,101,QB,8000,KC@BUF 01:00PM,KC,22.0\n"
        "QB,Backup QB (102),Backup QB,102,QB,4500,KC@BUF 01:00PM,KC,12.0\n",
        encoding="utf-8",
    )

    config = DKOptimizerConfig(
        players_csv=dk_csv,
        backup_quarterbacks=("Backup QB",),
        exclude_out_injured=False,
    )
    loader = DraftKingsDataLoader(config)
    optimizer = loader.initialize_and_load_optimizer()

    starter = next(p for p in optimizer.player_pool.all_players if p.full_name == "Starter QB")
    backup = next(p for p in optimizer.player_pool.all_players if p.full_name == "Backup QB")

    assert starter.fppg == 22.0
    assert backup.fppg == 0.0


def test_empty_backup_quarterbacks_in_custom_yaml(tmp_path):
    """Verify that an explicit empty list in YAML settings is preserved and does not fall back to default."""
    yaml_file = tmp_path / "custom_settings.yaml"
    yaml_file.write_text(
        "global:\n"
        "  backup_quarterbacks:\n"
        "    - Global QB\n"
        "fanduel:\n"
        "  backup_quarterbacks: []\n"
        "draftkings:\n"
        "  backup_quarterbacks: []\n",
        encoding="utf-8",
    )

    fd_cfg = SimOptimizerConfig.from_settings(settings_path=yaml_file)
    assert fd_cfg.backup_quarterbacks == ()

    dk_cfg = DraftKingsConfig.from_settings(settings_path=yaml_file)
    assert dk_cfg.backup_quarterbacks == ()

    dk_opt_cfg = DKOptimizerConfig.from_settings(settings_path=yaml_file)
    assert dk_opt_cfg.backup_quarterbacks == ()


def test_draftkings_env_precedence(monkeypatch):
    """Verify DFS_DK_BACKUP_QUARTERBACKS overrides generic DFS_BACKUP_QUARTERBACKS when both are present."""
    monkeypatch.setenv("DFS_BACKUP_QUARTERBACKS", "Generic QB")
    monkeypatch.setenv("DFS_DK_BACKUP_QUARTERBACKS", "DraftKings Specific QB")

    dk_cfg = DraftKingsConfig.from_settings()
    assert dk_cfg.backup_quarterbacks == ("DraftKings Specific QB",)

    dk_opt_cfg = DKOptimizerConfig.from_settings()
    assert dk_opt_cfg.backup_quarterbacks == ("DraftKings Specific QB",)
