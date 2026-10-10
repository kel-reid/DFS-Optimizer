"""
================================================================================
NFL DFS Simulation & Optimization Configurations (src/config.py)
================================================================================
Defines typed dataclasses for quantitative parameters, solver constraints,
exposure ceilings, and simulation hyperparameters across DFS platforms.
================================================================================
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("FanDuelSimOptimizer")


def load_yaml_settings(settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration dictionary from YAML settings file."""
    if settings_path is None:
        default_path = Path("config/settings.yaml")
        if default_path.exists():
            settings_path = default_path
        else:
            return {}

    if not settings_path.exists():
        return {}

    try:
        import yaml  # type: ignore[import-untyped]

        with open(settings_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Could not parse settings from %s: %s", settings_path, exc)
        return {}


def detect_entry_fee(template_path: Optional[Path]) -> Optional[float]:
    """Inspects template CSV for entry fee column (e.g. 'entry_fee', 'Entry Fee', 'fee') and returns parsed fee."""
    if not template_path:
        return None
    p = Path(template_path)
    if not p.exists() or not p.is_file():
        return None
    try:
        import csv

        with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            fee_idx = None
            for idx, col in enumerate(header):
                cleaned_col = col.strip().lower()
                if cleaned_col in ("entry_fee", "entry fee", "entry ($)", "fee", "entryfee"):
                    fee_idx = idx
                    break
            if fee_idx is not None:
                first_row = next(reader, None)
                if first_row and len(first_row) > fee_idx:
                    val_str = first_row[fee_idx].strip().replace("$", "").replace(",", "")
                    if val_str:
                        fee = float(val_str)
                        if fee > 0:
                            return fee
    except Exception as exc:
        logger.debug("Could not auto-detect entry fee from %s: %s", p, exc)
    return None


@dataclass
class BaseOptimizerConfig:
    """Base runtime configuration and quantitative hyperparameters for DFS simulation engines."""

    # File paths
    players_csv: Path = Path("data/players.csv")
    template_csv: Path = Path("data/template.csv")
    output_csv: Path = Path("data/completed_lineups.csv")
    projections_csv: Optional[Path] = None

    # Simulation scale parameters
    num_candidates: int = 500        # Size of candidate pool generated via MILP (Stage 1)
    num_field_lineups: int = 10_000  # Number of opponent lineups in tournament field (Stage 2)
    num_sim_trials: int = 5_000      # Number of Monte Carlo game slate realizations (Stage 3)
    num_selected_lineups: int = 150  # Target portfolio size to export (Stage 4)

    # Contest financial parameters
    entry_fee: float = 1.00          # Entry fee per lineup ($1.00 neutral default, auto-detected from template or CLI)
    salary_cap: int = 60_000         # Salary cap ($60,000 FD, $50,000 DK)
    min_field_salary: int = 58_500   # Minimum realistic salary for human field opponents

    # Portfolio exposure ceilings (applied to final selected lineups)
    max_qb_exposure: float = 0.25    # Starting QB exposure ceiling (25% = 37 lineups)
    max_rb_exposure: float = 0.25    # Running back exposure ceiling (25% = 37 lineups)
    max_wr_exposure: float = 0.25    # Wide receiver exposure ceiling (25% = 37 lineups)
    max_te_exposure: float = 0.25    # Tight end exposure ceiling (25% = 37 lineups)
    max_def_exposure: float = 0.20   # Team defense exposure ceiling (20% = 30 lineups)
    max_exposure: float = 0.25       # General individual player exposure ceiling (25% = 37 lineups)
    single_game_max_exposure: float = 0.65  # Default exposure ceiling for Single Game / Showdown slates

    # Candidate generation solver constraints
    stack_ratio: float = 0.80        # 80% primary stacked / 20% unconstrained
    max_repeating_players: int = 6   # Guarantees >= 3 unique players between every pair of candidates
    randomness_deviation: float = 0.25  # ±25% uniform random projection jitter during MILP solving
    exclude_out_injured: bool = True # Prune confirmed OUT, IR, and Doubtful players
    strict_exposure_caps: bool = False # If True, fail if candidate pool cannot fulfill K lineups under hard caps
    zero_unprojected: bool = True    # SaberSim standard: zeroes out unprojected / N/A players
    random_seed: int = 42            # Seed for reproducible Monte Carlo trials

    # Slate configuration & format
    slate: Optional[str] = None       # Specific slate name (e.g. "sunday-night", "main-slate")
    week: Optional[str] = "week-05"   # NFL Week (e.g. "week-05", "5")
    slate_date: Optional[str] = None  # Slate calendar date (e.g. "2026-10-04")
    is_single_game: bool = False      # True for Single Game / Showdown format
    backup_quarterbacks: Optional[Tuple[str, ...]] = None  # Non-starting backup QBs to zero out

    def _validate_exposures(self) -> None:
        exposure_fields = [
            ("max_qb_exposure", self.max_qb_exposure),
            ("max_rb_exposure", self.max_rb_exposure),
            ("max_wr_exposure", self.max_wr_exposure),
            ("max_te_exposure", self.max_te_exposure),
            ("max_def_exposure", self.max_def_exposure),
            ("max_exposure", self.max_exposure),
            ("single_game_max_exposure", self.single_game_max_exposure),
        ]
        for name, val in exposure_fields:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Invalid {name}: {val}. Exposure caps must be between 0.0 and 1.0."
                )


@dataclass
class SimOptimizerConfig(BaseOptimizerConfig):
    """Runtime configuration and quantitative hyperparameters for the FanDuel simulation engine."""

    # File paths
    players_csv: Path = Path("data/week-05/main-slate/players.csv")
    template_csv: Path = Path("data/week-05/main-slate/Completed-FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv")
    output_csv: Path = Path("data/week-05/main-slate/completed_lineups.csv")

    # Contest financial parameters
    salary_cap: int = 60_000         # FanDuel Classic salary cap ($60,000)
    min_field_salary: int = 58_500   # Minimum realistic salary for human field opponents

    def __post_init__(self) -> None:
        if self.week:
            from src.data.loader import normalize_week
            self.week = normalize_week(self.week)

        if self.backup_quarterbacks is None:
            cfg_dict = load_yaml_settings()
            global_cfg = cfg_dict.get("global", {})
            fd_cfg = cfg_dict.get("fanduel", {})
            b_qbs = fd_cfg.get("backup_quarterbacks", global_cfg.get("backup_quarterbacks", []))
            self.backup_quarterbacks = tuple(b_qbs) if b_qbs else ()

        self._validate_exposures()

    @classmethod
    def from_settings(cls, settings_path: Optional[Path] = None, **overrides: Any) -> SimOptimizerConfig:
        """Constructs SimOptimizerConfig by merging config/settings.yaml, environment variables, and keyword overrides."""
        cfg_dict = load_yaml_settings(settings_path)
        global_cfg = cfg_dict.get("global", {})
        fd_cfg = cfg_dict.get("fanduel", {})
        sim_cfg = fd_cfg.get("simulation", {})
        exp_cfg = fd_cfg.get("exposure_caps", {})
        solver_cfg = fd_cfg.get("solver", {})

        params: Dict[str, Any] = {}

        b_qbs = fd_cfg.get("backup_quarterbacks", global_cfg.get("backup_quarterbacks"))
        if b_qbs is not None:
            params["backup_quarterbacks"] = tuple(b_qbs)

        if "salary_cap" in fd_cfg:
            params["salary_cap"] = int(fd_cfg["salary_cap"])
        if "min_field_salary" in fd_cfg:
            params["min_field_salary"] = int(fd_cfg["min_field_salary"])
        if "default_entry_fee" in fd_cfg:
            params["entry_fee"] = float(fd_cfg["default_entry_fee"])

        if "num_candidates" in sim_cfg:
            params["num_candidates"] = int(sim_cfg["num_candidates"])
        if "num_field_lineups" in sim_cfg:
            params["num_field_lineups"] = int(sim_cfg["num_field_lineups"])
        if "num_sim_trials" in sim_cfg:
            params["num_sim_trials"] = int(sim_cfg["num_sim_trials"])
        if "num_selected_lineups" in sim_cfg:
            params["num_selected_lineups"] = int(sim_cfg["num_selected_lineups"])

        for k in ["max_qb_exposure", "max_rb_exposure", "max_wr_exposure", "max_te_exposure", "max_def_exposure", "max_exposure", "single_game_max_exposure"]:
            if k in exp_cfg:
                params[k] = float(exp_cfg[k])

        for k in ["stack_ratio", "randomness_deviation"]:
            if k in solver_cfg:
                params[k] = float(solver_cfg[k])
        for k in ["max_repeating_players"]:
            if k in solver_cfg:
                params[k] = int(solver_cfg[k])
        for k in ["exclude_out_injured", "strict_exposure_caps", "zero_unprojected"]:
            if k in solver_cfg:
                params[k] = bool(solver_cfg[k])

        if "random_seed" in global_cfg:
            params["random_seed"] = int(global_cfg["random_seed"])

        env_map: Dict[str, Tuple[str, Any]] = {
            "DFS_SALARY_CAP": ("salary_cap", int),
            "DFS_MIN_FIELD_SALARY": ("min_field_salary", int),
            "DFS_ENTRY_FEE": ("entry_fee", float),
            "DFS_NUM_CANDIDATES": ("num_candidates", int),
            "DFS_NUM_FIELD": ("num_field_lineups", int),
            "DFS_NUM_TRIALS": ("num_sim_trials", int),
            "DFS_NUM_LINEUPS": ("num_selected_lineups", int),
            "DFS_STACK_RATIO": ("stack_ratio", float),
            "DFS_MAX_REPEATING": ("max_repeating_players", int),
            "DFS_RANDOMNESS": ("randomness_deviation", float),
            "DFS_ZERO_UNPROJECTED": ("zero_unprojected", lambda v: str(v).lower() in ("1", "true", "yes")),
            "DFS_SINGLE_GAME_MAX_EXPOSURE": ("single_game_max_exposure", float),
            "DFS_SLATE": ("slate", str),
            "DFS_WEEK": ("week", str),
            "DFS_SLATE_DATE": ("slate_date", str),
            "DFS_IS_SINGLE_GAME": ("is_single_game", lambda v: str(v).lower() in ("1", "true", "yes")),
            "DFS_BACKUP_QUARTERBACKS": ("backup_quarterbacks", lambda v: tuple(qb.strip() for qb in str(v).split(",") if qb.strip())),
            "DFS_RANDOM_SEED": ("random_seed", int),
        }
        for env_var, (attr, cast) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                params[attr] = cast(val)

        params.update({k: v for k, v in overrides.items() if v is not None})

        # Auto-detect entry fee from template CSV if not explicitly overridden by kwargs or env
        if ("entry_fee" not in overrides or overrides.get("entry_fee") is None) and "DFS_ENTRY_FEE" not in os.environ:
            tmpl_path = params.get("template_csv")
            detected_fee = detect_entry_fee(tmpl_path)
            if detected_fee is not None:
                params["entry_fee"] = detected_fee

        if "week" in params and params["week"] is not None:
            from src.data.loader import normalize_week
            params["week"] = normalize_week(params["week"])

        # Auto-discover slate-specific paths if slate is provided and explicit paths are missing
        slate = params.get("slate")
        week = params.get("week")
        slate_date = params.get("slate_date")
        if slate:
            from src.data.loader import find_players_csv, find_template_csv, normalize_week
            from src.data.projections import find_projections_csv

            norm_week = normalize_week(week)
            if "players_csv" not in overrides or params.get("players_csv") is None:
                try:
                    params["players_csv"] = find_players_csv(slate=slate, week=norm_week, slate_date=slate_date)
                except Exception:
                    pass
            if "template_csv" not in overrides or params.get("template_csv") is None:
                try:
                    params["template_csv"] = find_template_csv(slate=slate, week=norm_week, slate_date=slate_date)
                    if ("entry_fee" not in overrides or overrides.get("entry_fee") is None) and "DFS_ENTRY_FEE" not in os.environ:
                        detected_fee = detect_entry_fee(params.get("template_csv"))
                        if detected_fee is not None:
                            params["entry_fee"] = detected_fee
                except Exception:
                    pass
            if "projections_csv" not in overrides or params.get("projections_csv") is None:
                try:
                    params["projections_csv"] = find_projections_csv(slate=slate, week=norm_week, slate_date=slate_date)
                except Exception:
                    pass
            if "output_csv" not in overrides or params.get("output_csv") is None:
                w_str = norm_week or "week-05"
                slate_dir = Path(f"data/{w_str}/{slate}")
                if not slate_dir.is_dir() and slate_date:
                    date_dir = Path(f"data/{slate_date}/{slate}")
                    if date_dir.is_dir():
                        slate_dir = date_dir
                if slate_dir.is_dir():
                    params["output_csv"] = slate_dir / "completed_lineups.csv"
                else:
                    out_d = Path(f"data/output/{slate}")
                    out_d.mkdir(parents=True, exist_ok=True)
                    tmpl_p = params.get("template_csv")
                    tmpl_name = tmpl_p.name if tmpl_p else "template.csv"
                    params["output_csv"] = out_d / f"Completed-{tmpl_name}"

        return cls(**params)


@dataclass
class DraftKingsConfig(BaseOptimizerConfig):
    """Runtime configuration and quantitative hyperparameters for the DraftKings simulation engine."""

    players_csv: Path = Path("data/players/DKSalaries.csv")
    template_csv: Path = Path("data/templates/DKEntries.csv")
    output_csv: Path = Path("data/output/Completed-DKEntries.csv")
    salary_cap: int = 50_000
    min_field_salary: int = 48_500
    id_format: str = "name_id"

    # Backward compatibility aliases
    salaries_csv: Optional[Path] = None
    entries_csv: Optional[Path] = None
    num_lineups: Optional[int] = None
    max_dst_exposure: Optional[float] = None
    _initialized: bool = False

    def __post_init__(self) -> None:
        if self.salaries_csv is not None:
            self.players_csv = self.salaries_csv
        else:
            super().__setattr__("salaries_csv", self.players_csv)

        if self.entries_csv is not None:
            self.template_csv = self.entries_csv
        else:
            super().__setattr__("entries_csv", self.template_csv)

        if self.num_lineups is not None:
            self.num_selected_lineups = self.num_lineups
        else:
            super().__setattr__("num_lineups", self.num_selected_lineups)

        if self.max_dst_exposure is not None:
            self.max_def_exposure = self.max_dst_exposure
        else:
            super().__setattr__("max_dst_exposure", self.max_def_exposure)

        if self.week:
            from src.data.loader import normalize_week
            self.week = normalize_week(self.week)

        if self.backup_quarterbacks is None:
            cfg_dict = load_yaml_settings()
            global_cfg = cfg_dict.get("global", {})
            dk_cfg = cfg_dict.get("draftkings", {})
            b_qbs = dk_cfg.get("backup_quarterbacks", global_cfg.get("backup_quarterbacks", []))
            self.backup_quarterbacks = tuple(b_qbs) if b_qbs else ()

        self._validate_exposures()
        super().__setattr__("_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if not getattr(self, "_initialized", False):
            return
        if name == "salaries_csv":
            super().__setattr__("players_csv", value)
        elif name == "players_csv":
            super().__setattr__("salaries_csv", value)
        elif name == "entries_csv":
            super().__setattr__("template_csv", value)
        elif name == "template_csv":
            super().__setattr__("entries_csv", value)
        elif name == "num_lineups":
            super().__setattr__("num_selected_lineups", value)
        elif name == "num_selected_lineups":
            super().__setattr__("num_lineups", value)
        elif name == "max_dst_exposure":
            super().__setattr__("max_def_exposure", value)
        elif name == "max_def_exposure":
            super().__setattr__("max_dst_exposure", value)

    @classmethod
    def from_settings(cls, settings_path: Optional[Path] = None, **overrides: Any) -> DraftKingsConfig:
        """Constructs DraftKingsConfig by merging config/settings.yaml, environment variables, and keyword overrides."""
        cfg_dict = load_yaml_settings(settings_path)
        global_cfg = cfg_dict.get("global", {})
        dk_cfg = cfg_dict.get("draftkings", {})
        sim_cfg = dk_cfg.get("simulation", {})
        exp_cfg = dk_cfg.get("exposure_caps", {})
        solver_cfg = dk_cfg.get("solver", {})

        params: Dict[str, Any] = {}

        b_qbs = dk_cfg.get("backup_quarterbacks", global_cfg.get("backup_quarterbacks"))
        if b_qbs is not None:
            params["backup_quarterbacks"] = tuple(b_qbs)

        if "salary_cap" in dk_cfg:
            params["salary_cap"] = int(dk_cfg["salary_cap"])
        if "min_field_salary" in dk_cfg:
            params["min_field_salary"] = int(dk_cfg["min_field_salary"])
        if "default_entry_fee" in dk_cfg:
            params["entry_fee"] = float(dk_cfg["default_entry_fee"])

        if "num_candidates" in sim_cfg:
            params["num_candidates"] = int(sim_cfg["num_candidates"])
        if "num_field_lineups" in sim_cfg:
            params["num_field_lineups"] = int(sim_cfg["num_field_lineups"])
        if "num_sim_trials" in sim_cfg:
            params["num_sim_trials"] = int(sim_cfg["num_sim_trials"])
        if "num_selected_lineups" in sim_cfg:
            params["num_selected_lineups"] = int(sim_cfg["num_selected_lineups"])
        elif "num_lineups" in sim_cfg:
            params["num_selected_lineups"] = int(sim_cfg["num_lineups"])

        for k in ["max_qb_exposure", "max_rb_exposure", "max_wr_exposure", "max_te_exposure", "max_def_exposure", "max_exposure", "single_game_max_exposure"]:
            if k in exp_cfg:
                params[k] = float(exp_cfg[k])
        # Map max_dst_exposure from settings if present
        if "max_dst_exposure" in exp_cfg:
            params["max_def_exposure"] = float(exp_cfg["max_dst_exposure"])

        for k in ["stack_ratio", "randomness_deviation"]:
            if k in solver_cfg:
                params[k] = float(solver_cfg[k])
        for k in ["max_repeating_players"]:
            if k in solver_cfg:
                params[k] = int(solver_cfg[k])
        for k in ["exclude_out_injured", "strict_exposure_caps", "zero_unprojected"]:
            if k in solver_cfg:
                params[k] = bool(solver_cfg[k])

        if "random_seed" in global_cfg:
            params["random_seed"] = int(global_cfg["random_seed"])

        env_map: Dict[str, Tuple[str, Any]] = {
            "DFS_DK_SALARY_CAP": ("salary_cap", int),
            "DFS_SALARY_CAP": ("salary_cap", int),
            "DFS_DK_MIN_FIELD_SALARY": ("min_field_salary", int),
            "DFS_MIN_FIELD_SALARY": ("min_field_salary", int),
            "DFS_DK_ENTRY_FEE": ("entry_fee", float),
            "DFS_ENTRY_FEE": ("entry_fee", float),
            "DFS_DK_NUM_CANDIDATES": ("num_candidates", int),
            "DFS_NUM_CANDIDATES": ("num_candidates", int),
            "DFS_DK_NUM_FIELD": ("num_field_lineups", int),
            "DFS_NUM_FIELD": ("num_field_lineups", int),
            "DFS_DK_NUM_TRIALS": ("num_sim_trials", int),
            "DFS_NUM_TRIALS": ("num_sim_trials", int),
            "DFS_DK_NUM_LINEUPS": ("num_selected_lineups", int),
            "DFS_NUM_LINEUPS": ("num_selected_lineups", int),
            "DFS_DK_STACK_RATIO": ("stack_ratio", float),
            "DFS_STACK_RATIO": ("stack_ratio", float),
            "DFS_DK_MAX_REPEATING": ("max_repeating_players", int),
            "DFS_MAX_REPEATING": ("max_repeating_players", int),
            "DFS_DK_RANDOMNESS": ("randomness_deviation", float),
            "DFS_RANDOMNESS": ("randomness_deviation", float),
            "DFS_DK_MAX_DST_EXPOSURE": ("max_def_exposure", float),
            "DFS_DK_MAX_DEF_EXPOSURE": ("max_def_exposure", float),
            "DFS_MAX_DEF_EXPOSURE": ("max_def_exposure", float),
            "DFS_BACKUP_QUARTERBACKS": ("backup_quarterbacks", lambda v: tuple(qb.strip() for qb in str(v).split(",") if qb.strip())),
            "DFS_DK_BACKUP_QUARTERBACKS": ("backup_quarterbacks", lambda v: tuple(qb.strip() for qb in str(v).split(",") if qb.strip())),
            "DFS_RANDOM_SEED": ("random_seed", int),
        }
        for env_var, (attr, cast) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                params[attr] = cast(val)

        # Normalize alias keys in overrides
        normalized_overrides: Dict[str, Any] = {}
        for k, v in overrides.items():
            if v is not None:
                if k == "salaries_csv":
                    normalized_overrides["players_csv"] = v
                elif k == "entries_csv":
                    normalized_overrides["template_csv"] = v
                elif k == "num_lineups":
                    normalized_overrides["num_selected_lineups"] = v
                elif k == "max_dst_exposure":
                    normalized_overrides["max_def_exposure"] = v
                else:
                    normalized_overrides[k] = v

        params.update(normalized_overrides)

        # Auto-detect entry fee from template CSV if not explicitly overridden by kwargs or env
        if "entry_fee" not in normalized_overrides and "DFS_DK_ENTRY_FEE" not in os.environ and "DFS_ENTRY_FEE" not in os.environ:
            tmpl_path = params.get("template_csv") or params.get("entries_csv")
            detected_fee = detect_entry_fee(tmpl_path)
            if detected_fee is not None:
                params["entry_fee"] = detected_fee

        if "week" in params and params["week"] is not None:
            from src.data.loader import normalize_week
            params["week"] = normalize_week(params["week"])

        return cls(**params)


# Type alias for DraftKings pipeline
DKOptimizerConfig = DraftKingsConfig
