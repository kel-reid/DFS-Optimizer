#!/usr/bin/env python3
"""
================================================================================
DFS Site Auto-Detection Engine
================================================================================
Identifies whether target contest files belong to FanDuel or DraftKings
using file name signatures, with CSV header inspection as a robust fallback.
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Set, Union

logger = logging.getLogger("DFSSiteDetector")

# Explicit file name signature patterns
DRAFTKINGS_NAME_KEYWORDS: Sequence[str] = (
    "dksalaries",
    "dkentries",
    "draftkings",
    "dk_",
    "dk-",
    "dk ",
    "dk.",
)

FANDUEL_NAME_KEYWORDS: Sequence[str] = (
    "fanduel",
    "players-list",
    "entries-upload-template",
    "fd_",
    "fd-",
    "fd ",
    "fd.",
)


def detect_site_from_name(filename: str) -> Optional[str]:
    """
    Examines file name string against known platform naming conventions.
    Returns 'draftkings', 'fanduel', or None if signature is ambiguous.
    """
    lower = filename.lower()

    # Priority 1: Exact keyword or substring matches
    for kw in DRAFTKINGS_NAME_KEYWORDS:
        if kw in lower:
            return "draftkings"
    for kw in FANDUEL_NAME_KEYWORDS:
        if kw in lower:
            return "fanduel"

    # Priority 2: Stem prefixes (e.g. dkcontest.csv, fdcontest.csv)
    stem = Path(filename).stem.lower()
    if stem.startswith("dk"):
        return "draftkings"
    if stem.startswith("fd"):
        return "fanduel"

    return None


def detect_site_from_content(path: Path) -> Optional[str]:
    """
    Inspects the initial line/header of a CSV for platform-specific tokens.
    DraftKings uses 'DST', 'TeamAbbrev', 'AvgPointsPerGame'.
    FanDuel uses 'DEF', 'FPPG', 'Injury Indicator', 'Nickname'.
    """
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            header = f.readline().lower()
            if any(token in header for token in ("avgpointspergame", "teamabbrev", "dst")):
                return "draftkings"
            if any(token in header for token in ("fppg", "injury indicator", "nickname", "def")):
                return "fanduel"
    except Exception as e:
        logger.debug("Could not inspect header of %s: %s", path, e)
    return None


def resolve_site(
    players_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
    slate: Optional[str] = None,
    week: Optional[Union[str, int]] = None,
    slate_date: Optional[str] = None,
) -> str:
    """
    Resolves target DFS site:
      1. Evaluates file name signatures and CSV headers from explicitly supplied files.
      2. If slate is provided, inspects slate-specific directory (data/{week}/{slate}/).
      3. Scans data/templates, data/players, and data/ directories for signatures.
      4. Verifies no ambiguous multi-site files exist simultaneously.
      5. Defaults to 'fanduel' if completely unresolvable.

    Returns:
      Site string ('draftkings' or 'fanduel')
    """
    # 1. Signatures and content inspection from explicitly supplied paths
    explicit_files = [p for p in (template_path, players_path) if p is not None]
    explicit_sites: Set[str] = set()
    for p in explicit_files:
        detected = detect_site_from_name(p.name) or detect_site_from_content(p)
        if detected:
            explicit_sites.add(detected)
    if len(explicit_sites) > 1:
        raise ValueError(
            f"Ambiguous explicit inputs: template and player files resolve to conflicting DFS sites: {explicit_sites}. "
            "Please ensure both files correspond to the same DFS platform."
        )
    if explicit_sites:
        return next(iter(explicit_sites))

    # 2. Slate directory inspection if slate is provided
    if slate:
        from src.data.loader import normalize_week
        norm_w = normalize_week(week)
        candidate_slate_dirs: list[Path] = []
        if norm_w:
            candidate_slate_dirs.append(Path(f"data/{norm_w}/{slate}"))
        if slate_date:
            candidate_slate_dirs.append(Path(f"data/{slate_date}/{slate}"))
        for p in sorted(Path("data").glob(f"week-*/{slate}"), reverse=True):
            if p not in candidate_slate_dirs:
                candidate_slate_dirs.append(p)
        candidate_slate_dirs.append(Path(f"data/{slate}"))

        for s_dir in candidate_slate_dirs:
            if s_dir.exists() and s_dir.is_dir():
                slate_detected: Set[str] = set()
                for p in sorted(s_dir.glob("*.csv")):
                    if p.name.startswith("Completed-") or p.name == "completed_lineups.csv":
                        continue
                    detected = detect_site_from_name(p.name) or detect_site_from_content(p)
                    if detected:
                        slate_detected.add(detected)
                if slate_detected:
                    if len(slate_detected) > 1:
                        raise ValueError(
                            f"Ambiguous slate files detected in '{s_dir}': {sorted(slate_detected)}. "
                            "Please ensure files in this slate directory correspond to a single DFS platform."
                        )
                    return next(iter(slate_detected))

    # 3. File Name Signatures & Content Inspection from standard workspace directories
    scan_dirs = (
        Path("data/templates"),
        Path("data/players"),
        Path("data"),
    )
    all_workspace_detected: Set[str] = set()
    dir_detections = {}

    for scan_dir in scan_dirs:
        if scan_dir.exists():
            detected_in_dir: Set[str] = set()
            for p in sorted(scan_dir.glob("*.csv")):
                # Skip completed output files
                if p.name.startswith("Completed-"):
                    continue
                detected = detect_site_from_name(p.name) or detect_site_from_content(p)
                if detected:
                    detected_in_dir.add(detected)
            if detected_in_dir:
                dir_detections[scan_dir] = detected_in_dir
                all_workspace_detected.update(detected_in_dir)

    if len(all_workspace_detected) > 1:
        details = ", ".join(f"'{d}': {sorted(sites)}" for d, sites in dir_detections.items())
        raise ValueError(
            f"Ambiguous workspace files detected across directories: {details}. "
            "Please keep only one DFS site's files in the workspace or specify --template-csv explicitly."
        )

    if all_workspace_detected:
        return next(iter(all_workspace_detected))

    return "fanduel"
