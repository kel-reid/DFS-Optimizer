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
from typing import Optional, Sequence, Tuple

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
    explicit_site: Optional[str] = None,
    players_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
) -> str:
    """
    Resolves target DFS site:
      1. Respects manual user flag ('fanduel' or 'draftkings') if specified.
      2. Evaluates file name signatures from template and players files.
      3. Scans data/templates, data/players, and data/ directories for signatures.
      4. Falls back to CSV header token inspection.
      5. Defaults to 'fanduel' if completely unresolvable.

    Returns:
      Site string ('draftkings' or 'fanduel')
    """
    # 1. Manual user override
    if explicit_site and explicit_site.lower() in ("fanduel", "draftkings"):
        return site

    # 2. File Name Signatures from explicitly supplied paths
    explicit_files = [p for p in (template_path, players_path) if p is not None]
    for p in explicit_files:
        detected = detect_site_from_name(p.name)
        if detected:
            return detected

    # 3. File Name Signatures from standard workspace directories
    # Check templates first (target contest), then players, then root data dir
    scan_dirs = (
        Path("data/templates"),
        Path("data/players"),
        Path("data"),
    )
    for scan_dir in scan_dirs:
        if scan_dir.exists():
            for p in sorted(scan_dir.glob("*.csv")):
                # Skip completed output files
                if p.name.startswith("Completed-"):
                    continue
                detected = detect_site_from_name(p.name)
                if detected:
                    return detected

    # 4. Content Inspection Fallback
    all_files_to_inspect = list(explicit_files)
    for scan_dir in scan_dirs:
        if scan_dir.exists():
            for p in scan_dir.glob("*.csv"):
                if not p.name.startswith("Completed-") and p not in all_files_to_inspect:
                    all_files_to_inspect.append(p)

    for p in all_files_to_inspect:
        detected = detect_site_from_content(p)
        if detected:
            return detected

    return "fanduel"
