#!/usr/bin/env python3
"""
================================================================================
DFS Site Auto-Detection Engine
================================================================================
Identifies whether target contest files belong to FanDuel or DraftKings
using file name signatures, with CSV header inspection as a robust fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("DFSSiteDetector")

# Explicit file name signature patterns
DRAFTKINGS_NAME_KEYWORDS: Sequence[str] = (
    "dksalaries",
    "dkentries",
    "draftkings",
    "dk_",
    "dk-",
)

FANDUEL_NAME_KEYWORDS: Sequence[str] = (
    "fanduel",
    "players-list",
    "entries-upload-template",
    "fd_",
    "fd-",
)


def detect_site_from_name(filename: str) -> Optional[str]:
    """
    Examines file name string against known platform naming conventions.
    Returns 'draftkings', 'fanduel', or None if signature is ambiguous.
    """
    lower = filename.lower()
    for kw in DRAFTKINGS_NAME_KEYWORDS:
        if kw in lower:
            return "draftkings"
    for kw in FANDUEL_NAME_KEYWORDS:
        if kw in lower:
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
      3. Falls back to CSV header token inspection.
      4. Defaults to 'fanduel' if completely unresolvable.
    """
    # 1. Manual user override
    if explicit_site and explicit_site.lower() in ("fanduel", "draftkings"):
        site = explicit_site.lower()
        logger.info("[CONFIG] Site explicitly set to: %s", site.upper())
        return site

    # 2. File Name Signatures
    files_to_check = [p for p in (template_path, players_path) if p is not None]

    for p in files_to_check:
        detected = detect_site_from_name(p.name)
        if detected:
            logger.info("[AUTO-DETECT] Identified %s from file name signature: '%s'",
                        detected.upper(), p.name)
            return detected

    # Also inspect any existing files in data/templates or data/players
    for scan_dir in (Path("data/templates"), Path("data/players")):
        if scan_dir.exists():
            for p in scan_dir.glob("*.csv"):
                detected = detect_site_from_name(p.name)
                if detected:
                    logger.info("[AUTO-DETECT] Identified %s from folder file signature: '%s'",
                                detected.upper(), p.name)
                    return detected

    # 3. Content Inspection Fallback
    for p in files_to_check:
        detected = detect_site_from_content(p)
        if detected:
            logger.info("[AUTO-DETECT] Identified %s from CSV header inspection: '%s'",
                        detected.upper(), p.name)
            return detected

    # 4. Default
    logger.info("[AUTO-DETECT] No distinct signature detected. Defaulting to: FANDUEL")
    return "fanduel"
