#!/usr/bin/env python3
"""
================================================================================
Top-Level Pipeline Entry Point (Auto-Detects Platform)
================================================================================
"""
import sys
from pathlib import Path

# Ensure src module is discoverable
sys.path.insert(0, str(Path(__file__).parent))

from src.site_detector import resolve_site


def main() -> None:
    site = resolve_site()
    if site == "draftkings":
        from src.build_draftkings_lineups import run_draftkings_pipeline
        run_draftkings_pipeline()
    else:
        from src.build_fanduel_lineups import main as run_fanduel_main
        run_fanduel_main()


if __name__ == "__main__":
    main()
