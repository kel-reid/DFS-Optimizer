#!/usr/bin/env python3
"""
================================================================================
DFS NFL Classic MME Pipeline Launcher
================================================================================
Supports FanDuel and DraftKings with intelligent file name signature auto-detection:
  python run.py                     # Auto-detects site from file name signatures
  python run.py --site fanduel      # Explicitly forces FanDuel pipeline
  python run.py --site draftkings   # Explicitly forces DraftKings pipeline
"""

import sys
import argparse
from pathlib import Path

# Ensure src module is discoverable
sys.path.insert(0, str(Path(__file__).parent))

from src.site_detector import resolve_site


def main() -> None:
    # Inspect --site and file arguments without conflicting with site-specific CLI arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--site", choices=["auto", "fanduel", "draftkings"], default="auto")
    parser.add_argument("--players-csv", type=Path, default=None)
    parser.add_argument("--template-csv", type=Path, default=None)
    args, _ = parser.parse_known_args()

    site = resolve_site(
        explicit_site=None if args.site == "auto" else args.site,
        players_path=args.players_csv,
        template_path=args.template_csv,
    )

    if site == "draftkings":
        from src.build_draftkings_lineups import run_draftkings_pipeline
        run_draftkings_pipeline()
    else:
        from src.build_fanduel_lineups import main as run_fanduel_main
        run_fanduel_main()


if __name__ == "__main__":
    main()
