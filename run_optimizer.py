#!/usr/bin/env python3
"""
================================================================================
NFL DFS Multi-Site Optimization Launcher (run_optimizer.py)
================================================================================
Generates 150 correlated, risk-controlled tournament lineups for FanDuel and
DraftKings with intelligent file name signature auto-detection:

  python3 run_optimizer.py                  # Auto-detects site and auto-uses .venv
  python3 run_optimizer.py --site fanduel   # Forces FanDuel optimization pipeline
  python3 run_optimizer.py --site draftkings# Forces DraftKings optimization pipeline

Or via executable script:
  ./run_optimizer
================================================================================
"""

import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Auto-Environment Bootstrapper
# Automatically uses .venv if executed with system python without active venv
# -----------------------------------------------------------------------------
project_dir = Path(__file__).parent.resolve()
venv_dir = project_dir / ".venv"
venv_python = venv_dir / "bin" / "python"

if venv_python.exists() and sys.prefix != str(venv_dir):
    try:
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    except Exception:
        pass

# Ensure src module is discoverable
sys.path.insert(0, str(project_dir))

import argparse
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
