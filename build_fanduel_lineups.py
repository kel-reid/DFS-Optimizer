#!/usr/bin/env python3
"""Top-level entry point for FanDuel NFL Classic MME Pipeline."""
import sys
from pathlib import Path

# Ensure src module is discoverable
sys.path.insert(0, str(Path(__file__).parent))

from src.build_fanduel_lineups import main

if __name__ == "__main__":
    main()
