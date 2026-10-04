#!/usr/bin/env python3
"""
Root compatibility wrapper for the FanDuel simulation pipeline.
Forwards execution to src.build_fanduel_lineups.
"""
from src.build_fanduel_lineups import main

if __name__ == "__main__":
    main()
