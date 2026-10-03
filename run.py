#!/usr/bin/env python3
"""Compatibility alias forwarding to run_optimizer.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_optimizer import main

if __name__ == "__main__":
    main()
