# FanDuel NFL DFS Optimization Tool

A quantitative Python optimization pipeline for generating MME tournament lineups for FanDuel NFL DFS contests.

Powered by `pydfs-lineup-optimizer` (PuLP / CBC integer linear programming solver backend) and `pandas`.

---

## Project Directory Structure

```
dfs-optimizer/
|-- data/
|   |-- players/       # Official FanDuel player pool CSVs (*players-list.csv)
|   |   \-- FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv
|   |-- templates/     # Reserved FanDuel contest entry templates (*entries-upload-template.csv)
|   |   \-- FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv
|   \-- output/        # Upload-ready completed CSVs (strictly 151 lines, 13 columns)
|       \-- Completed-FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv
|-- src/
|   |-- __init__.py
|   \-- build_fanduel_lineups.py   # Core ILP solver, stacking logic & export engine
|-- scripts/
|   \-- generate_mock_fanduel_data.py  # Realistic mock test slate generator
|-- run.py             # Top-level pipeline launcher
|-- build_fanduel_lineups.py   # Top-level runner wrapper
|-- requirements.txt   # Python package dependencies
\-- README.md          # Documentation & workflow guide
```

---

## The 3-Step Routine (Each Contest / Week)

| Step | Action | Location |
| :--- | :--- | :--- |
| **1. Player List** | Drop the official FanDuel player download CSV into: | `data/players/` |
| **2. Contest Template** | Drop your reserved contest entries template CSV into: | `data/templates/` |
| **3. Run Optimizer** | Execute the runner command: | `python run.py` |

The completed, upload-ready file will automatically be created in:
`data/output/Completed-<template-name>.csv`

---

## Strategic & Quantitative Constraints

1. **Roster Architecture & Salary Cap**:
   - Strictly enforces FanDuel's 9-player Classic format: `[QB, RB, RB, WR, WR, WR, TE, FLEX, DEF]`.
   - Bounded by the official **$60,000** salary cap.
   - Enforces league diversity rules (players from $\ge 3$ distinct NFL teams, $\le 4$ players from any single team).

2. **Primary Correlation Stacking (80 / 20 Allocation)**:
   - **Phase 1 (80% / 120 lineups)**: Enforces $QB + \ge 1$ Pass Catcher (`WR` or `TE`) from the same franchise using `PositionsStack`.
   - **Phase 2 (20% / 30 lineups)**: Solves unconstrained rosters, enabling standalone rushing quarterbacks (e.g. Josh Allen, Lamar Jackson) without forced pass-catcher pairings.

3. **Negative Correlation Elimination**:
   - Strictly prohibits rostering defensive units (`DEF`) with opposing offensive skill players (`QB`, `RB`, `WR`, `TE`) using `restrict_positions_for_opposing_team`.

4. **Lineup Uniqueness**:
   - Enforces `max_repeating_players = 6`, guaranteeing that every lineup in the 150-entry portfolio differs by at least **3 unique players** from every other lineup.

5. **Pre-Solve Injury & Backup QB Filter**:
   - Prunes confirmed `IR`, `O`, `D`, and `PUP` players while retaining active and Questionable (`Q`) starters.
   - Zeroes out projected points (`FPPG = 0.0`) for non-starting backup QBs (specifically Case Keenum and low-volume backups) so the solver allocates 100% of QB exposure exclusively to active starting quarterbacks.

6. **Portfolio Risk & Diversity Controls**:
   - **Position Exposure Ceilings**:
     - Starting QBs: Max **25%** (max 37 / 150 lineups, distributed across 8–11 starting QBs)
     - Team Defenses: Max **20%** (max 30 / 150 lineups, distributed across 9–10 defenses)
     - Flex Running Backs, Wide Receivers, Tight Ends: Max **25%** (max 37 / 150 lineups)
   - **Monte Carlo Ceiling Diversity**:
     - Applies `RandomFantasyPointsStrategy` with $\pm 25.0\%$ stochastic deviation to simulate variance and create an organic, descending exposure curve.

---

## CLI Options & Customization

```bash
python run.py \
  [--players-csv data/players/my-players.csv] \
  [--template-csv data/templates/my-contest.csv] \
  [--output-csv data/output/my-completed.csv] \
  [--num-lineups 150] \
  [--stack-ratio 0.80] \
  [--max-qb-exposure 0.25] \
  [--max-rb-exposure 0.25] \
  [--max-wr-exposure 0.25] \
  [--max-te-exposure 0.25] \
  [--max-def-exposure 0.20] \
  [--max-exposure 0.25] \
  [--max-repeating 6] \
  [--randomness 0.25] \
  [--keep-injured]
```

### Parameters:
- `--stack-ratio`: Proportion of lineups with mandatory QB + WR/TE same-team stack (default: `0.80` = 80% stacked / 20% unconstrained).
- `--max-qb-exposure`: Maximum portfolio exposure per QB (default: `0.25` / 25%).
- `--max-rb-exposure`: Maximum portfolio exposure per running back (default: `0.25` / 25%).
- `--max-wr-exposure`: Maximum portfolio exposure per wide receiver (default: `0.25` / 25%).
- `--max-te-exposure`: Maximum portfolio exposure per tight end (default: `0.25` / 25%).
- `--max-def-exposure`: Maximum portfolio exposure per defense (default: `0.20` / 20%).
- `--max-exposure`: General individual player ceiling across portfolio (default: `0.25` / 25%).
- `--max-repeating`: Maximum repeating players allowed between any two lineups (default: `6` of 9, forcing $\ge 3$ unique players per roster).
- `--randomness`: Monte Carlo ceiling variance multiplier (default: `0.25` / ±25%).
- `--keep-injured`: Retain players marked with injury indicators (default: `False`, filters out confirmed OUT/IR).

---

## Setup & Installation

```bash
# 1. Clone repository
git clone https://github.com/kel-reid/DFS-Optimizer.git
cd dfs-optimizer

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run optimization
python run.py
```
