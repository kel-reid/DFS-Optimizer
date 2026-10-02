# NFL DFS Multi-Site Optimization Tool (FanDuel & DraftKings)

A quantitative Python optimization pipeline for generating mass multi-entry (MME) tournament lineups for FanDuel and DraftKings NFL DFS contests.

Powered by `pydfs-lineup-optimizer` (PuLP / CBC integer linear programming solver backend) and `pandas`.

---

## Project Directory Structure

```
dfs-optimizer/
|-- data/
|   |-- players/       # Official player pools (FanDuel or DraftKings)
|   |   |-- FanDuel-NFL-2026 EDT-10 EDT-04 EDT-134747-players-list.csv
|   |   \-- DKSalaries.csv (DraftKings)
|   |-- templates/     # Reserved contest entry templates
|   |   |-- FanDuel-NFL-2026-10-04-134747-entries-upload-template.csv
|   |   \-- DKEntries.csv (DraftKings)
|   \-- output/        # Upload-ready completed CSVs
|       \-- Completed-<template-name>.csv
|-- src/
|   |-- __init__.py
|   |-- site_detector.py           # File name signature auto-detection engine
|   |-- build_fanduel_lineups.py   # FanDuel solver ($60k cap, DEF, Half-PPR)
|   \-- build_draftkings_lineups.py# DraftKings solver ($50k cap, DST, Full PPR)
|-- scripts/
|   |-- generate_mock_fanduel_data.py    # Mock FanDuel test generator
|   \-- generate_mock_draftkings_data.py # Mock DraftKings test generator
|-- run.py             # Top-level pipeline launcher (auto-detects site)
|-- build_fanduel_lineups.py   # Top-level runner wrapper
|-- requirements.txt   # Python package dependencies
\-- README.md          # Documentation & workflow guide
```

---

## Intelligent Site Auto-Detection

The engine automatically detects whether you are solving for **FanDuel** or **DraftKings** by inspecting the file name signatures of files placed in `data/players/` and `data/templates/`:

* **DraftKings Signatures**: File names containing `DKSalaries`, `DKEntries`, `DraftKings`, `dk_`, or `dk-`.
  * Triggers the **$50,000 salary cap**, Full PPR scoring, and `DST` roster mapping.
* **FanDuel Signatures**: File names containing `FanDuel`, `players-list`, `entries-upload-template`, `fd_`, or `fd-`.
  * Triggers the **$60,000 salary cap**, Half-PPR scoring, and `DEF` roster mapping.

You can also explicitly specify the site via the `--site` flag:
```bash
python run.py                   # Auto-detects site from file name signatures
python run.py --site fanduel    # Explicit FanDuel run
python run.py --site draftkings # Explicit DraftKings run
```

---

## The 3-Step Routine (Each Contest / Week)

| Step | Action | Location |
| :--- | :--- | :--- |
| **1. Player List** | Drop your site's player list CSV into: | `data/players/` |
| **2. Contest Template** | Drop your reserved contest entries template into: | `data/templates/` |
| **3. Run Optimizer** | Execute the runner command: | `python run.py` |

The completed, upload-ready file will automatically be created in:
`data/output/Completed-<template-name>.csv`

---

## Strategic & Quantitative Constraints

1. **Roster Architecture & Salary Cap**:
   * **FanDuel**: 9 players (`QB, RB, RB, WR, WR, WR, TE, FLEX, DEF`) under **$60,000** salary cap.
   * **DraftKings**: 9 players (`QB, RB, RB, WR, WR, WR, TE, FLEX, DST`) under **$50,000** salary cap.
   * Enforces league diversity rules (players from >= 3 distinct NFL teams, <= 4 players from any single team).

2. **Primary Correlation Stacking (80 / 20 Allocation)**:
   * **Phase 1 (80% / 120 lineups)**: Enforces QB + >= 1 Pass Catcher (`WR` or `TE`) from the same franchise using `PositionsStack`.
   * **Phase 2 (20% / 30 lineups)**: Solves unconstrained rosters, enabling standalone rushing quarterbacks (e.g. Josh Allen, Lamar Jackson) without forced pass-catcher pairings.

3. **Negative Correlation Elimination**:
   * Strictly prohibits rostering defensive units (`DEF` / `DST`) with opposing offensive skill players (`QB`, `RB`, `WR`, `TE`) using `restrict_positions_for_opposing_team`.

4. **Lineup Uniqueness**:
   * Enforces `max_repeating_players = 6`, guaranteeing that every lineup in the 150-entry portfolio differs by at least **3 unique players** from every other lineup.

5. **Pre-Solve Injury & Backup QB Filter**:
   * Prunes confirmed `IR`, `O`, `D`, and `PUP` players while retaining active and Questionable (`Q`) starters.
   * Zeroes out projected points (`FPPG = 0.0`) for non-starting backup QBs (including Case Keenum) so the solver allocates 100% of QB volume exclusively to active starting quarterbacks.

6. **Portfolio Risk & Diversity Controls**:
   * **Position Exposure Ceilings**:
     * Starting QBs: Max **25%** (max 37 / 150 lineups, distributed across 8-11 starting QBs)
     * Team Defenses: Max **20%** (max 30 / 150 lineups, distributed across 9-10 defenses)
     * Flex Running Backs, Wide Receivers, Tight Ends: Max **25%** (max 37 / 150 lineups)
   * **Monte Carlo Ceiling Diversity**:
     * Applies `RandomFantasyPointsStrategy` with +/- 25.0% stochastic deviation to simulate variance and create an organic, descending exposure curve.

---

## CLI Options & Customization

```bash
python run.py \
  [--site {auto,fanduel,draftkings}] \
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

---

## Setup & Installation

```bash
# 1. Clone repository
git clone https://github.com/kel-reid/DFS-Optimizer.git
cd DFS-Optimizer

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run optimization
python run.py
```
