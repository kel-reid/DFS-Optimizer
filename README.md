# NFL DFS Optimization Engine

A quantitative optimization and Monte Carlo simulation engine for generating MME lineups for FanDuel and DraftKings NFL DFS contests.

Powered by `pydfs-lineup-optimizer` (PuLP / CBC integer linear programming solver backend), `numpy`, and `pandas`.


## Intelligent Site Auto-Detection

The engine automatically inspects file name signatures in `data/templates/`, `data/players/`, and `data/` to determine the target DFS platform without requiring manual flags.


## Strategic & Quantitative Constraints

1. **Roster Architecture & Salary Cap**:
   * **Classic Formats**:
     * **FanDuel Classic**: 9 players (`QB, RB, RB, WR, WR, WR, TE, FLEX, DEF`) under **$60,000** salary cap (players from $\ge 3$ teams).
     * **DraftKings Classic**: 9 players (`QB, RB, RB, WR, WR, WR, TE, FLEX, DST`) under **$50,000** salary cap (players from $\ge 2$ teams).
   * **Single Game / Showdown Formats**:
     * **FanDuel Single Game**: 5 players (1 `MVP` at $1.5\times$ scoring multiplier + 4 `AnyFLEX`) under **$60,000** salary cap.
     * **DraftKings Showdown**: 6 players (1 `CPT` at $1.5\times$ salary and scoring multiplier + 5 `FLEX`) under **$50,000** salary cap.

2. **Primary Correlation Stacking (80 / 20 Allocation)**:
   * **Phase 1 (80% / 120 lineups)**: Enforces QB + >= 1 Pass Catcher (`WR` or `TE`) from the same franchise using `PositionsStack`.
   * **Phase 2 (20% / 30 lineups)**: Solves unconstrained rosters, enabling standalone rushing quarterbacks (e.g. Josh Allen, Lamar Jackson) without forced pass-catcher pairings.

3. **Negative Correlation Elimination**:
   * Strictly prohibits rostering defensive units (`DEF` / `DST`) with opposing offensive skill players (`QB`, `RB`, `WR`, `TE`) using `restrict_positions_for_opposing_team`.

4. **Lineup Uniqueness**:
   * Enforces `max_repeating_players = 6`, guaranteeing that every lineup in the 150-entry portfolio differs by at least **3 unique players** from every other lineup.

5. **Pre-Solve Injury & Backup QB Filter**:
   * Prunes confirmed `IR`, `O`, `D`, and `PUP` players while retaining active and Questionable (`Q`) starters.
   * Zeroes out projected points (`FPPG = 0.0`) for non-starting backup QBs so the solver allocates 100% of QB volume exclusively to active starting quarterbacks.

6. **Portfolio Risk & Diversity Controls**:
   * **Position Exposure Ceilings**:
     * Starting QBs: Max **25%** (max 37 / 150 lineups, distributed across 8-11 starting QBs)
     * Team Defenses: Max **20%** (max 30 / 150 lineups, distributed across 9-10 defenses)
     * Flex Running Backs, Wide Receivers, Tight Ends: Max **25%** (max 37 / 150 lineups)
   * **Monte Carlo Ceiling Diversity**:
     * Applies `RandomFantasyPointsStrategy` with +/- 25.0% stochastic deviation to simulate variance and create an organic, descending exposure curve.

7. **Normalized GPP Percentile Payout Structure**:
   * Ranks candidates against the field distribution across 5,000 Monte Carlo game trials using `np.searchsorted`.
   * Dynamic finish percentiles adaptable to any contest size and entry fee:
     * **Top 0.01%** (1st place tier): **10,000x** entry fee
     * **Top 0.1%** (Elite tier): **500x** entry fee
     * **Top 1.0%** (High equity tier): **20x** entry fee
     * **Top 5.0%** (Mid cash tier): **5x** entry fee
     * **Top 20.0%** (Min-cash line): **1.5x** entry fee
   * **Contest-Agnostic Entry Fee Resolution**:
     * Automatically auto-detects entry fee from template CSV column headers (`Entry Fee`, `entry_fee`, `Fee`).
     * Supports environment variable overrides (`DFS_ENTRY_FEE`, `DFS_DK_ENTRY_FEE`) and CLI `--entry-fee` flag.
     * Defaults to a neutral **$1.00** baseline if unstated, allowing seamless scaling across any buy-in tier ($0.05 micro-stakes to $100+ high-stakes).

## Simulation Dimensions: N = 500, T = 5,000, K = 150

The Monte Carlo simulation pipeline parameterizes scale across three distinct mathematical dimensions:

* **N = 500 Candidate Lineups (`--num-candidates`)**:
  * The MILP solver generates a broad pool of 500 structurally viable, high-ceiling candidates (80% primary stacked, 20% unconstrained rushing QBs).
* **T = 5,000 Game Slate Trials (`--num-trials`)**:
  * Independent simulated realizations of the full game slate with right-skewed Gamma distributions and log-normal team offensive shocks $\exp(\sigma_{\text{team}} Z_{\text{team}} - 0.5\sigma_{\text{team}}^2)$.
* **K = 150 Portfolio Lineups (`--num-lineups`)**:
  * The target entry portfolio size exported to the contest template (standard FanDuel/DraftKings 150-max MME contests).

## Setup & Local Execution

### 1. Local Environment Setup

```bash
# Clone repository
git clone https://github.com/kel-reid/DFS-Optimizer.git
cd DFS-Optimizer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Optimizer

Both FanDuel and DraftKings share the unified 4-stage Monte Carlo optimization and simulation pipeline (`CandidatePoolGenerator` $\to$ `OpponentFieldSimulator` $\to$ `CorrelatedGameEngine` $\to$ `PortfolioSelector`).

#### FanDuel Execution

Contest slates are organized into directories under `data/week-<XX>/<slate>/`:
- `players.csv` (FanDuel player list)
- `entries_template.csv` (FanDuel contest template)
- `projections.csv` (External forward-looking projections)

```bash
# Run for a specific slate (e.g. Sunday Night Single Game)
python src/build_fanduel_lineups.py --week 5 --slate sunday-night

# Run for Main Slate
python src/build_fanduel_lineups.py --week 5 --slate main-slate

# Or run with custom simulation parameters:
python src/build_fanduel_lineups.py --week 5 --slate sunday-night --entry-fee 1.00 --num-candidates 500 --num-trials 5000
```

*Completed lineup file is written to `data/week-<XX>/<slate>/completed_lineups.csv`.*

#### DraftKings Execution

DraftKings runs against player salary and contest template files (either in `data/players/` and `data/templates/` or a custom slate path):

```bash
# Run DraftKings with automatic template fee detection
python src/build_draftkings_lineups.py

# Run with explicit file paths and simulation parameters (also accepts --salaries-csv and --entries-csv):
python src/build_draftkings_lineups.py \
  --players-csv data/players/DKSalaries.csv \
  --template-csv data/templates/DKEntries.csv \
  --num-candidates 500 \
  --num-trials 5000 \
  --entry-fee 3.00
```

*Completed lineup file is written to `data/output/Completed-DKEntries.csv`.*

## Quality Verification & Tests

Run the full automated test suite, linter, and type checker:

```bash
# Run all unit and integration tests
.venv/bin/pytest -v

# Run linter
ruff check .

# Run static type checker
mypy src tests
```

