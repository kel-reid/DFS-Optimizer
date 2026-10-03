# NFL DFS Optimization Engine for FanDuel & DraftKings

A high-performance quantitative optimization and Monte Carlo simulation engine for generating Mass Multi-Entry (MME) portfolios for FanDuel and DraftKings NFL DFS contests.

Powered by `pydfs-lineup-optimizer` (PuLP / CBC integer linear programming solver backend), `numpy`, and `pandas`.

---

## Architecture & Module Organization

The codebase is organized into single-responsibility packages designed for production extensibility:

```
dfs-optimizer/
├── run_optimizer.py             # Root CLI entry point with auto-detection & launcher
├── run_optimizer                # Executable bash launcher wrapper
├── Dockerfile                   # Production container with CBC solver & unprivileged user
├── docker-compose.yml           # Docker Compose runner with mounted data volumes
├── pyproject.toml               # Build metadata, ruff linter, and pytest configuration
├── requirements.txt             # Core production dependencies
├── requirements-dev.txt         # Development & CI dependencies (pytest, ruff)
├── config/
│   └── settings.yaml            # Contest hyperparameters, exposure caps, and solver settings
├── src/
│   ├── config.py                # Strongly-typed dataclass configuration schemas
│   ├── site_detector.py         # Signature & CSV content inspection auto-detection engine
│   ├── build_fanduel_lineups.py # FanDuel pipeline facade
│   ├── build_draftkings_lineups.py # DraftKings pipeline facade
│   ├── data/
│   │   ├── loader.py            # Player pool ingestion, inactive pruning & backup QB filtering
│   │   ├── projections.py       # Forward-looking projection matching & FPPG overwriting
│   │   └── exporter.py          # Contest template slicing & formatted upload CSV generation
│   └── engine/
│       ├── solver.py            # CBC integer linear programming candidate generator (MILP)
│       ├── field.py             # Power-law tournament field opponent simulator (M = 10,000)
│       ├── simulator.py         # Correlated game outcome engine with team shocks & BLAS scoring
│       └── selector.py          # Portfolio selection maximizing ROI subject to exposure caps
├── tests/                       # Pytest test suite with synthetic fixtures
│   ├── conftest.py              # Self-contained mock player pool & template fixtures
│   ├── test_detector.py         # Unit tests for site auto-detection
│   ├── test_projections.py      # Unit tests for name normalization & projection mapping
│   ├── test_constraints.py      # Unit tests for salary caps, stacking, and exposure limits
│   └── test_simulation.py       # Unit tests for field generation, covariance, and export
└── data/
    ├── players/                 # Official contest player pricing CSVs
    ├── templates/               # Contest entry upload template CSVs
    ├── projections/             # Weekly forward-looking projection CSVs
    └── output/                  # Completed, ready-to-upload CSV files
```

---

## Intelligent Site Auto-Detection

The engine automatically inspects file name signatures in `data/templates/`, `data/players/`, and `data/` to determine the target DFS platform without requiring manual flags:

### Signature Detection Rules

| Platform | Keyword Signatures | Prefix Match | Activated Rules |
| :--- | :--- | :--- | :--- |
| **DraftKings** | `DKSalaries`, `DKEntries`, `DraftKings`, `dk_`, `dk-`, `dk `, `dk.` | `dk*` (e.g. `dkcontest.csv`) | **$50,000 Cap**, Full PPR, `QB/2RB/3WR/TE/FLEX/DST` |
| **FanDuel** | `FanDuel`, `players-list`, `entries-upload-template`, `fd_`, `fd-`, `fd `, `fd.` | `fd*` (e.g. `fdcontest.csv`) | **$60,000 Cap**, Half-PPR, `QB/2RB/3WR/TE/FLEX/DEF` |

### Resolution Hierarchy

1. **Explicit Paths**: Inspects file names and CSV headers passed directly to `--template-csv` or `--players-csv`.
2. **Template Directory**: Inspects `data/templates/` (the target contest entries template is prioritized).
3. **Player Directory**: Inspects `data/players/` (player pricing and projection files).
4. **Data Root**: Inspects files dropped directly into `data/`.
5. **CSV Content Fallback**: If file names are generic (e.g. `salaries.csv`), inspects headers for site tokens:
   * DraftKings: `DST`, `TeamAbbrev`, `AvgPointsPerGame`.
   * FanDuel: `DEF`, `FPPG`, `Injury Indicator`, `Nickname`.

You can simply run the optimizer and it will automatically detect the site:
```bash
./run_optimizer
# Or:
python run_optimizer.py
```

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
   * CLI `--entry-fee` parameter (default: 0.05) makes Sim ROI calculation adaptable to any buy-in level.

---

## Simulation Dimensions: N = 500, T = 5,000, K = 150

The Monte Carlo simulation pipeline parameterizes scale across three distinct mathematical dimensions:

* **N = 500 Candidate Lineups (`--num-candidates`)**:
  * The MILP solver generates a broad pool of 500 structurally viable, high-ceiling candidates (80% primary stacked, 20% unconstrained rushing QBs).
* **T = 5,000 Game Slate Trials (`--num-trials`)**:
  * Independent simulated realizations of the full game slate with right-skewed Gamma distributions and log-normal team offensive shocks $\exp(\sigma_{\text{team}} Z_{\text{team}} - 0.5\sigma_{\text{team}}^2)$.
* **K = 150 Portfolio Lineups (`--num-lineups`)**:
  * The target entry portfolio size exported to the contest template (standard FanDuel/DraftKings 150-max MME contests).

---

## Setup & Local Execution

### 1. Local Environment Setup

```bash
# Clone repository
git clone https://github.com/kel-reid/DFS-Optimizer.git
cd DFS-Optimizer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Or install with development & testing dependencies
pip install -r requirements-dev.txt
```

### 2. Running the Optimizer

Place your contest files in the respective directories:
- Player list in `data/players/`
- Entries upload template in `data/templates/`
- (Optional) Weekly projections in `data/projections/`

Then launch:
```bash
./run_optimizer
# Or run with custom CLI parameters:
python run_optimizer.py --entry-fee 0.05 --num-candidates 500 --num-trials 5000
```

The completed upload CSV will be written to `data/output/Completed-[template-name].csv`.

---

## Running Automated Tests

Run the test suite with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with lint check
ruff check .
```

---

## Running with Docker & Docker Compose

A production `Dockerfile` with the `coinor-cbc` solver and an unprivileged user is included.

### Run with Docker Compose:

```bash
docker compose up --build
```

### Or build and run standalone container:

```bash
docker build -t dfs-optimizer .
docker run --rm -v $(pwd)/data:/app/data dfs-optimizer
```
