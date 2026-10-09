# Project Rules

- Do not perform Git commits under any circumstances unless explicitly instructed to do so by the user.
- Always run the full pytest test suite (`.venv/bin/pytest -v`) and application verification checks before code is committed.
- Whenever fixes addressing PR review comments are committed and pushed, always post a response directly to the discussion thread with the commit reference and a summary of the resolution, and mark the corresponding GitHub review thread as resolved.
- Whenever asked to review PR comments, use the following template:
  - **Location**:
  - **Finding**:
  - **Technical Assessment**:
  - **Recommendation**:
- Do not manually request reviews from automated review bots on pull requests; let automated webhook triggers or the user handle review requests.

## Senior Engineering Protocols

- **Anti-Regression & Invariant Tracing Protocol**: When addressing code review comments, bug reports, or algorithmic edge cases, never perform isolated line edits that only address the cited lines. Explicitly trace and verify the change across the entire pipeline:
  1. Validate boundary preconditions before solving or sampling (e.g. valid salary intervals, active player count, format-specific roster sizes).
  2. Maintain conserved identities across candidate generation, field simulation, vectorized scoring, and template export (e.g. matching dimension $P$, symmetric scoring weights like $1.5\times$ MVP/CPT, and exact portfolio size $K$).
  3. Verify behavior across all slate formats and platforms: Classic vs. Single Game/Showdown, FanDuel vs. DraftKings.
- **Mathematical & Contest Rule Invariants**:
  - **Salary Cap & Roster Composition**: Every generated lineup must strictly satisfy $\sum \text{Salary} \le \text{Salary Cap}$, contest-specific roster slot counts (9 for Classic, 5 for FD Single Game, 6 for DK Showdown), and team diversity rules ($\ge 3$ teams for Classic, $\ge 2$ teams for Single Game).
  - **Scoring & Simulation Scale Symmetry**: The candidate matrix $C$ and opponent field matrix $F$ must represent identical scoring scales. Any scoring multipliers ($1.5\times$ for MVP/CPT) must be applied symmetrically to both candidate and field matrices so simulated percentiles and ROI are mathematically accurate.
  - **Exposure Bounds & Portfolio Integrity**: Player exposures in the exported portfolio must strictly comply with configured position ceilings without integer underflow; always guard against small-$K$ edge cases (`max(1, math.floor(K * cap))`).
- **Boundary Validation Uniformity & Data Externalization**:
  - Validate and sanitize input files (players, templates, projections) at the ingestion boundary before solvers or simulation engines execute.
  - Never embed dynamic domain data (e.g., depth charts, starting QBs, injury lists) as hardcoded literals in source code; externalize them into structured configuration (`settings.yaml`) or projection feeds.
- **Architectural Symmetry & Separation of Concerns**:
  - Maintain strict separation between data ingestion/sanitization, integer linear programming solvers (PuLP/CBC), Monte Carlo simulation engines (NumPy BLAS), and export facades. Avoid monolithic scripts.
  - Maintain architectural parity across supported platforms (DraftKings and FanDuel must both leverage the shared simulation and portfolio selection pipeline).
- **Adversarial Invariant Testing & Verification**:
  - Every bugfix, optimization, and feature must be backed by isolated, reproducible unit and integration tests covering both happy paths and edge cases (boundary conditions, small sample sizes $K < 4$, format transitions, and empty/unprojected player states).
  - Always run the complete test suite (`.venv/bin/pytest -v`), linter (`ruff check .`), and type checker (`mypy`) before proposing code completion or commits.
- **Analytical Code Review Standards**:
  - When performing code reviews, analyze root causes, algorithmic complexity ($O(N)$), numerical stability, maintainability, and operational ergonomics.
  - Deliver clear, structured findings with technical assessments and concrete recommendations.
