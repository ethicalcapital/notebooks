# Notebooks Directory

This folder contains marimo notebooks-as-code (`*.py`) that power the site.

- Primary app: `portfolio_growth_simulator.py`
- All notebooks export to interactive HTML (WASM) on every push to `main`.

## Run Locally

- Development (with logs):
  - `uvx marimo -d run notebooks/portfolio_growth_simulator.py`
- Standard run:
  - `uv run notebooks/portfolio_growth_simulator.py`

### Diagnostic script mode (no UI)

To verify the simulation engine without the UI:

- `uv run notebooks/portfolio_growth_simulator.py --diag --years 30 --sims 500 --out diag_out`
- Add `--real` for inflation-adjusted CSVs
- Outputs: `percentiles.csv`, `final_values.csv`, `sample_paths.csv`

## Using the App

Before charts compute:

- Accept the disclaimer (top panel)
- Ensure allocations sum to exactly 100%
- Click “Run simulation” (status will show “Running…” then “Done”).

Advanced controls live in an accordion; charts render after a successful run.

## Conventions

- UI is split into English (simple) vs Nerd (detailed) tracks
- Heavy computations are gated behind a Run button
- Exports: CSV links and a ZIP bundle are provided after a run

## Checks

- Static checks for notebooks: `uv run marimo check notebooks/`
- Syntax: `python3 -m py_compile notebooks/portfolio_growth_simulator.py`

