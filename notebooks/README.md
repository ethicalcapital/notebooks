# Notebooks Directory

This folder contains marimo notebooks-as-code (`*.py`) that power the site.

## Available Notebooks

- **`portfolio_growth_simulator.py`** - Monte Carlo portfolio growth simulation with comprehensive analysis
- **`portfolio_growth_simple.py`** - Lightweight portfolio growth calculator for quick estimates
- **`safe_withdrawal_rate.py`** - Safe Withdrawal Rate calculator using historical backtesting (1928-2020)
- **`button_add_two.py`** - Simple demo notebook for testing functionality

All notebooks export to interactive HTML (WASM) on every push to `main`.

## Run Locally

### Prerequisites

**Install marimo with full dependencies:**
```bash
uv tool install "marimo[recommended]" --with numpy --with pandas --with plotly
```

This installs marimo with optional dependencies (altair, duckdb, openai, polars) plus the required notebook libraries.

### Running Notebooks

**Portfolio Growth Simulator:**
- Development: `marimo edit notebooks/portfolio_growth_simulator.py`
- Production: `marimo run notebooks/portfolio_growth_simulator.py`

**Safe Withdrawal Rate Calculator (Full):**
- `marimo run notebooks/safe_withdrawal_rate.py`

**Safe Withdrawal Rate Calculator (WASM-Compatible):**
- `marimo run notebooks/safe_withdrawal_rate_simple.py`

**Simple Portfolio Calculator:**
- `marimo run notebooks/portfolio_growth_simple.py`

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

## Safe Withdrawal Rate Features

The SWR calculator includes:

- **Historical Backtesting:** Uses real market data from 1928-2020
- **Flexible Strategies:** Fixed, variable, and floor-based withdrawal rules
- **Risk Analysis:** Configurable risk aversion parameters
- **Failure Rate Analysis:** Probability of portfolio exhaustion
- **Multiple Presets:** Bengen 4% rule, conservative, moderate, and aggressive strategies

## Troubleshooting

### Service Worker Errors
If you see browser console errors like:
```
Error registering service worker: TypeError: Cannot read properties of null (reading 'postMessage')
```

**Solutions:**
1. **Use Edit Mode Instead:** `marimo edit notebook.py` (works around service worker issues)
2. **Try Different Browser:** Chrome typically has better marimo compatibility than Safari/Firefox
3. **Clear Browser Cache:** Hard refresh with Cmd+Shift+R
4. **Check Full Dependencies:** Ensure you installed `marimo[recommended]` with all optional dependencies

### Button/UI Not Working
- **Backend Errors:** Check terminal for runtime errors (tuple access, import issues)
- **Cell Dependencies:** Ensure UI elements are created in one cell and accessed in separate dependent cells
- **Missing Dependencies:** Install complete marimo: `uv tool install "marimo[recommended]" --with numpy --with pandas --with plotly`

## Checks

- Static checks for all notebooks: `uv run marimo check notebooks/`
- Syntax check: `python3 -m py_compile notebooks/*.py`

