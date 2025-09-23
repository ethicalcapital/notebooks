# Ethical Capital — Public Notebooks

This repository hosts Ethical Capital’s interactive marimo notebooks and publishes interactive HTML (WASM) to GitHub Pages.

- Live site: GitHub Pages (auto-deployed from `docs/` by Actions)
- Source notebooks: `notebooks/*.py` (marimo notebooks-as-code)

## Run locally (uv)

- One-liner to serve a notebook UI:
  - `uvx --with numpy --with pandas --with plotly marimo run notebooks/portfolio_growth_simulator.py`
- Or use inline PEP 723 deps with uv:
  - `uv run notebooks/portfolio_growth_simulator.py`

### Diagnostic script mode (no UI)

For quick headless verification of the simulation engine:

- Run once and export CSVs (percentiles, final values, sample paths):
  - `uv run notebooks/portfolio_growth_simulator.py --diag --years 30 --sims 500 --out diag_out`
- Add `--real` to export inflation‑adjusted values.
- Output directory will contain `percentiles.csv`, `final_values.csv`, and `sample_paths.csv`.

## Add a new notebook

1. Create a new marimo notebook (example):
   - `marimo create notebooks/my_notebook.py`
2. Develop locally:
   - `uvx --with numpy --with pandas --with plotly marimo run notebooks/my_notebook.py`
3. Commit and push to `main`. The GitHub Action will export to HTML into `docs/` and deploy to Pages.

## GitHub Pages deployment

The workflow `.github/workflows/publish.yml`:
- Installs `marimo`
- Exports every notebook in `notebooks/*.py` to interactive `docs/*.html` using `--to html-wasm`
- Generates `docs/index.html` with links
- Publishes to GitHub Pages

Repository settings to confirm (one-time):
- Pages → Build and deployment → Source: GitHub Actions
- Default branch: `main`

## Brand

Charts adopt Ethical Capital brand colors from `Ethical-Capital-Brand-Kit/palette/palette.json`.

## Disclaimers

This analysis illustrates mathematical principles only and does not predict or project the performance of any specific investment or investment strategy. Past performance does not guarantee future results.

## License

Except where otherwise noted, content in this repository is licensed under the Creative Commons Attribution‑ShareAlike 4.0 International (CC BY‑SA 4.0).

See `LICENSE` for the full legal text.
