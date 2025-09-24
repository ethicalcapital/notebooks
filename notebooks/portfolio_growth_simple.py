# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.16.1,<0.17",
#   "numpy>=2.0,<3",
# ]
# ///

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return mo,


@app.cell
def _():
    import numpy as np
    return np,


@app.cell
def _(mo):
    mo.md(
        """
    # 🌱 Portfolio Growth (Simple)

    A lightweight sandbox to sanity-check the English-mode defaults of the full simulator.
    """
    )
    return


@app.cell
def _(mo):
    disclaimer = mo.md(
        """
        **DISCLAIMER: Illustrative only.**

        This mini tool reuses the Ethical Capital defaults. It does not provide advice.
        """
    ).callout(kind="danger")
    accept = mo.ui.switch(value=False, label="I understand the illustrative-only disclaimer")
    return accept, disclaimer


@app.cell
def _(mo):
    current_value = mo.ui.number(label="Current portfolio ($)", value=100_000, start=0, stop=10_000_000, step=1_000)
    annual_savings = mo.ui.number(label="Annual savings ($)", value=12_000, start=0, stop=1_000_000, step=500)
    years = mo.ui.slider(label="Years", start=1, stop=40, value=30, step=1, show_value=True)
    growth = mo.ui.slider(label="Annual return (net %)", start=0.0, stop=20.0, value=6.0, step=0.25, show_value=True)
    return annual_savings, current_value, growth, years


@app.cell
def _(mo):
    run = mo.ui.button(label="Run illustration")
    reset = mo.ui.button(label="Reset inputs")
    return reset, run


@app.cell
def _(accept, disclaimer, mo):
    # Display disclaimer and accept switch
    mo.vstack([
        disclaimer,
        accept
    ])
    return


@app.cell
def _(annual_savings, current_value, growth, mo, years):
    # Display input controls
    mo.md("## Input Parameters").center()
    mo.vstack([
        mo.hstack([current_value, annual_savings], justify="space-around"),
        years,
        growth
    ])
    return


@app.cell
def _(mo, reset, run):
    # Display action buttons
    mo.hstack([run, reset], justify="center")
    return


@app.cell
def _(accept, mo, reset, run):
    run_clicks = int(getattr(run, "value", 0) or 0)
    reset_clicks = int(getattr(reset, "value", 0) or 0)
    ready = bool(accept.value)
    should_run = ready and run_clicks > reset_clicks and run_clicks > 0
    status = "✅ Ready" if ready else "⚠️ Accept the disclaimer to run"
    mo.callout(f"{status} · Run: {run_clicks} · Reset: {reset_clicks}", kind="info")
    return ready, reset_clicks, run_clicks, should_run


@app.cell
def _(
    accept,
    annual_savings,
    current_value,
    growth,
    mo,
    np,
    ready,
    reset,
    reset_clicks,
    run,
    run_clicks,
    should_run,
    years,
):
    results = None
    if should_run:
        years_span = int(years.value)
        monthly = float(annual_savings.value) / 12.0
        rate = float(growth.value) / 100.0
        months = years_span * 12
        balances = np.zeros(months + 1, dtype=float)
        balances[0] = float(current_value.value)
        for m in range(1, months + 1):
            prev = balances[m - 1]
            monthly_return = (1.0 + rate) ** (1 / 12) - 1
            balances[m] = prev * (1 + monthly_return) + monthly
        timeline = np.arange(0, months + 1) / 12.0
        results = (timeline, balances)

    if results is None:
        mo.md("Waiting for a run.")
    else:
        timeline, balances = results
        mo.ui.plot(
            x=timeline,
            y=balances,
            layout=dict(title="Illustrative portfolio balance", xaxis_title="Years", yaxis_title="Dollars"),
        )
        mo.callout(
            f"Median-style growth (deterministic) reaches ${balances[-1]:,.0f} after {int(years.value)} years.",
            kind="success",
        )
    return


if __name__ == "__main__":
    app.run()

