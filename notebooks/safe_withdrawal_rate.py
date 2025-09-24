# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.16.1,<0.17",
#   "numpy>=2.0,<3",
#   "pandas>=2.2,<3",
#   "matplotlib>=3.8,<4",
#   "plotly>=5.20,<6.1",
# ]
# ///

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import sys
    import os

    # Add the current directory to path for SWR modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Import SWR modules
    from SWRsimulation.SWRsimulationCE import SWRsimulationCE

    return SWRsimulationCE, mo, np, os, pd, sys, swr_path


@app.cell
def _(mo):
    mo.md(
        """
        # 🎯 Safe Withdrawal Rate Calculator

        This tool helps you determine optimal withdrawal strategies for retirement portfolios
        using historical market data and Monte Carlo analysis. Based on the excellent SWR
        library by @druce.

        **Key Features:**
        - Historical backtesting (1928-2020)
        - Certainty equivalent spending optimization
        - Flexible withdrawal strategies
        - Asset allocation optimization
        """
    )
    return


@app.cell
def _(mo):
    disclaimer = mo.md(
        """
        **⚠️  DISCLAIMER: For Educational Purposes Only**

        This calculator is for illustrative and educational purposes only. It does not
        constitute financial advice. Past performance does not guarantee future results.
        Consult with a qualified financial advisor before making investment decisions.
        """
    ).callout(kind="warn")

    accept_disclaimer = mo.ui.switch(
        value=False,
        label="I understand this is for educational purposes only"
    )

    mo.vstack([disclaimer, accept_disclaimer])
    return accept_disclaimer, disclaimer


@app.cell
def _(mo):
    # Load historical returns data
    try:
        swr_data_path = os.path.join(parent_dir, 'real_return_df.pickle')
        returns_df = pd.read_pickle(swr_data_path)
        data_loaded = True
        data_info = f"📈 Historical data loaded: {returns_df.shape[0]} years ({returns_df.index.min()}-{returns_df.index.max()})"
    except Exception as e:
        returns_df = None
        data_loaded = False
        data_info = f"❌ Error loading data: {e}"

    mo.md(data_info).callout(kind="success" if data_loaded else "danger")
    return data_info, data_loaded, returns_df, swr_data_path


@app.cell
def _(mo):
    mo.md("## Portfolio Configuration")
    return


@app.cell
def _(mo):
    # Portfolio inputs
    retirement_years = mo.ui.slider(
        label="Retirement duration (years)",
        start=20, stop=50, value=30, step=1, show_value=True
    )

    stock_allocation = mo.ui.slider(
        label="Stock allocation (%)",
        start=0, stop=100, value=60, step=5, show_value=True
    )

    mo.vstack([
        mo.md("### Time Horizon & Asset Allocation"),
        retirement_years,
        stock_allocation,
        mo.md(f"**Bond allocation:** {100 - stock_allocation.value}%")
    ])
    return retirement_years, stock_allocation


@app.cell
def _(mo):
    # Withdrawal strategy inputs
    mo.md("### Withdrawal Strategy")
    return


@app.cell
def _(mo):
    withdrawal_strategy = mo.ui.dropdown(
        label="Strategy preset",
        options={
            "bengen": "Bengen 4% Rule",
            "conservative": "Conservative Variable",
            "moderate": "Moderate Variable",
            "aggressive": "Aggressive Variable",
            "custom": "Custom Parameters"
        },
        value="bengen"
    )

    # Preset configurations
    presets = {
        "bengen": {"fixed": 4.0, "variable": 0.0, "floor": 4.0},
        "conservative": {"fixed": 3.5, "variable": 1.0, "floor": 3.5},
        "moderate": {"fixed": 3.0, "variable": 1.5, "floor": 3.0},
        "aggressive": {"fixed": 2.5, "variable": 2.0, "floor": 2.5},
        "custom": {"fixed": 4.0, "variable": 0.0, "floor": 4.0}
    }

    withdrawal_strategy
    return presets, withdrawal_strategy


@app.cell
def _(mo, presets, withdrawal_strategy):
    selected_preset = presets[withdrawal_strategy.value]

    fixed_pct = mo.ui.number(
        label="Fixed withdrawal (%)",
        value=selected_preset["fixed"],
        start=0.0, stop=8.0, step=0.1
    )

    variable_pct = mo.ui.number(
        label="Variable withdrawal (% of portfolio)",
        value=selected_preset["variable"],
        start=0.0, stop=5.0, step=0.1
    )

    floor_pct = mo.ui.number(
        label="Minimum withdrawal floor (%)",
        value=selected_preset["floor"],
        start=0.0, stop=8.0, step=0.1
    )

    mo.vstack([
        fixed_pct,
        variable_pct,
        floor_pct,
        mo.md(f"""
        **Withdrawal Formula:** max({floor_pct.value}%, {fixed_pct.value}% + {variable_pct.value}% × portfolio_value)
        """).callout(kind="info")
    ])
    return fixed_pct, floor_pct, selected_preset, variable_pct


@app.cell
def _(mo):
    # Risk preferences
    mo.md("### Risk Preferences")
    return


@app.cell
def _(mo):
    gamma = mo.ui.slider(
        label="Risk aversion (gamma)",
        start=0.5, stop=5.0, value=1.0, step=0.1, show_value=True
    )

    mo.vstack([
        gamma,
        mo.md("""
        **Risk Aversion Guide:**
        - 0.5-1.0: Low risk aversion (risk-seeking to neutral)
        - 1.0-2.0: Moderate risk aversion
        - 2.0-5.0: High risk aversion (very risk-averse)
        """).callout(kind="info")
    ])
    return gamma,


@app.cell
def _(mo):
    # Control buttons
    run_simulation = mo.ui.button(label="🚀 Run SWR Simulation")
    reset_inputs = mo.ui.button(label="🔄 Reset to Defaults")

    mo.hstack([run_simulation, reset_inputs], justify="center")
    return reset_inputs, run_simulation


@app.cell
def _(
    SWRsimulationCE,
    accept_disclaimer,
    data_loaded,
    fixed_pct,
    floor_pct,
    gamma,
    mo,
    np,
    retirement_years,
    returns_df,
    run_simulation,
    stock_allocation,
    variable_pct
):
    # Simulation execution
    simulation_results = None
    error_message = None

    if not accept_disclaimer.value:
        mo.md("⚠️ Please accept the disclaimer to run simulations").callout(kind="warn")
    elif not data_loaded:
        mo.md("❌ Historical data not available").callout(kind="danger")
    elif run_simulation.value and run_simulation.value > 0:
        try:
            # Create SWR simulation
            mo.md("🔄 Running SWR simulation...").callout(kind="info")

            bond_allocation = 100 - stock_allocation.value

            s = SWRsimulationCE({
                'simulation': {
                    'returns_df': returns_df,
                    'n_ret_years': retirement_years.value,
                },
                'allocation': {
                    'asset_weights': np.array([stock_allocation.value / 100, bond_allocation / 100])
                },
                'withdrawal': {
                    'fixed_pct': fixed_pct.value,
                    'variable_pct': variable_pct.value,
                    'floor_pct': floor_pct.value,
                },
                'evaluation': {
                    'gamma': gamma.value
                },
            })

            s.simulate()
            simulation_results = s

            mo.md("✅ Simulation completed successfully!").callout(kind="success")

        except Exception as e:
            error_message = str(e)
            mo.md(f"❌ Simulation error: {error_message}").callout(kind="danger")
    else:
        mo.md("👆 Configure your parameters above and click 'Run SWR Simulation'").callout(kind="info")

    return bond_allocation, error_message, s, simulation_results


@app.cell
def _(mo, simulation_results):
    # Display results
    if simulation_results:
        mo.md("## 📊 Simulation Results")
    else:
        ""
    return


@app.cell
def _(mo, simulation_results):
    # Results table
    if simulation_results:
        metrics_df = simulation_results.table_metrics()

        # Format the metrics for better display
        formatted_metrics = []
        for _, row in metrics_df.iterrows():
            metric_name = row['metric']
            value = row['value']

            if 'spending' in metric_name.lower():
                formatted_value = f"{value:.2f}%"
            elif 'portfolio' in metric_name.lower():
                formatted_value = f"${value:,.0f}"
            elif '%' in metric_name:
                formatted_value = f"{value:.1f}%"
            else:
                formatted_value = f"{value:.2f}"

            formatted_metrics.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Value': formatted_value
            })

        results_df = pd.DataFrame(formatted_metrics)
        mo.ui.table(results_df, label="Key Metrics")
    else:
        ""
    return formatted_metrics, formatted_value, metrics_df, results_df, row, value


@app.cell
def _(mo, simulation_results):
    # Summary insights
    if simulation_results:
        metrics = simulation_results.table_metrics()
        mean_spending = metrics.iloc[0]['value']  # Mean annual spending
        exhaustion_rate = metrics.iloc[4]['value']  # Exhaustion rate
        min_portfolio = metrics.iloc[3]['value']   # Minimum ending portfolio

        mo.md(f"""
        ### 🎯 Key Insights

        - **Average Annual Spending:** {mean_spending:.2f}% of initial portfolio
        - **Portfolio Failure Rate:** {exhaustion_rate:.1f}% of historical scenarios
        - **Worst-Case Ending Balance:** ${min_portfolio:,.0f} per $100k initial

        {"🟢 This withdrawal strategy has been historically sustainable" if exhaustion_rate == 0 else f"🟡 {exhaustion_rate:.1f}% of scenarios would have exhausted the portfolio"}
        """).callout(kind="success" if exhaustion_rate < 5 else "warn")
    else:
        ""
    return exhaustion_rate, mean_spending, metrics, min_portfolio


@app.cell
def _(mo):
    mo.md("""
    ## 💡 Understanding Safe Withdrawal Rates

    **What is SWR?** The Safe Withdrawal Rate is the percentage of your retirement portfolio
    you can withdraw annually with a high probability of not running out of money.

    **Key Concepts:**
    - **Fixed Withdrawal:** A constant percentage regardless of portfolio performance
    - **Variable Withdrawal:** Adjusts based on current portfolio value
    - **Floor:** Minimum withdrawal amount to maintain lifestyle

    **Historical Context:** This analysis uses real market returns from 1928-2020,
    including the Great Depression, multiple recessions, and various market cycles.
    """)
    return


if __name__ == "__main__":
    app.run()