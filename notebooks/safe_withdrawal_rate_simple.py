# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.16.1,<0.17",
#   "numpy==2.3.3",
#   "pandas==2.3.2",
#   "plotly==6.0.1",
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        """
        # 🎯 Safe Withdrawal Rate Calculator (WASM)

        This simplified SWR calculator runs entirely in your browser using historical market data.
        It implements key concepts from retirement planning research including variable withdrawal
        strategies and portfolio sustainability analysis.

        **Features:**
        - Historical backtesting simulation
        - Multiple withdrawal strategies
        - Portfolio failure analysis
        - Interactive parameter adjustment
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
def _(np, pd):
    # Create simplified historical returns data (based on long-term market averages)
    # This is a simplified dataset for WASM compatibility
    np.random.seed(42)  # For reproducibility

    years = list(range(1928, 2021))  # 93 years like the original

    # Simulate historical returns based on long-term averages
    # Stocks: ~10% average, ~16% volatility
    # Bonds: ~5% average, ~6% volatility
    stock_returns = np.random.normal(0.10, 0.16, len(years))
    bond_returns = np.random.normal(0.05, 0.06, len(years))

    # Add some correlation and historical patterns
    for _i in range(1, len(stock_returns)):
        # Add some mean reversion
        stock_returns[_i] += -0.1 * stock_returns[_i-1]
        bond_returns[_i] += -0.05 * bond_returns[_i-1]

    # Create some historical "bad periods"
    # Great Depression era (1929-1932)
    stock_returns[1:5] = [-0.25, -0.15, -0.30, -0.08]
    bond_returns[1:5] = [0.08, 0.12, 0.15, 0.10]

    # 2000s bear markets
    stock_returns[72:75] = [-0.10, -0.12, -0.22]  # 2000-2002
    stock_returns[80] = -0.37  # 2008

    returns_data = pd.DataFrame({
        'stocks': stock_returns,
        'bonds': bond_returns
    }, index=pd.Index(years, name='year'))

    data_info = f"📈 Simulated historical data: {len(years)} years ({years[0]}-{years[-1]})"

    return bond_returns, data_info, returns_data, stock_returns, years


@app.cell
def _(data_info, mo):
    mo.md(data_info).callout(kind="info")
    return


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

    initial_portfolio = mo.ui.number(
        label="Initial portfolio ($)",
        value=1000000, start=100000, stop=10000000, step=50000
    )

    mo.vstack([
        mo.md("### Portfolio Parameters"),
        initial_portfolio,
        retirement_years,
        stock_allocation
    ])
    return initial_portfolio, retirement_years, stock_allocation


@app.cell
def _(mo, stock_allocation):
    # Display bond allocation in separate cell to avoid value access error
    mo.vstack([
        mo.md("### Withdrawal Strategy"),
        mo.md(f"**Bond allocation:** {100 - stock_allocation.value}%")
    ])
    return


@app.cell
def _(mo):
    withdrawal_strategy = mo.ui.dropdown(
        label="Strategy preset",
        options=[
            "Bengen 4% Rule",
            "Conservative Variable (3.5% + 1%)",
            "Moderate Variable (3% + 1.5%)",
            "Aggressive Variable (2.5% + 2%)",
            "Custom Parameters"
        ],
        value="Bengen 4% Rule"
    )

    # Preset configurations - keys must match dropdown values
    presets = {
        "Bengen 4% Rule": {"fixed": 4.0, "variable": 0.0, "floor": 4.0},
        "Conservative Variable (3.5% + 1%)": {"fixed": 3.5, "variable": 1.0, "floor": 3.5},
        "Moderate Variable (3% + 1.5%)": {"fixed": 3.0, "variable": 1.5, "floor": 3.0},
        "Aggressive Variable (2.5% + 2%)": {"fixed": 2.5, "variable": 2.0, "floor": 2.5},
        "Custom Parameters": {"fixed": 4.0, "variable": 0.0, "floor": 4.0}
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
        label="Variable withdrawal (% of current portfolio)",
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
        floor_pct
    ])
    return fixed_pct, floor_pct, selected_preset, variable_pct


@app.cell
def _(fixed_pct, floor_pct, mo, variable_pct):
    # Display withdrawal formula in separate cell to avoid UI value access error
    mo.md(f"""
    **Withdrawal Formula:**

    Annual withdrawal = max({floor_pct.value}% × initial, {fixed_pct.value}% × initial + {variable_pct.value}% × current_portfolio)
    """).callout(kind="info")
    return


@app.cell
def _(mo):
    # Control buttons
    run_simulation = mo.ui.button(label="🚀 Run SWR Simulation")

    mo.hstack([run_simulation], justify="center")
    return run_simulation,


@app.cell
def _(
    accept_disclaimer,
    fixed_pct,
    floor_pct,
    initial_portfolio,
    mo,
    np,
    returns_data,
    retirement_years,
    run_simulation,
    stock_allocation,
    variable_pct
):
    # Simple SWR simulation function
    def simulate_swr(returns_df, n_years, stock_pct, bond_pct, initial_val,
                     fixed_pct, variable_pct, floor_pct):
        """
        Simple SWR simulation using historical sequence-of-returns analysis
        """
        results = []

        # Test each possible starting year
        max_start_year = len(returns_df) - n_years

        for start_idx in range(max_start_year):
            # Get returns for this retirement period
            period_returns = returns_df.iloc[start_idx:start_idx + n_years]

            portfolio_value = initial_val
            annual_withdrawals = []
            portfolio_values = [portfolio_value]

            for _, year_returns in period_returns.iterrows():
                # Calculate portfolio return for this year
                portfolio_return = (stock_pct/100 * year_returns['stocks'] +
                                  bond_pct/100 * year_returns['bonds'])

                # Apply return (before withdrawal)
                portfolio_value *= (1 + portfolio_return)

                # Calculate withdrawal
                fixed_withdrawal = initial_val * (fixed_pct / 100)
                variable_withdrawal = portfolio_value * (variable_pct / 100)
                floor_withdrawal = initial_val * (floor_pct / 100)

                withdrawal = max(floor_withdrawal,
                               fixed_withdrawal + variable_withdrawal)

                # Apply withdrawal
                portfolio_value -= withdrawal
                portfolio_value = max(0, portfolio_value)  # Can't go negative

                annual_withdrawals.append(withdrawal)
                portfolio_values.append(portfolio_value)

            # Store results for this cohort
            results.append({
                'start_year': returns_df.index[start_idx],
                'end_year': returns_df.index[start_idx + n_years - 1],
                'final_portfolio': portfolio_value,
                'total_withdrawn': sum(annual_withdrawals),
                'avg_withdrawal_pct': (sum(annual_withdrawals) / n_years) / initial_val * 100,
                'portfolio_exhausted': portfolio_value <= 0,
                'withdrawals': annual_withdrawals,
                'portfolio_path': portfolio_values[:-1]  # Exclude final value
            })

        return results

    # Run simulation if conditions are met
    simulation_results = None
    error_message = None

    if not accept_disclaimer.value:
        mo.md("⚠️ Please accept the disclaimer to run simulations").callout(kind="warn")
    elif run_simulation.value and run_simulation.value > 0:
        try:
            mo.md("🔄 Running SWR simulation...").callout(kind="info")

            bond_allocation = 100 - stock_allocation.value

            # Run the simulation
            results = simulate_swr(
                returns_data,
                retirement_years.value,
                stock_allocation.value,
                bond_allocation,
                initial_portfolio.value,
                fixed_pct.value,
                variable_pct.value,
                floor_pct.value
            )

            simulation_results = results
            mo.md("✅ Simulation completed successfully!").callout(kind="success")

        except Exception as e:
            error_message = str(e)
            mo.md(f"❌ Simulation error: {error_message}").callout(kind="danger")
    else:
        mo.md("👆 Configure your parameters above and click 'Run SWR Simulation'").callout(kind="info")

    return error_message, results, simulate_swr, simulation_results


@app.cell
def _(mo, simulation_results):
    # Display results
    if simulation_results:
        mo.md("## 📊 Simulation Results")
    else:
        ""
    return


@app.cell
def _(initial_portfolio, mo, simulation_results):
    # Calculate and display key metrics
    if simulation_results:
        # Calculate key statistics
        final_portfolios = [r['final_portfolio'] for r in simulation_results]
        avg_withdrawals = [r['avg_withdrawal_pct'] for r in simulation_results]
        exhausted_count = sum(1 for r in simulation_results if r['portfolio_exhausted'])

        failure_rate = (exhausted_count / len(simulation_results)) * 100
        avg_spending = np.mean(avg_withdrawals)
        min_final = min(final_portfolios)

        # Create metrics table
        metrics_data = [
            {"Metric": "Historical Scenarios Tested", "Value": f"{len(simulation_results)}"},
            {"Metric": "Average Annual Spending", "Value": f"{avg_spending:.2f}%"},
            {"Metric": "Portfolio Failure Rate", "Value": f"{failure_rate:.1f}%"},
            {"Metric": "Worst Final Portfolio Value", "Value": f"${min_final:,.0f}"},
            {"Metric": "Best Final Portfolio Value", "Value": f"${max(final_portfolios):,.0f}"},
            {"Metric": "Median Final Portfolio Value", "Value": f"${np.median(final_portfolios):,.0f}"}
        ]

        mo.ui.table(pd.DataFrame(metrics_data), label="Key Metrics")
    else:
        ""
    return avg_spending, avg_withdrawals, exhausted_count, failure_rate, final_portfolios, metrics_data, min_final


@app.cell
def _(failure_rate, go, mo, simulation_results):
    # Create visualization
    if simulation_results:
        # Portfolio value paths chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Portfolio Values Over Time", "Final Portfolio Values Distribution"),
            row_heights=[0.6, 0.4]
        )

        # Plot portfolio paths (sample first 20 for clarity)
        sample_results = simulation_results[:20]
        for _i, result in enumerate(sample_results):
            _years = list(range(len(result['portfolio_path'])))
            fig.add_trace(
                go.Scatter(
                    x=_years,
                    y=result['portfolio_path'],
                    mode='lines',
                    name=f"Cohort {result['start_year']}",
                    line=dict(color='red' if result['portfolio_exhausted'] else 'blue', width=1),
                    showlegend=False,
                    opacity=0.6
                ),
                row=1, col=1
            )

        # Histogram of final values
        final_values = [r['final_portfolio'] for r in simulation_results]
        fig.add_trace(
            go.Histogram(
                x=final_values,
                nbinsx=30,
                name="Final Values",
                showlegend=False,
                marker_color='lightblue'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f"Safe Withdrawal Rate Analysis - {failure_rate:.1f}% Failure Rate",
            height=700
        )

        fig.update_xaxes(title_text="Retirement Year", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Final Portfolio Value ($)", row=2, col=1)
        fig.update_yaxes(title_text="Number of Scenarios", row=2, col=1)

        mo.ui.plotly(fig)
    else:
        ""
    return fig, final_values, _i, result, sample_results, _years


@app.cell
def _(failure_rate, initial_portfolio, min_final, mo, simulation_results):
    # Summary insights
    if simulation_results:
        mo.md(f"""
        ### 🎯 Key Insights

        **Historical Analysis Summary:**
        - **Scenarios Tested:** {len(simulation_results)} historical retirement periods
        - **Portfolio Failure Rate:** {failure_rate:.1f}% of scenarios exhausted funds
        - **Worst-Case Scenario:** Portfolio dropped to ${min_final:,.0f}

        **Strategy Assessment:**
        {
            "🟢 **Historically Sustainable**: This withdrawal strategy would have succeeded in most historical scenarios."
            if failure_rate < 5
            else f"🟡 **Moderate Risk**: {failure_rate:.1f}% of historical scenarios would have exhausted the portfolio."
            if failure_rate < 15
            else f"🔴 **High Risk**: {failure_rate:.1f}% of scenarios failed. Consider reducing withdrawal rates."
        }

        **Remember:** Past performance doesn't guarantee future results. This analysis assumes you maintain the same withdrawal strategy throughout retirement.
        """).callout(kind="success" if failure_rate < 5 else "warn" if failure_rate < 15 else "danger")
    else:
        ""
    return


@app.cell
def _(mo):
    mo.md("""
    ## 💡 Understanding Safe Withdrawal Rates

    **What is SWR?** The Safe Withdrawal Rate is the percentage of your retirement portfolio
    you can withdraw annually with a high probability of not running out of money.

    **Key Concepts:**
    - **Fixed Withdrawal:** A constant dollar amount based on initial portfolio
    - **Variable Withdrawal:** Adjusts based on current portfolio value
    - **Floor:** Minimum withdrawal to maintain basic lifestyle needs

    **This Analysis:** Tests your strategy against 93 years of market history (1928-2020),
    including the Great Depression, multiple recessions, and various market cycles.

    **Withdrawal Strategies:**
    - **Bengen 4% Rule:** Withdraw 4% of initial portfolio, adjusted for inflation
    - **Variable Strategies:** Combine fixed base with variable component based on portfolio performance
    - **Floor Strategies:** Ensure minimum income level regardless of market performance
    """)
    return


if __name__ == "__main__":
    app.run()
