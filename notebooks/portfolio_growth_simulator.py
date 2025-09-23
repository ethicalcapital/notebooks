# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "marimo>=0.10.12,<0.11",
#   "numpy==2.3.3",
#   "pandas==2.3.2",
#   "plotly==6.3.0",
# ]
# ///

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo
    mo = marimo
    import numpy as np
    import pandas as pd
    import warnings

    warnings.filterwarnings("ignore")
    return mo, np, pd


@app.cell
def _(mo):
    mo.md(
        """
    # 🎯 Portfolio Growth Simulator with Monte Carlo Analysis

    This interactive tool helps you visualize how your investment portfolio might grow over time,
    considering the sequence of returns risk through Monte Carlo simulation.

    ## 📊 Configure Your Portfolio Below
    """
    )
    return


@app.cell
def _(
    asset_params,
    growth_pct,
    diversified_pct,
    income_pct,
    current_value,
    annual_savings,
    years_to_simulate,
    num_simulations,
    random_seed,
    base_fee_all,
    sub_fee_diversified,
    sub_fee_income,
    use_regime,
    regime_mode,
    fixed_regime,
    p_stay_bull,
    p_stay_normal,
    p_stay_bear,
    sampling_method,
    return_dist,
    t_df,
    block_len,
    withdrawal_start_year,
    withdrawal_rate,
    inflation_rate,
    bridge_enable,
    bridge_start_years,
    bridge_duration_years,
    bridge_rate,
    total_allocation,
):
    cfg = {
        "years": int(years_to_simulate.value),
        "n_sims": int(num_simulations.value),
        "initial": float(current_value.value),
        "monthly": float(annual_savings.value) / 12.0,
        "seed": int(random_seed.value) if random_seed.value is not None else 42,
        "weights": {
            "growth": float(growth_pct) / 100.0,
            "diversified": float(diversified_pct) / 100.0,
            "income": float(income_pct) / 100.0,
        },
        "fees": {
            "base": float(base_fee_all.value) / 100.0,
            "sub_div": float(sub_fee_diversified.value) / 100.0,
            "sub_inc": float(sub_fee_income.value) / 100.0,
        },
        "regimes": {
            "use": bool(use_regime.value),
            "mode": str(regime_mode.value),
            "fixed": str(fixed_regime.value),
            "p_stay": {
                "bull": float(p_stay_bull.value),
                "normal": float(p_stay_normal.value),
                "bear": float(p_stay_bear.value),
            },
        },
        "sampling": {
            "method": str(sampling_method.value),
            "dist": str(return_dist.value),
            "t_df": int(t_df.value),
            "block_len": int(block_len.value),
        },
        "withdrawals": {
            "start_year": int(withdrawal_start_year.value),
            "rate": float(withdrawal_rate.value) / 100.0,
            "bridge": {
                "enable": bool(bridge_enable.value),
                "start": int(bridge_start_years.value),
                "duration": int(bridge_duration_years.value),
                "rate": float(bridge_rate.value) / 100.0,
            },
        },
        "inflation": float(inflation_rate.value) / 100.0,
    }
    err = None
    # Basic validation (UI-only). Full gating stays in compute cell.
    if total_allocation != 100:
        err = "Allocations must sum to 100%."
    elif cfg["years"] <= 0:
        err = "Years to simulate must be at least 1."
    elif cfg["n_sims"] <= 0:
        err = "Number of simulations must be positive."
    elif cfg["initial"] < 0:
        err = "Initial investment cannot be negative."
    elif cfg["monthly"] < 0:
        err = "Monthly contribution cannot be negative."
    if cfg["seed"] < 0:
        cfg["seed"] = abs(cfg["seed"])
    if cfg["sampling"]["dist"] == "Student t" and cfg["sampling"]["t_df"] < 3:
        cfg["sampling"]["t_df"] = 3
    if cfg["sampling"]["block_len"] < 1:
        cfg["sampling"]["block_len"] = 1

    assets = {k: {"mean": float(v["mean_return"]), "vol": float(v["volatility"])} for k, v in asset_params.items()}
    return cfg, assets, err


@app.cell
def _():
    from functools import lru_cache
    import json as _json
    import numpy as _np
    import pandas as _pd

    @lru_cache(maxsize=64)
    def simulate_portfolio_cached(config_json: str, assets_json: str):
        cfg = _json.loads(config_json)
        assets = _json.loads(assets_json)

        weights = cfg["weights"]
        p_return = sum(weights[a] * assets[a]["mean"] for a in weights)
        p_vol = (_np.sqrt(sum((weights[a] * assets[a]["vol"]) ** 2 for a in weights)))

        years = cfg["years"]
        n_sims = cfg["n_sims"]
        initial = cfg["initial"]
        monthly = cfg["monthly"]
        seed = cfg["seed"]

        months = years * 12
        results = _np.zeros((n_sims, months + 1), dtype=float)
        results[:, 0] = initial
        _np.random.seed(seed)

        monthly_fee_rate = cfg["fees"]["base"] / 12.0
        monthly_fee_rate += weights["diversified"] * (cfg["fees"]["sub_div"]) / 12.0
        monthly_fee_rate += weights["income"] * (cfg["fees"]["sub_inc"]) / 12.0

        regime_multipliers = {
            "Bull": {"mean": 1.25, "vol": 0.85},
            "Normal": {"mean": 1.00, "vol": 1.00},
            "Bear": {"mean": 0.60, "vol": 1.30},
        }

        base_m_mean = p_return / 12.0
        base_m_sigma = p_vol / _np.sqrt(12.0)

        use_reg = cfg["regimes"]["use"]
        reg_mode = cfg["regimes"]["mode"]
        fixed = cfg["regimes"]["fixed"]
        p_bull = cfg["regimes"]["p_stay"]["bull"]
        p_norm = cfg["regimes"]["p_stay"]["normal"]
        p_bear = cfg["regimes"]["p_stay"]["bear"]

        samp_method = cfg["sampling"]["method"]
        dist_choice = cfg["sampling"]["dist"]
        dfv = cfg["sampling"]["t_df"]
        block = cfg["sampling"]["block_len"]

        infl = cfg["inflation"]
        start_w = cfg["withdrawals"]["start_year"]
        w_rate = cfg["withdrawals"]["rate"]
        bridge = cfg["withdrawals"]["bridge"]

        for sim in range(n_sims):
            pv = initial

            if use_reg:
                if reg_mode == "Fixed":
                    regimes = _np.array([fixed] * months, dtype=object)
                else:
                    regimes = _np.empty(months, dtype=object)
                    regimes[0] = "Normal"
                    for m in range(1, months):
                        prev = regimes[m - 1]
                        if prev == "Bull":
                            stay = float(p_bull); others = (1.0 - stay) / 2.0; probs = [stay, others, others]
                        elif prev == "Normal":
                            stay = float(p_norm); others = (1.0 - stay) / 2.0; probs = [others, stay, others]
                        else:
                            stay = float(p_bear); others = (1.0 - stay) / 2.0; probs = [others, others, stay]
                        regimes[m] = _np.random.choice(["Bull", "Normal", "Bear"], p=probs)
            else:
                regimes = _np.array(["Normal"] * months, dtype=object)

            returns_path = None
            if samp_method == "Block bootstrap":
                ref_len = max(months, 360)
                ref_states = _np.empty(ref_len, dtype=object)
                ref_states[0] = regimes[0]
                for m in range(1, ref_len):
                    prev = ref_states[m - 1]
                    if prev == "Bull":
                        stay = float(p_bull); others = (1.0 - stay) / 2.0; probs = [stay, others, others]
                    elif prev == "Normal":
                        stay = float(p_norm); others = (1.0 - stay) / 2.0; probs = [others, stay, others]
                    else:
                        stay = float(p_bear); others = (1.0 - stay) / 2.0; probs = [others, others, stay]
                    ref_states[m] = _np.random.choice(["Bull", "Normal", "Bear"], p=probs)
                ref_returns = _np.zeros(ref_len)
                for m in range(ref_len):
                    st = ref_states[m]
                    mu = base_m_mean * regime_multipliers[st]["mean"]
                    sig = max(1e-12, base_m_sigma * regime_multipliers[st]["vol"])
                    if dist_choice == "Student t":
                        tdf = max(3, int(dfv))
                        z = _np.random.standard_t(tdf)
                        z = z * _np.sqrt((tdf - 2) / tdf)
                        ref_returns[m] = mu + z * sig
                    else:
                        ref_returns[m] = _np.random.normal(mu, sig)
                b = max(1, int(block))
                chunks = []
                need = months
                while need > 0:
                    start = _np.random.randint(0, ref_len)
                    end = start + b
                    if end <= ref_len:
                        chunk = ref_returns[start:end]
                    else:
                        wrap = end - ref_len
                        chunk = _np.concatenate([ref_returns[start:], ref_returns[:wrap]])
                    chunks.append(chunk)
                    need -= b
                returns_path = _np.concatenate(chunks)[:months]

            for month in range(1, months + 1):
                year = month // 12
                st = regimes[month - 1]
                if returns_path is not None:
                    mret = returns_path[month - 1]
                else:
                    mu = base_m_mean * regime_multipliers[st]["mean"]
                    sig = max(1e-12, base_m_sigma * regime_multipliers[st]["vol"])
                    if dist_choice == "Student t":
                        tdf = max(3, int(dfv))
                        z = _np.random.standard_t(tdf)
                        z = z * _np.sqrt((tdf - 2) / tdf)
                        mret = mu + z * sig
                    else:
                        mret = _np.random.normal(mu, sig)

                pv *= (1.0 + mret)
                if monthly_fee_rate > 0:
                    pv *= max(0.0, 1.0 - monthly_fee_rate)

                in_bridge = bool(bridge["enable"]) and (int(bridge["start"]) <= year < int(bridge["start"] + bridge["duration"]))
                if in_bridge:
                    annual_withdrawal_b = max(0.0, pv) * float(bridge["rate"]) 
                    pv = max(0.0, pv - (annual_withdrawal_b / 12.0))
                else:
                    if year < int(start_w):
                        inflated_contrib = monthly * ((1.0 + infl) ** (month / 12.0))
                        pv += max(0.0, inflated_contrib)
                    if year >= int(start_w):
                        annual_w = max(0.0, pv) * float(w_rate)
                        pv = max(0.0, pv - (annual_w / 12.0))

                results[sim, month] = pv

        time_index = _np.arange(0, months + 1) / 12.0
        percentiles = {
            "5th": _np.percentile(results, 5, axis=0),
            "25th": _np.percentile(results, 25, axis=0),
            "50th": _np.percentile(results, 50, axis=0),
            "75th": _np.percentile(results, 75, axis=0),
            "95th": _np.percentile(results, 95, axis=0),
        }
        df_results = _pd.DataFrame({"Year": time_index, **percentiles, "Mean": _np.mean(results, axis=0)})
        sample_size = min(50, n_sims)
        sample_indices = _np.random.choice(n_sims, sample_size, replace=False)
        sample_paths = results[sample_indices, :]
        return {
            "df_results": df_results,
            "sample_paths": sample_paths,
            "all_results": results,
            "portfolio_return": p_return,
            "portfolio_vol": p_vol,
            "time_index": time_index,
            "initial_value": initial,
            "months": months,
        }

    def simulate_portfolio(cfg: dict, assets: dict):
        # Dump to json once in this cell to standardize keys for caching
        c = _json.dumps(cfg, sort_keys=True)
        a = _json.dumps(assets, sort_keys=True)
        return simulate_portfolio_cached(c, a)

    return simulate_portfolio

@app.cell
def _(mo):
    # Master disclaimer content (rendered later in a centralized control panel)
    disclaimer_md = mo.md(
        """
        **DISCLAIMER: FOR ILLUSTRATIVE PURPOSES ONLY**

        This tool provides hypothetical calculations and does not constitute investment advice.
        Using this tool does not create a client relationship with Invest Vegan LLC dba Ethical Capital.

        **KEY LIMITATIONS:**
        - Projections are mathematical illustrations, not predictions
        - Results depend on assumptions that may not match reality
        - Past performance does not guarantee future results
        - Cannot consider your complete financial situation
        - Not personalized investment, tax, or legal advice

        **INVESTMENT RISKS:**
        All investments risk loss of principal. Actual results will vary significantly from projections.

        By using this tool, you accept full responsibility for your financial decisions.
        Investment advisory services offered only through formal agreements with
        Invest Vegan LLC dba Ethical Capital, a registered investment adviser.

        For actual financial planning, consult an actual financial planner.

        © 2025 Invest Vegan LLC dba Ethical Capital
        """
    ).callout(kind="danger")

    accept_disclaimer = mo.ui.switch(
        value=False,
        label="I understand and agree to the disclaimer",
    )

    # Do not render here; the main layout cell will place this appropriately
    return accept_disclaimer, disclaimer_md


@app.cell
def _(
    boxplot_step_years,
    inflation_rate,
    mo,
    show_real,
    simulation_results,
):
    # Per-year distribution boxplots to clarify spread over time
    fig_boxes = None
    if simulation_results:
        import plotly as _plotly
        _go = _plotly.graph_objects
        step = max(1, int(boxplot_step_years.value))
        months = simulation_results["months"]
        total_years_box = int(months // 12)
        years_to_show = list(range(0, total_years_box + 1, step))
        if years_to_show[-1] != total_years_box:
            years_to_show.append(total_years_box)

        infl_box = inflation_rate.value / 100.0
        fig_boxes = _go.Figure()

        for y in years_to_show:
            box_month_idx = int(y * 12)
            vals = simulation_results["all_results"][:, box_month_idx]
            if show_real.value:
                vals = vals / ((1.0 + infl_box) ** (y))
            fig_boxes.add_trace(
                _go.Box(
                    y=vals,
                    name=f"Year {y}",
                    boxmean=True,
                    jitter=0,
                    pointpos=0,
                    marker_color="rgba(100,149,237,0.6)",
                    line=dict(color="rgba(100,149,237,1.0)"),
                    hovertemplate="$%{y:,.0f}",
                )
            )

        fig_boxes.update_layout(
            title="Per-Year Distribution of Possible Outcomes (Boxplots)",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Year",
            yaxis=dict(tickformat="$,.0f"),
            template="plotly_white",
            height=420,
            showlegend=False,
        )

        mo.ui.plotly(fig_boxes)
    return


@app.cell
def _(inflation_rate, mo, pd, show_real, simulation_results, _base64):
    # CSV export options
    df_percentiles = None
    df_finals = None
    df_paths = None
    if simulation_results:
        infl_csv = inflation_rate.value / 100.0
        time_index_csv = simulation_results["time_index"]

        # Percentiles DataFrame
        df_percentiles_local = simulation_results["df_results"].copy()
        if show_real.value:
            deflator_csv = (1.0 + infl_csv) ** time_index_csv
            for _col in ["5th", "25th", "50th", "75th", "95th", "Mean"]:
                df_percentiles_local[_col] = df_percentiles_local[_col].values / deflator_csv
        df_percentiles = df_percentiles_local

        # Final values DataFrame
        finals_csv = simulation_results["all_results"][:, -1]
        if show_real.value:
            horizon_years_csv = simulation_results["months"] / 12.0
            finals_csv = finals_csv / ((1.0 + infl_csv) ** horizon_years_csv)
        df_finals = pd.DataFrame({"FinalValue": finals_csv})

        # Sample paths DataFrame (limited)
        paths_arr = simulation_results["sample_paths"]
        if show_real.value:
            deflator_csv = (1.0 + infl_csv) ** time_index_csv
            paths_arr = paths_arr / deflator_csv
        data_csv = {"Year": time_index_csv}
        for path_idx in range(paths_arr.shape[0]):
            data_csv[f"Path_{path_idx+1}"] = paths_arr[path_idx]
        df_paths = pd.DataFrame(data_csv)

        # Download links
        percentiles_csv = df_percentiles.to_csv(index=False).encode("utf-8")
        finals_csv = df_finals.to_csv(index=False).encode("utf-8")
        paths_csv = df_paths.to_csv(index=False).encode("utf-8")

        def make_link(data: bytes, name: str) -> str:
            b64 = _base64.b64encode(data).decode("ascii")
            return f"<a download=\"{name}\" href=\"data:text/csv;base64,{b64}\">{name}</a>"

        link1 = make_link(
            percentiles_csv,
            f"percentiles_{'real' if show_real.value else 'nominal'}.csv",
        )
        link2 = make_link(
            finals_csv,
            f"final_values_{'real' if show_real.value else 'nominal'}.csv",
        )
        link3 = make_link(
            paths_csv,
            f"sample_paths_{'real' if show_real.value else 'nominal'}.csv",
        )

        mo.md(
            f"""
            ## ⬇️ Export CSV

            {mo.hstack([mo.md(link1), mo.md(link2), mo.md(link3)])}
            """
        )
    return


@app.cell
def _(inflation_rate, mo, pd, show_real, simulation_results, _io, _zipfile, _base64):
    # Export all (zip) as a single link
    zip_link = None
    if simulation_results:
        t = simulation_results["time_index"]
        infl = inflation_rate.value / 100.0

        # Percentiles
        dfp = simulation_results["df_results"].copy()
        if show_real.value:
            defl = (1.0 + infl) ** t
            for _col in ["5th", "25th", "50th", "75th", "95th", "Mean"]:
                dfp[_col] = dfp[_col].values / defl
        # Finals
        finals = simulation_results["all_results"][:, -1]
        if show_real.value:
            years_h = simulation_results["months"] / 12.0
            finals = finals / ((1.0 + infl) ** years_h)
        dff = pd.DataFrame({"FinalValue": finals})
        # Paths
        paths = simulation_results["sample_paths"]
        if show_real.value:
            defl = (1.0 + infl) ** t
            paths = paths / defl
        data = {"Year": t}
        for _k in range(paths.shape[0]):
            data[f"Path_{_k+1}"] = paths[_k]
        dfpaths = pd.DataFrame(data)

        # Build zip in memory
        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, mode="w", compression=_zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("percentiles.csv", dfp.to_csv(index=False))
            zf.writestr("final_values.csv", dff.to_csv(index=False))
            zf.writestr("sample_paths.csv", dfpaths.to_csv(index=False))
        b = buf.getvalue()
        b64 = _base64.b64encode(b).decode("ascii")
        name = f"mc_export_{'real' if show_real.value else 'nominal'}.zip"
        zip_link = f"<a download=\"{name}\" href=\"data:application/zip;base64,{b64}\">Download all (zip)</a>"

        mo.md(f"""
        {mo.md(zip_link)}
        """)
    return


@app.cell
def _(mo):
    # Portfolio configuration inputs
    current_value = mo.ui.number(
        value=100000,
        start=0,
        stop=10000000,
        step=1000,
        label="Current Value ($)",
    )

    annual_savings = mo.ui.number(
        value=12000,
        start=0,
        stop=600000,
        step=500,
        label="Annual Savings ($)",
    )

    years_to_simulate = mo.ui.slider(
        start=1,
        stop=50,
        value=30,
        step=1,
        label="Years to Simulate",
        show_value=True,
    )
    return annual_savings, current_value, years_to_simulate


@app.cell
def _(mo):
    # Portfolio allocation by risk level or custom
    risk_mode = mo.ui.dropdown(options=["Risk level", "Custom"], value="Risk level", label="Allocation mode")
    risk_level = mo.ui.radio(options=["1","2","3","4","5"], value="3", label="Risk level (1=conservative, 5=aggressive)")

    # Custom percentages (used when risk_mode == 'Custom')
    custom_growth = mo.ui.number(value=40, start=0, stop=100, step=1, label="Growth (%)")
    custom_diversified = mo.ui.number(value=40, start=0, stop=100, step=1, label="Diversified (%)")
    custom_income = mo.ui.number(value=20, start=0, stop=100, step=1, label="Income (%)")
    return (
        custom_diversified,
        custom_growth,
        custom_income,
        risk_level,
        risk_mode,
    )


@app.cell
def _(
    custom_diversified,
    custom_growth,
    custom_income,
    mo,
    risk_level,
    risk_mode,
):
    # Compute allocations from risk mode/level
    levels = {
        "1": {"growth": 10, "diversified": 40, "income": 50},
        "2": {"growth": 25, "diversified": 45, "income": 30},
        "3": {"growth": 40, "diversified": 40, "income": 20},
        "4": {"growth": 55, "diversified": 35, "income": 10},
        "5": {"growth": 70, "diversified": 25, "income": 5},
    }

    if risk_mode.value == "Risk level":
        sel = levels.get(risk_level.value, levels["3"])
        growth_pct = sel["growth"]
        diversified_pct = sel["diversified"]
        income_pct = sel["income"]
    else:
        growth_pct = float(custom_growth.value)
        diversified_pct = float(custom_diversified.value)
        income_pct = float(custom_income.value)

    total_allocation = growth_pct + diversified_pct + income_pct
    allocation_warning = mo.callout(
        f"Total allocation: {total_allocation}% {'⚠️ Must equal 100%' if total_allocation != 100 else '✅'}",
        kind="warn" if total_allocation != 100 else "success",
    )
    return (
        allocation_warning,
        diversified_pct,
        growth_pct,
        income_pct,
        total_allocation,
    )


@app.cell
def _(mo):
    # Withdrawal configuration
    withdrawal_start_year = mo.ui.slider(
        start=0,
        stop=50,
        value=30,
        step=1,
        label="Years until withdrawal",
        show_value=True,
    )

    withdrawal_rate = mo.ui.slider(
        start=0,
        stop=10,
        value=4,
        step=0.25,
        label="Annual withdrawal rate (%)",
        show_value=True,
    )

    # Optional temporary withdrawal schedule ("bridge")
    bridge_enable = mo.ui.switch(False, label="Add temporary withdrawal period")
    bridge_start_years = mo.ui.slider(start=0, stop=30, value=3, step=1, label="Bridge starts in (years)", show_value=True)
    bridge_duration_years = mo.ui.slider(start=1, stop=10, value=2, step=1, label="Bridge duration (years)", show_value=True)
    bridge_rate = mo.ui.slider(start=0.0, stop=15.0, value=4.0, step=0.25, label="Bridge withdrawal rate (%)", show_value=True)

    inflation_rate = mo.ui.slider(
        start=0,
        stop=6,
        value=2.5,
        step=0.25,
        label="Assumed Inflation Rate (%)",
        show_value=True,
    )

    # Layout is handled in a centralized control panel; don't render here.
    return (
        bridge_duration_years,
        bridge_enable,
        bridge_rate,
        bridge_start_years,
        inflation_rate,
        withdrawal_rate,
        withdrawal_start_year,
    )


@app.cell
def _(mo):
    # Monte Carlo parameters + distribution choices
    num_simulations = mo.ui.slider(
        start=10,
        stop=5000,
        value=500,
        step=10,
        label="Number of Simulations",
        show_value=True,
    )

    random_seed = mo.ui.number(
        value=42,
        start=1,
        stop=2**31 - 1,
        step=1,
        label="Random Seed (for reproducibility)",
    )

    return_dist = mo.ui.dropdown(
        options=["Student t", "Normal"],
        value="Student t",
        label="Return distribution (assumed)",
    )
    t_df = mo.ui.slider(
        start=3,
        stop=30,
        value=6,
        step=1,
        label="Student t degrees of freedom",
        show_value=True,
    )
    return num_simulations, random_seed, return_dist, t_df


@app.cell
def _(mo):
    # Regime model and sampling method
    use_regime = mo.ui.switch(False, label="Enable regime model")
    regime_mode = mo.ui.dropdown(options=["Fixed", "Dynamic"], value="Fixed", label="Regime mode")
    fixed_regime = mo.ui.radio(options=["Normal", "Bull", "Bear"], value="Normal", label="Fixed regime")

    # Persistence sliders for dynamic Markov regime switching
    p_stay_bull = mo.ui.slider(start=0.5, stop=0.99, value=0.9, step=0.01, label="P(stay Bull)", show_value=True)
    p_stay_normal = mo.ui.slider(start=0.5, stop=0.99, value=0.9, step=0.01, label="P(stay Normal)", show_value=True)
    p_stay_bear = mo.ui.slider(start=0.5, stop=0.99, value=0.9, step=0.01, label="P(stay Bear)", show_value=True)

    # Sampling method and block bootstrap settings
    sampling_method = mo.ui.dropdown(options=["Parametric", "Block bootstrap"], value="Parametric", label="Return sampling method")
    block_len = mo.ui.slider(start=3, stop=12, value=3, step=1, label="Block length (months)", show_value=True)
    return (
        block_len,
        fixed_regime,
        p_stay_bear,
        p_stay_bull,
        p_stay_normal,
        regime_mode,
        sampling_method,
        use_regime,
    )


@app.cell
def _(pd):
    # Asset class parameters (historical approximations)
    # Asset Class Parameters (rendered optionally in advanced section)

    # Return and volatility parameters
    asset_params = {
        # Based on provided historical characteristics (see appendix markdown below)
        "growth": {
            "mean_return": 0.12,  # 12% average annual return
            "volatility": 0.25,  # 25% standard deviation
            "name": "25-Stock Concentrated Equity",
        },
        "diversified": {
            "mean_return": 0.09,  # 9% average annual return (GTAA)
            "volatility": 0.10,  # 10% standard deviation
            "name": "Global Tactical Asset Allocation",
        },
        "income": {
            "mean_return": 0.06,  # 6% average annual return
            "volatility": 0.06,  # 6% standard deviation
            "name": "Income (Bond-Heavy Portfolio)",
        },
    }

    # Create a nice display table
    params_df = pd.DataFrame(
        [
            {
                "Asset Class": params["name"],
                "Illustrative Return": f"{params['mean_return']*100:.1f}%",
                "Volatility": f"{params['volatility']*100:.1f}%",
            }
            for params in asset_params.values()
        ]
    )
    return asset_params, params_df


@app.cell
def _(mo):
    # Fees & Costs (no commissions modeled)
    base_fee_all = mo.ui.slider(
        start=0.0,
        stop=3.0,
        value=1.0,
        step=0.05,
        label="Annual Fee – All Strategies (%)",
        show_value=True,
    )
    sub_fee_diversified = mo.ui.slider(
        start=0.0,
        stop=3.0,
        value=0.5,
        step=0.05,
        label="Sub-Manager Fee – Diversified (%)",
        show_value=True,
    )
    sub_fee_income = mo.ui.slider(
        start=0.0,
        stop=3.0,
        value=0.5,
        step=0.05,
        label="Sub-Manager Fee – Income (%)",
        show_value=True,
    )
    return base_fee_all, sub_fee_diversified, sub_fee_income


@app.cell
def _(mo):
    # Mode + Run + Reset
    english_mode = mo.ui.switch(value=True, label="English (simple)")
    run_button = mo.ui.button(label="Run simulation")
    reset_button = mo.ui.button(label="Reset")
    # Export in the same order as created to avoid mis-binding
    return english_mode, run_button, reset_button


@app.cell
def _(mo):
    # Assumptions strip for English mode (rendered later)
    assumptions = mo.callout(
        "Using illustrative defaults: Student t (df=6), parametric sampling, no regimes, base fee 1.0% + sub-fees, monthly fees applied as AUM drag.",
        kind="info",
    )
    return (assumptions,)


@app.cell
def _(mo):
    # Visualization settings and targets (advanced; rendered in control panel)
    show_real = mo.ui.switch(value=False, label="Show real (inflation-adjusted) values")
    log_scale = mo.ui.switch(value=False, label="Log scale (Y axis)")
    show_band_95 = mo.ui.switch(value=True, label="Show 5–95% band")
    show_band_75 = mo.ui.switch(value=True, label="Show 25–75% band")
    path_style = mo.ui.dropdown(options=["Sample", "All (WebGL)", "None"], value="Sample", label="Path overlay")
    paths_max = mo.ui.slider(start=0, stop=1000, value=200, step=10, label="Max paths to draw", show_value=True)
    boxplot_step_years = mo.ui.slider(
        start=1, stop=10, value=5, step=1, label="Boxplot: year step", show_value=True
    )
    target_final_value = mo.ui.number(
        value=1_000_000, start=0, stop=100_000_000, step=50_000, label="Target final value ($)"
    )
    return (
        boxplot_step_years,
        log_scale,
        path_style,
        paths_max,
        show_band_75,
        show_band_95,
        show_real,
        target_final_value,
    )


@app.cell
def _(mo):
    scenario = mo.ui.radio(
        options=["Conservative (25th)", "Median (50th)", "Optimistic (75th)"],
        value="Median (50th)",
        label="Scenario highlight",
    )
    return (scenario,)


@app.cell
def _(
    accept_disclaimer,
    allocation_warning,
    annual_savings,
    assumptions,
    base_fee_all,
    block_len,
    boxplot_step_years,
    bridge_duration_years,
    bridge_enable,
    bridge_rate,
    bridge_start_years,
    current_value,
    custom_diversified,
    custom_growth,
    custom_income,
    disclaimer_md,
    english_mode,
    fixed_regime,
    inflation_rate,
    log_scale,
    mo,
    num_simulations,
    p_stay_bear,
    p_stay_bull,
    p_stay_normal,
    params_df,
    path_style,
    paths_max,
    random_seed,
    regime_mode,
    reset_button,
    return_dist,
    risk_level,
    risk_mode,
    run_button,
    sampling_method,
    scenario,
    show_band_75,
    show_band_95,
    show_real,
    sub_fee_diversified,
    sub_fee_income,
    t_df,
    target_final_value,
    total_allocation,
    use_regime,
    withdrawal_rate,
    withdrawal_start_year,
    years_to_simulate,
):
    # Top control panel: essential inputs only
    _header = mo.md(
        "## 🚀 Get Started\nFill the required fields, read the disclaimer, accept it, then press Run."
    )
    _req_rows = [
        mo.hstack([english_mode]),
        mo.hstack([current_value, annual_savings, years_to_simulate]),
        mo.hstack([risk_mode, risk_level]),
    ]
    # If Custom is chosen, surface the custom sliders inline + warning
    if getattr(risk_mode, "value", "Risk level") == "Custom":
        _req_rows.append(mo.hstack([custom_growth, custom_diversified, custom_income]))
        _req_rows.append(allocation_warning)
    _req_rows.append(mo.hstack([withdrawal_start_year, scenario]))
    _required = mo.vstack(_req_rows)
    _assump = assumptions if english_mode.value else None

    # Quick status helper
    _ready_msgs = []
    if total_allocation != 100:
        _ready_msgs.append("Allocations must sum to 100%.")
    if not accept_disclaimer.value:
        _ready_msgs.append("Please accept the disclaimer.")
    _ready_callout = mo.callout("\n".join(_ready_msgs), kind="warn") if _ready_msgs else mo.callout("Ready to run.", kind="success")

    _disclaimer_block = mo.vstack([
        mo.md("### ⚠️ Disclaimer"),
        disclaimer_md,
        accept_disclaimer,
        _ready_callout,
        mo.hstack([run_button, reset_button]),
        mo.md(f"Run clicks: {int(getattr(run_button, 'value', 0) or 0)} | Reset clicks: {int(getattr(reset_button, 'value', 0) or 0)}"),
        mo.md("Note: In static HTML exports (non‑WASM) the Run button cannot recompute; use the live app (`marimo run ...`) or export with `marimo export html-wasm` for offline interactivity."),
    ])

    # Advanced options in an accordion
    _section_alloc = mo.vstack([
        mo.md("Customize (only used if 'Custom' selected):"),
        mo.hstack([custom_growth, custom_diversified, custom_income]),
        allocation_warning,
    ])

    _section_withdraw = mo.vstack([
        mo.hstack([withdrawal_rate, inflation_rate]),
        mo.vstack([
            bridge_enable,
            mo.hstack([bridge_start_years, bridge_duration_years, bridge_rate]),
        ]),
    ])

    _section_mc = mo.vstack([
        mo.hstack([num_simulations, random_seed, return_dist, t_df]),
        mo.vstack([
            mo.hstack([use_regime, regime_mode, fixed_regime]),
            mo.hstack([p_stay_bull, p_stay_normal, p_stay_bear]),
            mo.hstack([sampling_method, block_len]),
        ]),
    ])

    _section_fees = mo.vstack([
        mo.md("Assumes no commissions. Fees applied monthly as AUM %."),
        mo.hstack([base_fee_all, sub_fee_diversified, sub_fee_income]),
    ])

    _section_view = mo.vstack([
        mo.hstack([show_real, log_scale]),
        mo.hstack([show_band_95, show_band_75, path_style, paths_max]),
        mo.hstack([boxplot_step_years, target_final_value]),
    ])

    _section_params = mo.vstack([
        mo.md("Illustrative asset class parameters"),
        mo.ui.table(data=params_df),
    ])

    try:
        _advanced = mo.accordion(
            {
                "Customize Allocation": _section_alloc,
                "Withdrawals (Advanced)": _section_withdraw,
                "Monte Carlo & Sampling": _section_mc,
                "Fees & Costs": _section_fees,
                "View & Targets": _section_view,
                "Data & Parameters": _section_params,
            }
        )
    except Exception:
        # Fallback if accordion is unavailable
        _advanced = mo.vstack([
            mo.md("### Advanced Options"),
            _section_alloc,
            _section_withdraw,
            _section_mc,
            _section_fees,
            _section_view,
            _section_params,
        ])
    panel_ui = mo.vstack([
        _header,
        _required,
        _assump if _assump is not None else mo.md(""),
        _disclaimer_block,
        _advanced,
    ])
    panel_ui
    return


@app.cell
def _(
    mo,
    run_button,
    reset_button,
    accept_disclaimer,
    total_allocation,
    error_msg,
    simulation_results,
    err,
):
    # Status panel just below controls, using marimo.status when available
    def _status(kind: str, text: str):
        st = getattr(mo, "status", None)
        if st is not None:
            fn = getattr(st, kind, None)
            if callable(fn):
                return fn(text)
        # Fallback to callouts
        kind_map = {"success": "success", "error": "danger", "warning": "warn", "info": "info"}
        return mo.callout(text, kind=kind_map.get(kind, "info"))

    clicks = int(getattr(run_button, "value", 0) or 0)
    resets = int(getattr(reset_button, "value", 0) or 0)
    can_run = bool(accept_disclaimer.value) and (total_allocation == 100)
    running = (clicks > resets) and can_run and (not simulation_results) and (not error_msg)

    # Include config validation error if present
    _cfg_err = err

    if not accept_disclaimer.value:
        _status("info", "Please accept the disclaimer to enable Run.")
    elif total_allocation != 100:
        _status("warning", "Allocations must sum to 100%.")
    elif _cfg_err:
        _status("warning", _cfg_err)
    elif running:
        _status("spinner", "Running simulation...")
    elif error_msg:
        _status("error", error_msg)
    elif simulation_results:
        try:
            _status(
                "success",
                f"Done: {simulation_results['all_results'].shape[0]} runs, {simulation_results['months']/12:.1f} years.",
            )
        except Exception:
            _status("success", "Done.")
    else:
        _status("info", "Press Run to compute.")
    return


@app.cell
def _(
    mo,
    run_button,
    reset_button,
    accept_disclaimer,
    total_allocation,
):
    # Stable run token: increments only when a legitimate Run occurs
    token = mo.state(0)
    try:
        rc = int(getattr(run_button, "value", 0) or 0)
        rz = int(getattr(reset_button, "value", 0) or 0)
        if accept_disclaimer.value and total_allocation == 100 and rc > rz:
            # Use rc to avoid repeated sets for the same click count
            if token.value != rc:
                token.set(rc)
    except Exception:
        pass
    return (token,)


@app.cell
def _(
    mo,
    run_button,
    reset_button,
    token,
    accept_disclaimer,
    total_allocation,
):
    # Debug strip: make gating state explicit under the status panel
    _rc = int(getattr(run_button, "value", 0) or 0)
    _rz = int(getattr(reset_button, "value", 0) or 0)
    _ready = bool(accept_disclaimer.value) and (total_allocation == 100)
    mo.md(
        f"Ready: {_ready} | Run clicks: {_rc} | Reset clicks: {_rz} | Token: {int(getattr(token,'value',0) or 0)}"
    ).callout(kind="info")
    return


@app.cell
def _(
    accept_disclaimer,
    annual_savings,
    asset_params,
    base_fee_all,
    block_len,
    bridge_duration_years,
    bridge_enable,
    bridge_rate,
    bridge_start_years,
    current_value,
    diversified_pct,
    english_mode,
    fixed_regime,
    growth_pct,
    income_pct,
    inflation_rate,
    np,
    num_simulations,
    p_stay_bear,
    p_stay_bull,
    p_stay_normal,
    pd,
    random_seed,
    regime_mode,
    reset_button,
    return_dist,
    run_button,
    sampling_method,
    sub_fee_diversified,
    sub_fee_income,
    t_df,
    total_allocation,
    use_regime,
    withdrawal_rate,
    withdrawal_start_year,
    years_to_simulate,
    cfg,
    assets,
    simulate_portfolio,
    token,
):
    # New gating via stable run token; ignore direct button values here
    error_msg = None
    simulation_results = None
    if not accept_disclaimer.value or total_allocation != 100:
        # Not ready or not requested: publish names as None
        pass
    else:
        try:
            # token value change indicates a new run request
            _ = token.value  # depend on token only; value unused otherwise
            simulation_results = simulate_portfolio(cfg, assets)
        except Exception as e:
            error_msg = f"Simulation error: {e}"
            simulation_results = None
    return error_msg, simulation_results


@app.cell
def _(
    brand,
    english_mode,
    error_msg,
    inflation_rate,
    log_scale,
    mo,
    np,
    path_style,
    paths_max,
    scenario,
    show_band_75,
    show_band_95,
    show_real,
    simulation_results,
    withdrawal_start_year,
):
    fig = None
    if error_msg:
        mo.callout(f"Error: {error_msg}", kind="danger")
    elif simulation_results:
        import plotly as _plotly
        _go = _plotly.graph_objects
        mo.md("## Results")
        try:
            mo.callout(
                f"Ran {simulation_results['all_results'].shape[0]} simulations for {simulation_results['months']/12:.1f} years.",
                kind="success",
            )
        except Exception:
            pass
        # Create main visualization (fan chart)
        fig = _go.Figure()

        # Prepare data (real vs nominal)
        time_index_plot = simulation_results["time_index"]
        infl_plot = inflation_rate.value / 100.0
        deflator_plot = (1.0 + infl_plot) ** time_index_plot

        def adj(arr):
            return arr / deflator_plot if show_real.value else arr

        df_plot = simulation_results["df_results"]
        p5_line = adj(df_plot["5th"].values)
        p25_line = adj(df_plot["25th"].values)
        p50_line = adj(df_plot["50th"].values)
        p75_line = adj(df_plot["75th"].values)
        p95_line = adj(df_plot["95th"].values)

        # Brand-aware colors (fallbacks are already set in brand cell)
        color_blue = brand.get("primary", "#0072B2")
        color_orange = brand.get("accent", "#E69F00")
        color_gray = brand.get("muted", "#999999")
        color_lightgray = brand.get("band_fill_outer", "rgba(160,160,160,0.35)")

        # 95% band (area between p5 and p95) — hide in English mode
        if show_band_95.value and not english_mode.value:
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p95_line,
                    line=dict(color="rgba(34,139,34,0.0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    name="",
                )
            )
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p5_line,
                    fill="tonexty",
                    fillcolor=color_lightgray,
                    line=dict(color="rgba(34,139,34,0.0)"),
                    name="5–95% band",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )

        # 25–75% band (inner area) — always show, but use gray palette
        if show_band_75.value:
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p75_line,
                    line=dict(color="rgba(0,0,0,0.0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    name="",
                )
            )
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p25_line,
                    fill="tonexty",
                    fillcolor="rgba(120,120,120,0.35)",
                    line=dict(color="rgba(0,0,0,0.0)"),
                    name="25–75% band",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )

        # Median and mean lines — executive friendly colors
        fig.add_trace(
            _go.Scatter(
                x=time_index_plot,
                y=p50_line,
                mode="lines",
                line=dict(color=color_blue, width=2),
                name="Median (50th)",
                hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
            )
        )
        # Mean line
        mean_vals = adj(df_plot["Mean"].values)
        fig.add_trace(
            _go.Scatter(
                x=time_index_plot,
                y=mean_vals,
                mode="lines",
                line=dict(color=color_orange, width=1.5, dash="dot"),
                name="Mean",
                hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
            )
        )

        # Scenario highlight
        scen = scenario.value
        scen_line = None
        scen_name = None
        if scen.startswith("Conservative"):
            scen_line, scen_name = p25_line, "25th Percentile"
        elif scen.startswith("Median"):
            scen_line, scen_name = p50_line, "Median (50th)"
        elif scen.startswith("Optimistic"):
            scen_line, scen_name = p75_line, "75th Percentile"
        if scen_line is not None:
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=scen_line,
                    mode="lines",
                    line=dict(color=color_orange, width=3),
                    name=f"Scenario: {scen_name}",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )

        # Percentile edge lines — in English mode, show only 25/75 around the median
        if english_mode.value:
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p75_line,
                    mode="lines",
                    line=dict(color=color_gray, width=1),
                    name="75th Percentile",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p25_line,
                    mode="lines",
                    line=dict(color=color_gray, width=1),
                    name="25th Percentile",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )
        else:
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p95_line,
                    mode="lines",
                    line=dict(color=color_gray, width=1, dash="dash"),
                    name="95th Percentile",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p75_line,
                    mode="lines",
                    line=dict(color=color_gray, width=1),
                    name="75th Percentile",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p25_line,
                    mode="lines",
                    line=dict(color=color_gray, width=1),
                    name="25th Percentile",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )
            fig.add_trace(
                _go.Scatter(
                    x=time_index_plot,
                    y=p5_line,
                    mode="lines",
                    line=dict(color=color_gray, width=1, dash="dash"),
                    name="5th Percentile",
                    hovertemplate="Year: %{x:.1f}<br>Value: $%{y:,.0f}",
                )
            )

        # Add path overlays (grey) per selection
        if path_style.value != "None":
            all_paths = simulation_results["all_results"]
            num_paths_draw = int(paths_max.value)
            if num_paths_draw > 0 and all_paths.shape[0] > 0:
                num_paths_draw = min(num_paths_draw, all_paths.shape[0])
                # pick evenly spaced paths for representative coverage
                path_sel_idx = np.linspace(0, all_paths.shape[0] - 1, num=num_paths_draw, dtype=int)
                overlay_paths = all_paths[path_sel_idx, :]
                if show_real.value:
                    overlay_paths = overlay_paths / deflator_plot
                if path_style.value == "Sample":
                    # Draw each as a faint line
                    for overlay_path in overlay_paths:
                        fig.add_trace(
                            _go.Scattergl(
                                x=time_index_plot,
                                y=overlay_path,
                                mode="lines",
                                line=dict(color=brand.get("path_gray", "rgba(0,0,0,0.07)"), width=0.7),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                else:  # All (WebGL) in a single batched trace via None separators
                    paths_xs = []
                    paths_ys = []
                    for overlay_path in overlay_paths:
                        paths_xs.append(time_index_plot)
                        paths_ys.append(overlay_path)
                        paths_xs.append([None])
                        paths_ys.append([None])
                    paths_xs = np.concatenate(paths_xs)
                    paths_ys = np.concatenate(paths_ys)
                    fig.add_trace(
                        _go.Scattergl(
                            x=paths_xs,
                            y=paths_ys,
                            mode="lines",
                            line=dict(color=brand.get("path_gray", "rgba(0,0,0,0.07)"), width=0.7),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        # Add withdrawal phase marker
        if withdrawal_start_year.value > 0:
            fig.add_vline(
                x=withdrawal_start_year.value,
                line_dash="dot",
                line_color="orange",
                annotation_text="Withdrawals Begin",
                annotation_position="top",
            )

        fig.update_layout(
            title="Illustrative Portfolio Projections (Monte Carlo)",
            xaxis_title="Years",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            height=500,
            showlegend=True,
            yaxis=dict(tickformat="$,.0f"),
            template="plotly_white",
        )

        if log_scale.value:
            fig.update_yaxes(type="log")

        mo.ui.plotly(fig)
    else:
        mo.md("Waiting to run: press 'Run simulation'.")
    return


@app.cell
def _(
    base_fee_all,
    diversified_pct,
    english_mode,
    growth_pct,
    income_pct,
    inflation_rate,
    mo,
    np,
    show_real,
    simulation_results,
    sub_fee_diversified,
    sub_fee_income,
    target_final_value,
):
    final_values = None
    stats = None
    if simulation_results:
        # Calculate key statistics
        final_values = simulation_results["all_results"][:, -1]
        initial = simulation_results["initial_value"]
        years = simulation_results["months"] / 12.0
        infl_stats = inflation_rate.value / 100.0
        deflator_T = (1.0 + infl_stats) ** years
        final_values_real = final_values / deflator_T

        # Summary stats
        weights = {
            "growth": float(growth_pct) / 100.0,
            "diversified": float(diversified_pct) / 100.0,
            "income": float(income_pct) / 100.0,
        }
        fee_annual = (
            base_fee_all.value
            + weights["diversified"] * sub_fee_diversified.value
            + weights["income"] * sub_fee_income.value
        )
        gross_return = simulation_results["portfolio_return"] * 100.0
        net_return_est = max(0.0, gross_return - fee_annual)

        stats = {
            "Weighted illustrative return (gross)": f"{gross_return:.2f}%",
            "Effective annual fee (weighted)": f"{fee_annual:.2f}%",
            "Illustrative return (net, approx)": f"{net_return_est:.2f}%",
            "Portfolio Volatility": f"{simulation_results['portfolio_vol']*100:.2f}%",
            "Median Final Value (Nominal)": f"${np.median(final_values):,.0f}",
            "Median Final Value (Real)": f"${np.median(final_values_real):,.0f}",
            "95% Range (Nominal)": f"${np.percentile(final_values, 5):,.0f} - ${np.percentile(final_values, 95):,.0f}",
            "Probability of Loss (vs initial)": f"{(final_values < initial).mean()*100:.1f}%",
        }

        if not english_mode.value:
            # Detailed table
            more = {
                "Mean Final Value (Nominal)": f"${np.mean(final_values):,.0f}",
                "Mean Final Value (Real)": f"${np.mean(final_values_real):,.0f}",
                "Best Case (95th %ile)": f"${np.percentile(final_values, 95):,.0f}",
                "Worst Case (5th %ile)": f"${np.percentile(final_values, 5):,.0f}",
            }
            stats.update(more)
            import pandas as _pd
            df_stats = _pd.DataFrame({"Metric": list(stats.keys()), "Value": list(stats.values())})
            mo.vstack([
                mo.md("## 📊 Portfolio Statistics (Nerd)"),
                mo.ui.table(data=df_stats),
            ])
        else:
            # English summary
            target = float(target_final_value.value)
            final_vals_view = final_values_real if show_real.value else final_values
            p_exceed_summary = (final_vals_view >= target).mean() * 100.0 if target > 0 else None
            p5_stat = np.percentile(final_vals_view, 5)
            p50_stat = np.percentile(final_vals_view, 50)
            p95_stat = np.percentile(final_vals_view, 95)
            units = "real (today's $)" if show_real.value else "nominal"

            bullets = [
                f"Median ending value: ${p50_stat:,.0f} ({units})",
                f"Typical range: ${p5_stat:,.0f} to ${p95_stat:,.0f} (90% band)",
                f"Chance of finishing below start: {(final_values < initial).mean()*100:.1f}%",
            ]
            if p_exceed_summary is not None:
                bullets.append(f"Chance of beating target (${target:,.0f}): {p_exceed_summary:.1f}%")

            mo.md(
                "\n".join(
                    [
                        "## 📊 What this means (English)",
                        "- " + "\n- ".join(bullets),
                    ]
                )
            )
    return


@app.cell
def _(
    brand,
    english_mode,
    inflation_rate,
    mo,
    np,
    show_real,
    simulation_results,
    target_final_value,
):
    colors = None
    fig_dist = None
    final_vals_hist = None
    percentiles_to_mark = None
    if simulation_results:
        import plotly as _plotly
        _go = _plotly.graph_objects
        # Create distribution of final values
        final_vals_hist = simulation_results["all_results"][:, -1]
        infl_hist = inflation_rate.value / 100.0
        horizon_years_hist = simulation_results["months"] / 12.0
        if show_real.value:
            final_vals_hist = final_vals_hist / ((1.0 + infl_hist) ** horizon_years_hist)

        fig_dist = _go.Figure()
        fig_dist.add_trace(
            _go.Histogram(
                x=final_vals_hist,
                nbinsx=30,
                name="Final Portfolio Values",
                marker_color=brand.get("accent", "#14b8a6"),
                hovertemplate="Range: $%{x:,.0f}<br>Count: %{y}",
            )
        )

        # Add vertical lines for percentiles
        percentiles_to_mark = [5, 25, 50, 75, 95]
        colors = ["red", "orange", "blue", "green", "darkgreen"]

        for perc_mark, color in zip(percentiles_to_mark, colors):
            val = np.percentile(final_vals_hist, perc_mark)
            fig_dist.add_vline(
                x=val,
                line_dash="dash" if perc_mark in [5, 95] else "solid",
                line_color=brand.get("muted", color),
                annotation_text=f"{perc_mark}th %ile",
                annotation_position="top",
            )

        # Target line and exceedance probability
        target_hist = float(target_final_value.value)
        if target_hist > 0:
            target_eff_hist = (
                target_hist / ((1.0 + infl_hist) ** horizon_years_hist)
                if show_real.value
                else target_hist
            )
            exceed_pct = (final_vals_hist >= target_eff_hist).mean() * 100.0
            fig_dist.add_vline(
                x=target_eff_hist,
                line_dash="dot",
                line_color=brand.get("primary", "#581c87"),
                annotation_text=f"Target (P≥={exceed_pct:.1f}%)",
                annotation_position="top",
            )

        fig_dist.update_layout(
            title="Distribution of Possible Final Values",
            xaxis_title="Final Portfolio Value ($)",
            yaxis_title="Number of Simulations",
            xaxis=dict(tickformat="$,.0f"),
            template="plotly_white",
            height=400,
            showlegend=False,
        )

        if not english_mode.value:
            mo.md(
                """
                ## 🎯 Final Value Distribution

                This histogram shows the range of possible outcomes at the end of the simulation period.
                """
            )

        mo.ui.plotly(fig_dist)
    return


@app.cell
def _(
    brand,
    english_mode,
    inflation_rate,
    mo,
    np,
    show_real,
    simulation_results,
    target_final_value,
):
    # Final value CDF (exceedance curve) clarifies probabilities
    fig_cdf = None
    if simulation_results:
        import plotly as _plotly
        _go = _plotly.graph_objects
        final_vals_cdf = simulation_results["all_results"][:, -1]
        infl_cdf = inflation_rate.value / 100.0
        horizon_years_cdf = simulation_results["months"] / 12.0
        if show_real.value:
            final_vals_cdf = final_vals_cdf / ((1.0 + infl_cdf) ** horizon_years_cdf)

        cdf_xs = np.sort(final_vals_cdf)
        cdf = np.arange(1, len(cdf_xs) + 1) / len(cdf_xs)
        exceedance_y = 1.0 - cdf

        fig_cdf = _go.Figure()
        fig_cdf.add_trace(
            _go.Scatter(
                x=cdf_xs,
                y=exceedance_y,
                mode="lines",
                line=dict(color=brand.get("accent", "#14b8a6"), width=2),
                name="P(Final ≥ x)",
                hovertemplate="x: $%{x:,.0f}<br>P≥x: %{y:.2f}",
            )
        )

        target_cdf = float(target_final_value.value)
        if target_cdf > 0:
            target_eff_cdf = (
                target_cdf / ((1.0 + infl_cdf) ** horizon_years_cdf)
                if show_real.value
                else target_cdf
            )
            p_exceed_cdf = (final_vals_cdf >= target_eff_cdf).mean()
            fig_cdf.add_vline(
                x=target_eff_cdf,
                line_dash="dot",
                line_color=brand.get("primary", "#581c87"),
                annotation_text=f"Target (P≥={p_exceed_cdf*100:.1f}%)",
                annotation_position="top",
            )

        fig_cdf.update_layout(
            title="Exceedance of Possible Final Values (1 − CDF)",
            xaxis_title="Final Portfolio Value ($)",
            yaxis_title="Probability of Exceedance",
            xaxis=dict(tickformat="$,.0f"),
            template="plotly_white",
            height=380,
            showlegend=False,
        )

        if not english_mode.value:
            mo.ui.plotly(fig_cdf)
    return


@app.cell
def _(
    boxplot_step_years,
    brand,
    inflation_rate,
    mo,
    np,
    show_real,
    simulation_results,
    withdrawal_start_year,
):
    depleted = None
    fig_survival = None
    success_rate = None
    survival_rates = None
    withdrawal_month = None
    withdrawal_values = None
    years_in_withdrawal = None

    if simulation_results and withdrawal_start_year.value > 0:
        import plotly as _plotly
        _go = _plotly.graph_objects
        # Analyze withdrawal phase success
        withdrawal_month = int(withdrawal_start_year.value * 12)
        if withdrawal_month < simulation_results["all_results"].shape[1]:
            withdrawal_values = simulation_results["all_results"][:, withdrawal_month:]

            # Adjust for real values if requested (deflate by month)
            if show_real.value:
                infl_withdraw = inflation_rate.value / 100.0
                t_w = np.arange(withdrawal_values.shape[1]) / 12.0
                deflator_withdraw = (1.0 + infl_withdraw) ** (
                    t_w + withdrawal_start_year.value
                )
                withdrawal_values = withdrawal_values / deflator_withdraw

            # Check how many simulations run out of money
            depleted = (withdrawal_values <= 0).any(axis=1)
            success_rate = (1 - depleted.mean()) * 100

            # Create survival chart
            months_in_withdrawal = withdrawal_values.shape[1]
            survival_rates = []
            years_in_withdrawal = []

            for step_idx in range(months_in_withdrawal):
                surviving = (withdrawal_values[:, step_idx] > 0).mean() * 100
                survival_rates.append(surviving)
                years_in_withdrawal.append(step_idx / 12.0)

            fig_survival = _go.Figure()
            fig_survival.add_trace(
                _go.Scatter(
                    x=years_in_withdrawal,
                    y=survival_rates,
                    mode="lines",
                    line=dict(color=brand.get("accent", "#14b8a6"), width=2),
                    fill="tozeroy",
                    fillcolor=brand.get("band_fill", "rgba(120,120,120,0.35)"),
                    name="Portfolio Survival Rate",
                    hovertemplate="Years: %{x:.1f}<br>Survival Rate: %{y:.1f}%",
                )
            )

            fig_survival.update_layout(
                title="Portfolio Survival Rate During Withdrawal Phase",
                xaxis_title="Years into Withdrawal",
                yaxis_title="Percentage of Portfolios Still Funded (%)",
                template="plotly_white",
                height=400,
                yaxis=dict(range=[0, 105]),
            )

            mo.md(
                f"""
                ## 💰 Withdrawal Phase Analysis

                **Success Rate:** {success_rate:.1f}% of portfolios maintained positive balance throughout withdrawal phase
                """
            )

            mo.ui.plotly(fig_survival)

            # Time-to-depletion histogram (for those that deplete)
            if depleted.any():
                # Find first depletion month per sim (NaN if never)
                first_zero_idx = np.argmax(withdrawal_values <= 0, axis=1)
                # If a row never depletes, argmax gives 0; mask those using depleted
                ttd_years = first_zero_idx[depleted] / 12.0

                fig_ttd = _go.Figure()
                fig_ttd.add_trace(
                    _go.Histogram(
                        x=ttd_years,
                        nbinsx=max(5, int(withdrawal_values.shape[1] / (12 * boxplot_step_years.value))),
                        marker_color=brand.get("warning", "#f59e0b"),
                        name="Time to depletion (years)",
                        hovertemplate="Years: %{x:.1f}<br>Count: %{y}",
                    )
                )
                fig_ttd.update_layout(
                    title="Distribution: Time to Depletion (possible outcomes)",
                    xaxis_title="Years from withdrawal start",
                    yaxis_title="Count",
                    template="plotly_white",
                    height=350,
                    showlegend=False,
                )
                mo.ui.plotly(fig_ttd)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 📚 Understanding the Results

    ### Sequence of Returns Risk
    The Monte Carlo simulation shows how the **order** of returns matters, not just the average.
    Two portfolios with the same average return can have vastly different outcomes depending on when
    good and bad years occur.

    ### Reading the Charts
    - **Gray lines**: Individual simulation paths showing possible futures
    - **Blue line**: Median outcome (50% of simulations above, 50% below)
    - **Green bands**: Likely range of outcomes (25th-75th percentile)
    - **Red/Green dashed**: Extreme outcomes (5th and 95th percentiles)

    ### Key Insights
    1. **Wide range of outcomes**: Even with the same strategy, results vary significantly
    2. **Time matters**: Longer time horizons generally lead to better outcomes but also wider ranges
    3. **Withdrawal timing**: Starting withdrawals during market downturns can severely impact portfolio longevity
    4. **Diversification benefit**: Mixing asset classes can reduce volatility while maintaining returns

    ### Limitations
    - Based on historical approximations; future returns may differ
    - Assumes constant allocation (no rebalancing shown)
    - Does not account for taxes or fees
    - Simplified correlation model between asset classes
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    # Portfolio Monte Carlo simulation parameters from historical data

    Monte Carlo simulations require accurate parameter inputs to generate reliable portfolio projections. Based on comprehensive research across academic literature, investment firms, and historical market data, the most critical finding is that **traditional normal distribution assumptions underestimate portfolio risk by 20-50%**, with extreme events occurring 5-10 times more frequently than standard models predict. The following parameters, derived from empirical evidence spanning 1926-2024, provide the quantitative foundation for robust portfolio simulation models.

    ## Asset class return distributions reveal consistent non-normality

    Historical analysis of equity returns demonstrates pronounced deviations from normal distributions across all major asset classes. **Growth stocks exhibit negative skewness of -0.5 to -1.0 with excess kurtosis of 1.5 to 3.0**, indicating more frequent and severe left-tail events than normal distributions predict. The Russell 1000 Growth Index shows compound annual returns of **9.31% with 17% standard deviation** over long periods, but these averages mask significant regime variations and tail risks.

    Technology sector equities display even more extreme characteristics, with the NASDAQ showing **excess kurtosis of 2.0 to 4.0** due to pronounced boom-bust cycles. Small-cap growth stocks demonstrate the highest volatility parameters, with annual standard deviations of **18-25%** and excess kurtosis reaching **2.5 to 5.0**. Maximum drawdowns for these concentrated equity positions have historically exceeded **60% in major corrections**, with recovery periods spanning 2-4 years.

    Diversified equity portfolios show somewhat moderated but still significant non-normal characteristics. S&P 500 returns from 1982-2001 exhibited conventional skewness of **-2.39 and kurtosis of 53.62** when including all observations, though removing the 1987 crash reduces these to **-0.26 and 6.80** respectively. International developed markets show **skewness of -0.67 with kurtosis of 3.59** when currency-hedged, while emerging markets surprisingly display **positive skewness of 0.56** despite higher overall volatility of **21-22% annually**.

    Fixed income assets present different distribution challenges. Investment-grade bonds show relatively normal distributions with modest excess kurtosis of **2-4**, but high-yield bonds exhibit fat-tail characteristics similar to equities. The current Bloomberg Aggregate Bond Index duration of **approximately 6 years** compares to a long-term average of 4.97 years, creating heightened interest rate sensitivity. International bonds add **10-15% additional volatility** when unhedged due to currency effects.

    Alternative investments require special consideration in Monte Carlo models. Real estate investment trusts historically delivered **11.1% annual returns with 13.5-16.6% volatility**, showing correlations with broad equities ranging from **44% to 66%** depending on market conditions. Commodities exhibit unique challenges with contango conditions costing investors approximately **6.87% annually since 1998** through negative roll yields. Hedge fund indices show survivorship bias requiring **1-3% downward adjustment** to historical returns, while private equity demonstrates J-curve effects with typical hold periods of 7-10 years.

    ## Correlation dynamics and volatility clustering drive modeling accuracy

    Asset correlations vary dramatically across market regimes, fundamentally affecting portfolio risk. During normal market conditions, stock-bond correlations range from **-0.2 to +0.4**, providing diversification benefits. However, during the 2008 financial crisis, correlations across risk assets approached **1.0**, with real estate REITs showing near-perfect correlation with equities instead of providing anticipated diversification. The 2020 COVID crisis saw initial correlation spikes to **0.8-0.9** across most risk assets before rapid normalization due to unprecedented stimulus.

    Volatility clustering follows predictable GARCH(1,1) patterns with **α coefficients of 0.05-0.15 and β coefficients of 0.80-0.95** for most equity series, creating persistence parameters typically between **0.90-0.99**. The VIX index shows mean reversion with a half-life of **7.6 days** toward its long-run mean of approximately **14.6**, though it typically trades at a **1-4 point premium** to subsequent realized volatility. These clustering effects persist with autocorrelation in absolute returns lasting **10-15 months** for most financial series.

    Market regime identification proves crucial for accurate modeling. Bear markets, defined as 20%+ declines, last **6-12 months for event-driven selloffs and 12-18 months for cyclical downturns**, with average decline magnitudes of approximately **30%**. Interest rate regimes typically persist for **5-15 years** with transition periods of 6-18 months. Mean reversion half-lives vary significantly by asset class: **18.5 years for US equities** during normal periods but as short as 2 years during high uncertainty, while emerging markets show **30-month half-lives** and commodities exhibit **12-18 month** mean reversion cycles.

    Momentum effects operate on different timescales with optimal lookback periods of **1-3 months for short-term, 6-12 months for medium-term, and 2-5 years for long-term momentum**. Cross-asset momentum correlations of **0.3-0.5** suggest coordinated market movements that Monte Carlo models must capture through appropriate correlation matrices rather than independent sampling.

    ## Sequence of returns risk dominates retirement portfolio outcomes

    Research spanning 1871-2024 reveals that **70% of portfolio failures involve negative returns in the first 5 years of retirement**. A 100% equity portfolio supports only a **3.1% starting withdrawal rate** for 90% success over 30 years, significantly below the traditional 4% rule. The risk increases dramatically with early losses - a 15% portfolio drop in year one combined with a 3.3% withdrawal rate creates **6 times higher failure risk** compared to positive early returns.

    Updated Trinity Study data confirms the **4% rule remains viable for 30-year periods** but requires high stock allocations of 50-75%. International analysis reveals sobering results: only **5 countries historically supported 4%+ withdrawal rates** even with perfect foresight - Sweden (4.5%), Canada (4.4%), New Zealand (4.1%), Denmark (4.1%), and the United States (4.0%). Global diversification improved success rates from **65.7% to 78.3%** for 4% withdrawals, still leaving substantial failure risk.

    CAPE ratio valuations show **0.77 correlation with safe withdrawal rates**, providing more predictive power than for returns themselves. At CAPE ratios above 20, the 4% rule shows **37% failure rates**, rising to **43.2% when markets are also at all-time highs**. Current CAPE levels near 27 suggest withdrawal rates closer to **4.13%** for traditional planning horizons. Dynamic strategies like Guyton-Klinger guardrails allow higher starting rates of **5.2-5.6%** but historically required spending cuts up to **59% in adverse scenarios** like the 1966 cohort.

    Rising equity glidepaths from **30% stocks at retirement to 60% over 10 years** prove most effective in high-valuation environments, while static 60% equity allocations perform well in favorable markets. Tax-efficient withdrawal sequencing can reduce taxes by **40%+** and extend portfolio longevity by 1-2 years through optimal ordering of account types.

    ## Implementation best practices enhance simulation accuracy

    Block bootstrapping with **3-6 month blocks** captures serial correlation and mean reversion more effectively than traditional Monte Carlo's assumption of independent returns. Factor-specific optimal block sizes range from **1 month for investment factors to 5-6 months for low volatility strategies**. Circular bootstrapping prevents endpoint bias when sampling from limited historical data.

    Rebalancing studies consistently show **annual rebalancing as optimal**, with more frequent rebalancing increasing costs without proportional benefits. Threshold-based approaches using **5% absolute deviation triggers** or **25% relative triggers for small allocations** generated **56 basis points of additional return** over 10 years in T. Rowe Price research. Volatility reduction of approximately **18%** results from annual rebalancing versus buy-and-hold strategies.

    Inflation modeling requires distributional approaches rather than point estimates. The 1970s saw inflation ranging from **6.2% to 13.5%** with high volatility, while the 2021-2023 surge peaked at **9.1%** before moderating. Tax-loss harvesting adds **1.08% annually on average**, ranging from **0.51% in low-volatility periods to 2.13% during high-volatility markets** like the Great Depression era.

    Standard practice uses **10,000 simulations** for most applications, with convergence studies showing 1,000 sufficient for basic analysis and 100,000 for precision-critical applications. Monthly time steps provide optimal balance between computational efficiency and accuracy. Sensitivity analysis should test **±1-2% variations in returns and ±20-50% variations in volatility** to assess model robustness.

    ## Extreme events provide critical tail risk parameters

    Historical market crashes demonstrate the severity of tail events that Monte Carlo simulations must capture. The 1929 crash saw **89% peak-to-trough decline** with 25-year recovery to previous highs. Black Monday 1987 produced a **22.6% single-day decline**, the largest percentage drop in history. The 2000-2002 dot-com crash erased **49% from the S&P 500 and 78% from NASDAQ**, while 2008's financial crisis created a **57% drawdown** over 17 months. The 2020 COVID crash compressed a **34% decline into just one month** before the fastest recovery in history.

    Interest rate shocks create different risk patterns. The Volcker era saw federal funds rates peak at **19.1% in June 1981**, successfully reducing inflation from 14% to 3% but causing back-to-back recessions. The 2013 taper tantrum drove **100 basis point rate increases** in just four months, while 2022's rate increases proceeded at the **fastest pace in 40 years** with multiple 75 basis point hikes.

    Currency effects add complexity to international portfolios. Hedging reduces volatility substantially for quarterly horizons with benefits persisting to 5-year periods, contradicting earlier research suggesting mean reversion eliminates long-term hedging benefits. Hedging costs depend on interest rate differentials, with collar strategies providing optimal risk-adjusted returns. Static **100% hedging for fixed income** and dynamic hedging for equities based on correlation changes represents current best practice.

    ## Conclusion

    Monte Carlo simulations achieve maximum accuracy when incorporating these empirically-derived parameters rather than simplified normal distribution assumptions. The data reveals consistent patterns: negative skewness and excess kurtosis across most asset classes, time-varying correlations that spike during crises, sequence risk that dominates retirement outcomes, and extreme events occurring far more frequently than traditional models predict. Implementing block bootstrapping, regime-switching models, and fat-tailed distributions while maintaining **10,000+ simulation runs** creates robust portfolio projections that better reflect the complex realities of financial markets. These parameters, drawn from nearly a century of market data, provide the quantitative foundation for investment planning that appropriately balances risk and return in an uncertain world. can you add this as a markdown cell at the bottom and als use this as the source of the growth income and diversification strategy (only w/ 1% fees and no commmissions on all strategies. add a "sub-manager fee" for both diversification and income I'll conduct detailed research on the specific parameters and historical characteristics of these three investment strategies for your Monte Carlo simulation.Based on my research, here are the recommended parameters for your three investment strategies in the Monte Carlo simulation:

    ## **1. Actively Managed 25-Stock Strategy (Concentrated Equity)**

    ### Historical Parameters:
    - **Illustrative Annual Return**: 11-13% (use 12%)
    - **Annual Volatility**: 22-28% (use 25%)
    - **Skewness**: -0.8 (negative, indicating more frequent downside surprises)
    - **Excess Kurtosis**: 2.5 (fat tails - extreme events more common)
    - **Maximum Drawdown Risk**: 40-60% in severe corrections

    ### Key Characteristics:
    - Individual stocks in concentrated portfolios experience 4+ drawdowns of 10% or more even in strong years
    - 72-96% of individual stocks have 15%+ drawdowns annually, even in bull markets
    - Volatility drag significantly impacts long-term returns - a stock up 50% then down 50% loses 25% overall
    - Concentrated portfolios tend to cluster in popular growth names, creating boom-bust cycles

    ## **2. Global Tactical Asset Allocation (GTAA)**

    ### Historical Parameters:
    - **Illustrative Annual Return**: 8-10% (use 9%)
    - **Annual Volatility**: 7-12% (use 10%)
    - **Skewness**: -0.3 (slightly negative)
    - **Excess Kurtosis**: 1.0 (moderate fat tails)
    - **Sharpe Ratio**: 0.8-1.2 historically

    ### Key Characteristics:
    - Beta to global markets typically 0.8, providing downside protection
    - Best performance during market stress and regime changes
    - 40-year backtests show worst annual loss of -5.76% for aggressive versions
    - Correlation to traditional portfolios: 0.4-0.6
    - Rebalancing monthly across 6-13 asset classes based on momentum signals

    ## **3. Income Strategy (Bond-Heavy Portfolio)**

    ### Historical Parameters:
    - **Illustrative Annual Return**: 5-7% (use 6%)
    - **Annual Volatility**: 4-8% (use 6%)
    - **Skewness**: 0.1 (nearly symmetric)
    - **Excess Kurtosis**: 0.5 (slight fat tails)
    - **Maximum Drawdown**: 10-15% in rate shock scenarios

    ### Composition Assumptions (typical income portfolio):
    - 30% dividend stocks (7-8% return, 12% volatility)
    - 50% investment-grade bonds (4-5% return, 4% volatility)
    - 20% high-yield bonds (6-7% return, 8% volatility)

    ## **Critical Additional Parameters to Consider**

    ### Correlation Matrix (Normal Markets):
    ```
                        25-Stock  GTAA   Income
    25-Stock Strategy      1.00   0.60    0.40
    GTAA                   0.60   1.00    0.30  
    Income                 0.40   0.30    1.00
    ```

    ### Crisis Correlations (2008-style event):
    - All correlations increase by 0.2-0.4
    - Stock strategies correlation approaches 0.9
    - Bonds may temporarily correlate at 0.6+ with stocks

    ### Volatility Clustering Parameters:
    - Use GARCH(1,1) with α = 0.10, β = 0.85
    - Volatility persistence = 0.95
    - High volatility regimes last 3-6 months on average

    ### Sequence of Returns Adjustments:
    - First 5 years matter most for withdrawal strategies
    - Apply 20% penalty to assumed returns in high valuation environments (CAPE > 25)
    - Increase volatility by 30% in first 3 years of retirement phase

    ## **Best Practices for Implementation**

    1. **Use Block Bootstrapping**: Sample 3-month blocks from historical data rather than individual monthly returns to preserve serial correlation

    2. **Implement Fat Tails**: Use Student's t-distribution with 5-7 degrees of freedom rather than normal distribution, or mixture models combining normal and jump components

    3. **Dynamic Correlations**: Increase correlations by this formula during drawdowns:
       ```
       Crisis_Correlation = Normal_Correlation + 0.3 × (1 - Normal_Correlation) × Market_Stress_Indicator
       ```

    4. **Rebalancing Assumptions**:
       - 25-Stock: Quarterly rebalancing with 2% threshold triggers
       - GTAA: Monthly momentum rebalancing
       - Income: Annual rebalancing or 5% deviation triggers

    5. **Cost Adjustments**:
       - 25-Stock: 1.5% annual fees + 0.5% trading costs
       - GTAA: 0.8% management fee + 0.3% trading costs  
       - Income: 0.5% fees + 0.1% trading costs

    6. **Tax Drag** (if applicable):
       - 25-Stock: 1.2% annual tax drag from turnover
       - GTAA: 0.8% (more tax-efficient)
       - Income: 1.5% (ordinary income treatment)

    ## **Recommended Simulation Adjustments**

    For more realistic results than simple Monte Carlo:

    1. **Regime Switching**: Implement 3 regimes (bull/normal/bear) with transition probabilities
    2. **Mean Reversion**: Apply 0.03 coefficient to returns (3% pull toward long-term mean)
    3. **Valuation Adjustment**: Reduce forward returns by (Current_PE / Historical_PE - 1) × 0.3
    4. **Inflation Scenarios**: Model 3 inflation regimes with different asset class responses
    5. **Liquidity Constraints**: In crisis scenarios, reduce ability to rebalance by 50%

    These parameters will give you much more realistic portfolio projections than using simple normal distributions with constant parameters. The concentrated equity strategy offers highest potential returns but with severe drawdown risk, GTAA provides the best risk-adjusted returns, and the income strategy offers stability but may struggle to keep pace with inflation over long periods.
    """
    )
    return


@app.cell
def _():
    # Load Ethical Capital brand palette (JSON), with safe fallbacks
    import json, os
    brand = {
        "primary": "#581c87",  # ECIC Purple
        "accent": "#14b8a6",   # ECIC Teal
        "warning": "#f59e0b",  # Amber
        "danger": "#ef4444",
        "text": "#111827",
        "muted": "#4b5563",
        "surface": "#ffffff",
        "surface_light": "#f9fafb",
        "border": "#e5e7eb",
    }
    try:
        brand_palette_path = os.path.join("Ethical-Capital-Brand-Kit", "palette", "palette.json")
        if os.path.exists(brand_palette_path):
            with open(brand_palette_path, "r") as f:
                pal_json = json.load(f)
                pal = pal_json.get("palette", {})
                sem = pal_json.get("semantic", {})
                brand["primary"] = sem.get("primary", pal.get("ecic-purple", brand["primary"]))
                # Prefer solid teal accent
                brand["accent"] = pal.get("ecic-teal", brand["accent"])
                brand["warning"] = sem.get("warning", brand["warning"])
                brand["text"] = pal.get("text-dark", brand["text"])
                brand["muted"] = pal.get("text-medium", brand["muted"])
                brand["surface"] = pal.get("surface-white", brand["surface"])
                brand["surface_light"] = pal.get("surface-light", brand["surface_light"])
                brand["border"] = pal.get("border-gray", brand["border"])
    except Exception:
        pass

    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        hx = hex_color.lstrip('#')
        r = int(hx[0:2], 16)
        g = int(hx[2:4], 16)
        b = int(hx[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Provide common derived colors
    brand["band_fill"] = hex_to_rgba(brand["muted"], 0.35)
    brand["band_fill_outer"] = hex_to_rgba(brand["muted"], 0.15)
    brand["path_gray"] = "rgba(0,0,0,0.07)"

    # Do not render brand info in the UI; return palette only
    return (brand,)


@app.cell
def _():
    import base64 as _base64, io as _io, zipfile as _zipfile
    return _base64, _io, _zipfile


if __name__ == "__main__":
    # Minimal CLI for headless diagnostics
    import sys as _sys
    if any(flag in _sys.argv for flag in ("--diag", "--script")):
        import argparse as _argparse
        import os as _os
        import numpy as _np
        import pandas as _pd

        def _diag_build_defaults():
            assets = {
                "growth": {"mean": 0.12, "vol": 0.25},
                "diversified": {"mean": 0.09, "vol": 0.10},
                "income": {"mean": 0.06, "vol": 0.06},
            }
            cfg = {
                "years": 30,
                "n_sims": 500,
                "initial": 100000.0,
                "monthly": 12000.0 / 12.0,
                "seed": 42,
                "weights": {"growth": 0.40, "diversified": 0.40, "income": 0.20},
                "fees": {"base": 0.01, "sub_div": 0.005, "sub_inc": 0.005},
                "regimes": {
                    "use": False,
                    "mode": "Fixed",
                    "fixed": "Normal",
                    "p_stay": {"bull": 0.9, "normal": 0.9, "bear": 0.9},
                },
                "sampling": {"method": "Parametric", "dist": "Student t", "t_df": 6, "block_len": 3},
                "withdrawals": {
                    "start_year": 30,
                    "rate": 0.04,
                    "bridge": {"enable": False, "start": 0, "duration": 0, "rate": 0.0},
                },
                "inflation": 0.025,
            }
            return cfg, assets

        def _diag_simulate_portfolio(cfg: dict, assets: dict):
            weights = cfg["weights"]
            p_return = sum(weights[a] * assets[a]["mean"] for a in weights)
            p_vol = (_np.sqrt(sum((weights[a] * assets[a]["vol"]) ** 2 for a in weights)))

            years = int(cfg["years"]) ; n_sims = int(cfg["n_sims"]) ; initial = float(cfg["initial"]) ; monthly = float(cfg["monthly"]) ; seed = int(cfg["seed"]) if cfg["seed"] is not None else 42
            months = years * 12
            results = _np.zeros((n_sims, months + 1), dtype=float)
            results[:, 0] = initial
            _np.random.seed(seed)

            monthly_fee_rate = cfg["fees"]["base"] / 12.0
            monthly_fee_rate += weights["diversified"] * (cfg["fees"]["sub_div"]) / 12.0
            monthly_fee_rate += weights["income"] * (cfg["fees"]["sub_inc"]) / 12.0

            regime_multipliers = {"Bull": {"mean": 1.25, "vol": 0.85}, "Normal": {"mean": 1.00, "vol": 1.00}, "Bear": {"mean": 0.60, "vol": 1.30}}
            base_m_mean = p_return / 12.0 ; base_m_sigma = p_vol / _np.sqrt(12.0)

            use_reg = bool(cfg["regimes"]["use"]) ; reg_mode = cfg["regimes"]["mode"] ; fixed = cfg["regimes"]["fixed"]
            p_bull = float(cfg["regimes"]["p_stay"]["bull"]) ; p_norm = float(cfg["regimes"]["p_stay"]["normal"]) ; p_bear = float(cfg["regimes"]["p_stay"]["bear"]) 
            samp_method = cfg["sampling"]["method"] ; dist_choice = cfg["sampling"]["dist"] ; dfv = int(cfg["sampling"]["t_df"]) ; block = int(cfg["sampling"]["block_len"]) 

            infl = float(cfg["inflation"]) ; start_w = int(cfg["withdrawals"]["start_year"]) ; w_rate = float(cfg["withdrawals"]["rate"]) ; bridge = cfg["withdrawals"]["bridge"]

            for sim in range(n_sims):
                pv = initial
                if use_reg:
                    if reg_mode == "Fixed":
                        regimes = _np.array([fixed] * months, dtype=object)
                    else:
                        regimes = _np.empty(months, dtype=object); regimes[0] = "Normal"
                        for m in range(1, months):
                            prev = regimes[m - 1]
                            if prev == "Bull":
                                stay = p_bull ; others = (1.0 - stay) / 2.0 ; probs = [stay, others, others]
                            elif prev == "Normal":
                                stay = p_norm ; others = (1.0 - stay) / 2.0 ; probs = [others, stay, others]
                            else:
                                stay = p_bear ; others = (1.0 - stay) / 2.0 ; probs = [others, others, stay]
                            regimes[m] = _np.random.choice(["Bull", "Normal", "Bear"], p=probs)
                else:
                    regimes = _np.array(["Normal"] * months, dtype=object)

                returns_path = None
                if samp_method == "Block bootstrap":
                    ref_len = max(months, 360)
                    ref_states = _np.empty(ref_len, dtype=object) ; ref_states[0] = regimes[0]
                    for m in range(1, ref_len):
                        prev = ref_states[m - 1]
                        if prev == "Bull":
                            stay = p_bull ; others = (1.0 - stay) / 2.0 ; probs = [stay, others, others]
                        elif prev == "Normal":
                            stay = p_norm ; others = (1.0 - stay) / 2.0 ; probs = [others, stay, others]
                        else:
                            stay = p_bear ; others = (1.0 - stay) / 2.0 ; probs = [others, others, stay]
                        ref_states[m] = _np.random.choice(["Bull", "Normal", "Bear"], p=probs)
                    ref_returns = _np.zeros(ref_len)
                    for m in range(ref_len):
                        st = ref_states[m]
                        mu = base_m_mean * regime_multipliers[st]["mean"]
                        sig = max(1e-12, base_m_sigma * regime_multipliers[st]["vol"])
                        if dist_choice == "Student t":
                            tdf = max(3, int(dfv))
                            z = _np.random.standard_t(tdf)
                            z = z * _np.sqrt((tdf - 2) / tdf)
                            ref_returns[m] = mu + z * sig
                        else:
                            ref_returns[m] = _np.random.normal(mu, sig)
                    b = max(1, int(block)) ; chunks = [] ; need = months
                    while need > 0:
                        start = _np.random.randint(0, ref_len) ; end = start + b
                        if end <= ref_len:
                            chunk = ref_returns[start:end]
                        else:
                            wrap = end - ref_len ; chunk = _np.concatenate([ref_returns[start:], ref_returns[:wrap]])
                        chunks.append(chunk) ; need -= b
                    returns_path = _np.concatenate(chunks)[:months]

                for month in range(1, months + 1):
                    year = month // 12
                    st = regimes[month - 1]
                    if returns_path is not None:
                        mret = returns_path[month - 1]
                    else:
                        mu = base_m_mean * regime_multipliers[st]["mean"]
                        sig = max(1e-12, base_m_sigma * regime_multipliers[st]["vol"])
                        if dist_choice == "Student t":
                            tdf = max(3, int(dfv)) ; z = _np.random.standard_t(tdf) ; z = z * _np.sqrt((tdf - 2) / tdf)
                            mret = mu + z * sig
                        else:
                            mret = _np.random.normal(mu, sig)

                    pv *= (1.0 + mret)
                    if monthly_fee_rate > 0:
                        pv *= max(0.0, 1.0 - monthly_fee_rate)

                    in_bridge = bool(bridge["enable"]) and (int(bridge["start"]) <= year < int(bridge["start"] + bridge["duration"]))
                    if in_bridge:
                        annual_withdrawal_b = max(0.0, pv) * float(bridge["rate"]) ; pv = max(0.0, pv - (annual_withdrawal_b / 12.0))
                    else:
                        if year < int(start_w):
                            inflated_contrib = monthly * ((1.0 + infl) ** (month / 12.0)) ; pv += max(0.0, inflated_contrib)
                        if year >= int(start_w):
                            annual_w = max(0.0, pv) * float(w_rate) ; pv = max(0.0, pv - (annual_w / 12.0))

                    results[sim, month] = pv

            time_index = _np.arange(0, months + 1) / 12.0
            percentiles = {
                "5th": _np.percentile(results, 5, axis=0),
                "25th": _np.percentile(results, 25, axis=0),
                "50th": _np.percentile(results, 50, axis=0),
                "75th": _np.percentile(results, 75, axis=0),
                "95th": _np.percentile(results, 95, axis=0),
            }
            df_results = _pd.DataFrame({"Year": time_index, **percentiles, "Mean": _np.mean(results, axis=0)})
            sample_size = min(50, n_sims) ; sample_indices = _np.random.choice(n_sims, sample_size, replace=False) ; sample_paths = results[sample_indices, :]
            return {"df_results": df_results, "sample_paths": sample_paths, "all_results": results, "portfolio_return": p_return, "portfolio_vol": p_vol, "time_index": time_index, "initial_value": initial, "months": months}

        def _diag_write_exports(res: dict, out_dir: str, real: bool = False, infl: float = 0.025):
            _os.makedirs(out_dir, exist_ok=True)
            t = res["time_index"]
            defl = (1.0 + infl) ** t
            dfp = res["df_results"].copy()
            if real:
                for _col in ["5th", "25th", "50th", "75th", "95th", "Mean"]:
                    dfp[_col] = dfp[_col].values / defl
            dff = _pd.DataFrame({"FinalValue": res["all_results"][:, -1] / ((1.0 + infl) ** (res["months"] / 12.0)) if real else res["all_results"][:, -1]})
            paths = res["sample_paths"] / defl if real else res["sample_paths"]
            dfpaths = _pd.DataFrame({"Year": t, **{f"Path_{i+1}": paths[i] for i in range(paths.shape[0])}})
            dfp.to_csv(_os.path.join(out_dir, "percentiles.csv"), index=False)
            dff.to_csv(_os.path.join(out_dir, "final_values.csv"), index=False)
            dfpaths.to_csv(_os.path.join(out_dir, "sample_paths.csv"), index=False)

        parser = _argparse.ArgumentParser(description="Diagnostic script mode: run Monte Carlo once and export CSVs.")
        parser.add_argument("--out", default="diag_out", help="Output directory for CSVs (default: diag_out)")
        parser.add_argument("--years", type=int, default=30)
        parser.add_argument("--sims", type=int, default=500)
        parser.add_argument("--real", action="store_true", help="Export real (inflation-adjusted) values")
        args, _rest = parser.parse_known_args()

        cfg, assets = _diag_build_defaults()
        cfg["years"] = int(args.years)
        cfg["n_sims"] = int(args.sims)
        res = _diag_simulate_portfolio(cfg, assets)
        _diag_write_exports(res, args.out, real=bool(args.real), infl=float(cfg["inflation"]))

        # Print a compact summary
        finals = res["all_results"][:, -1]
        years = res["months"] / 12.0
        deflator_T = (1.0 + float(cfg["inflation"])) ** years
        finals_real = finals / deflator_T
        print(f"Ran {cfg['n_sims']} sims for {years:.1f} years. Median=${_np.median(finals):,.0f} (nominal), Real=${_np.median(finals_real):,.0f}.")
        print(f"CSV exports written to: {args.out}")
    else:
        app.run()
