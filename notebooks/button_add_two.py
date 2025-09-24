# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.16.1,<0.17",
#   "plotly>=5.20,<6.1",
# ]
# ///

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="small")


@app.cell
def _():
    import marimo as mo
    return mo,


@app.cell
def _():
    import plotly.graph_objects as go
    return go,


@app.cell
def _(mo):
    add_button = mo.ui.button(label="Add 2 + 2")
    mo.vstack([
        mo.md("""Press the button to compute 2 + 2 and plot the result."""),
        add_button,
    ])
    return add_button


@app.cell
def _(add_button, go, mo):
    clicks = int(getattr(add_button, "value", 0) or 0)
    result = 2 + 2
    if clicks == 0:
        mo.callout("Waiting for the button press…", kind="info")
    else:
        fig = go.Figure(go.Bar(x=["2 + 2"], y=[result], marker_color="#2563eb"))
        fig.update_layout(
            title="Result of 2 + 2",
            yaxis_title="Value",
            xaxis_title="Expression",
            yaxis=dict(range=[0, result + 1]),
        )
        mo.ui.plotly(fig)
        mo.md(f"Button presses: {clicks} · Result: {result}").callout(kind="success")
    return


if __name__ == "__main__":
    app.run()

