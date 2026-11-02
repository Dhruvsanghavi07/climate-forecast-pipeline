"""Build a single-page HTML dashboard with Plotly (charts load plotly.js from its CDN)."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK, MUTED, ACCENT, BAND, OBS = "#1b2330", "#5d6878", "#0b6e4f", "rgba(11,110,79,0.18)", "#3b5b92"
LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
    margin=dict(l=60, r=20, t=50, b=40),
    height=380,
    hovermode="x unified",
)


def fig_forecast(y: pd.Series, fc: pd.DataFrame, model: str) -> go.Figure:
    recent = y.iloc[-12 * 8 :]
    x_obs = recent.index.to_timestamp()
    x_fc = pd.PeriodIndex(fc["month"], freq="M").to_timestamp()
    f = go.Figure()
    f.add_scatter(x=x_obs, y=recent.values, name="Observed", line=dict(color=OBS, width=2))
    f.add_scatter(
        x=list(x_fc) + list(x_fc[::-1]),
        y=list(fc["hi80"]) + list(fc["lo80"][::-1]),
        fill="toself",
        fillcolor=BAND,
        line=dict(width=0),
        name="80% interval",
        hoverinfo="skip",
    )
    f.add_scatter(x=x_fc, y=fc["yhat"], name=f"Forecast ({model})", line=dict(color=ACCENT, width=2, dash="dash"))
    f.update_layout(title="Mauna Loa CO2: last 8 years and 24-month forecast", yaxis_title="CO2 (ppm)", **LAYOUT)
    return f


def fig_backtest(metrics: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for model, g in metrics.groupby("model"):
        f.add_scatter(x=g["horizon"], y=g["mae"], name=model, mode="lines+markers", marker=dict(size=4))
    f.update_layout(
        title="Backtest error by forecast horizon (40 origins, 2014-2024)",
        xaxis_title="Months ahead",
        yaxis_title="Mean absolute error (ppm)",
        **LAYOUT,
    )
    f.update_yaxes(type="log")
    return f


def fig_growth(growth: pd.DataFrame) -> go.Figure:
    g = growth.dropna()
    f = go.Figure(go.Bar(x=g["year"], y=g["growth_ppm"], marker_color=OBS, name="Yearly increase"))
    f.update_layout(title="Year-over-year CO2 increase (annual means)", yaxis_title="ppm per year", **LAYOUT)
    return f


def fig_warming(w: dict) -> go.Figure:
    a = w["annual"]
    f = make_subplots()
    f.add_scatter(
        x=a["co2_ppm"],
        y=a["anomaly_c"],
        mode="markers",
        text=a["year"],
        name="Year",
        marker=dict(color=a["year"].astype(int), colorscale="Viridis", size=7, showscale=False),
    )
    f.update_layout(
        title=f"Temperature anomaly vs CO2, {w['years']} (r² = {w['r2']})",
        xaxis_title="CO2 (ppm, log scale)",
        yaxis_title="Temperature anomaly (°C)",
        **LAYOUT,
    )
    f.update_xaxes(type="log")
    return f


def table(df: pd.DataFrame) -> str:
    return df.to_html(classes="tbl", border=0, float_format=lambda v: f"{v:,.3f}")


def build(out: Path, *, y, fc, model, bt_metrics, bt_summary, coverage, growth, warming, checks, run_info) -> Path:
    figs = [fig_forecast(y, fc, model), fig_backtest(bt_metrics), fig_growth(growth), fig_warming(warming)]
    parts = [
        f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False, config={"displayModeBar": False})
        for i, f in enumerate(figs)
    ]
    nxt = fc.iloc[11]
    checks_html = "".join(
        f"<li class='{'ok' if c.passed else 'bad'}'>{html.escape(c.name)}: {html.escape(c.detail)}</li>" for c in checks
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CO2 Forecast Dashboard</title>
<style>
body{{margin:0;background:#f6f7f9;color:{INK};font:15px/1.55 Inter,system-ui,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px 16px 60px}} h1{{margin:0 0 4px;font-size:1.8rem}}
.muted{{color:{MUTED}}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:#fff;border:1px solid #dde2ea;border-radius:10px;padding:14px}}
.kpi b{{display:block;font-size:1.5rem}} .card{{background:#fff;border:1px solid #dde2ea;border-radius:10px;
padding:8px;margin:14px 0;overflow-x:auto}} .tbl{{border-collapse:collapse;width:100%;font-size:.9rem}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #eef1f5;text-align:right}} .tbl th:first-child,
.tbl td:first-child{{text-align:left}} li.ok::marker{{content:"✓  ";color:{ACCENT}}} li.bad::marker{{content:"✗  ";color:#b3261e}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>CO2 Forecast Dashboard</h1>
<p class="muted">Data through {html.escape(str(y.index[-1]))} · pipeline run {run_info["run_id"]} at
{html.escape(run_info["finished"])} · NOAA Global Monitoring Laboratory (Mauna Loa) and NOAA global temperature</p>
<div class="kpis">
 <div class="kpi"><span class="muted">Latest monthly CO2</span><b>{y.iloc[-1]:.2f} ppm</b>{y.index[-1]}</div>
 <div class="kpi"><span class="muted">Forecast for {nxt["month"]}</span><b>{nxt["yhat"]:.1f} ppm</b>
   80% interval {nxt["lo80"]:.1f} to {nxt["hi80"]:.1f}</div>
 <div class="kpi"><span class="muted">Chosen model, 1-12 month error</span><b>{bt_summary.loc[model, "mae_1_12"]:.2f} ppm</b>
   {model}, {bt_summary.loc[model, "skill_vs_naive_1_12"]:.0%} better than seasonal naive</div>
 <div class="kpi"><span class="muted">Warming per CO2 doubling (observed)</span><b>{warming["c_per_doubling"]:.2f} °C</b>
   correlational, {warming["years"]}</div>
</div>
<div class="card">{parts[0]}</div>
<div class="grid"><div class="card">{parts[1]}</div><div class="card">{parts[2]}</div></div>
<div class="card">{parts[3]}<p class="muted" style="padding:0 10px">Slope of annual temperature anomaly on ln(CO2),
times ln 2. Other forcings that rose with CO2 are folded in, so this is not a causal climate-sensitivity estimate.</p></div>
<h2>Backtest summary</h2><p class="muted">Mean absolute error in ppm. Held-out coverage of the 80% interval for
{model}: {coverage:.0%}.</p><div class="card">{table(bt_summary)}</div>
<h2>Data checks this run</h2><ul>{checks_html}</ul>
</div></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
