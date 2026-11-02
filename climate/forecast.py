"""Monthly CO2 forecasting with a rolling-origin backtest.

Models (all predict 1..H months past the last observed month):
- seasonal_naive:        same month last year. The baseline every model must beat.
- seasonal_naive_drift:  same month last year + the average yearly increase of the last 5 years.
- trend_fourier:         Ridge regression on a quadratic time trend + 3 annual Fourier pairs,
                         fit on the last 20 years.
- growth_ridge / growth_gbm: model the 12-month growth g(t) = y(t) - y(t-12) from its own
                         last 12 values (Ridge or gradient boosting), forecast g recursively,
                         and rebuild y(t) = y(t-12) + g(t). Growth is smooth and bounded, which
                         suits tree models that cannot extrapolate a rising level.

Backtest: origins every 3 months over the last 10 years (40 origins); at each origin a model only sees data
up to that month (no look-ahead). Errors are kept per horizon so short- and long-range accuracy
can be compared.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

Forecaster = Callable[[pd.Series, int], np.ndarray]


def seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    last_year = y.to_numpy()[-12:]
    return np.array([last_year[i % 12] for i in range(h)])


def seasonal_naive_drift(y: pd.Series, h: int, years: int = 5) -> np.ndarray:
    v = y.to_numpy()
    drift = (v[-1] - v[-1 - 12 * years]) / years  # ppm per year
    return np.array([v[-12 + (i % 12)] + drift * (i // 12 + 1) for i in range(h)])


def _design(t: np.ndarray, month: np.ndarray, k: int = 3) -> np.ndarray:
    cols = [t, t**2]
    for j in range(1, k + 1):
        cols += [np.sin(2 * np.pi * j * month / 12), np.cos(2 * np.pi * j * month / 12)]
    return np.column_stack(cols)


def trend_fourier(y: pd.Series, h: int, years: int = 20) -> np.ndarray:
    y = y.iloc[-12 * years :]
    t = np.arange(len(y), dtype=float) / 120.0  # decades, keeps the quadratic term well scaled
    month = np.array([p.month for p in y.index], dtype=float)
    model = Ridge(alpha=1e-3).fit(_design(t, month), y.to_numpy())
    t_f = (np.arange(len(y), len(y) + h, dtype=float)) / 120.0
    m_f = np.array([(y.index[-1] + i).month for i in range(1, h + 1)], dtype=float)
    return model.predict(_design(t_f, m_f))


def _growth_model(make_estimator: Callable[[], object], lags: int = 12, years: int = 30) -> Forecaster:
    def forecast(y: pd.Series, h: int) -> np.ndarray:
        v = y.to_numpy()[-12 * years :]
        g = v[12:] - v[:-12]
        x = np.array([g[i - lags : i] for i in range(lags, len(g))])
        target = g[lags:]
        est = make_estimator().fit(x, target)
        hist_g, hist_y = list(g), list(v)
        out = []
        for _ in range(h):
            nxt_g = float(est.predict(np.array(hist_g[-lags:])[None, :])[0])
            nxt_y = hist_y[-12] + nxt_g
            hist_g.append(nxt_g)
            hist_y.append(nxt_y)
            out.append(nxt_y)
        return np.array(out)

    return forecast


MODELS: dict[str, Forecaster] = {
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_drift": seasonal_naive_drift,
    "trend_fourier": trend_fourier,
    "growth_ridge": _growth_model(lambda: Ridge(alpha=1.0)),
    "growth_gbm": _growth_model(
        lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    ),
}


@dataclass
class Backtest:
    errors: pd.DataFrame  # columns: model, origin, horizon, error (actual - forecast)

    def metrics(self) -> pd.DataFrame:
        e = self.errors
        return (
            e.assign(abs=e["error"].abs(), sq=e["error"] ** 2)
            .groupby(["model", "horizon"])
            .agg(mae=("abs", "mean"), rmse=("sq", lambda s: float(np.sqrt(s.mean()))))
            .reset_index()
        )

    def summary(self) -> pd.DataFrame:
        """MAE for months 1-12 and 13-24 ahead, plus skill vs the seasonal naive baseline (1 - MAE/MAE_naive)."""
        e = self.errors.assign(abs=self.errors["error"].abs())
        e["range"] = np.where(e["horizon"] <= 12, "mae_1_12", "mae_13_24")
        t = e.pivot_table(index="model", columns="range", values="abs", aggfunc="mean")
        base = t.loc["seasonal_naive", "mae_1_12"]
        t["skill_vs_naive_1_12"] = 1 - t["mae_1_12"] / base
        return t.sort_values("mae_1_12").round(3)


def backtest(
    y: pd.Series, models: dict[str, Forecaster] = MODELS, h: int = 24, years: int = 10, step: int = 3
) -> Backtest:
    rows = []
    last_origin = len(y) - h - 1
    origins = range(last_origin - 12 * years + step, last_origin + 1, step)
    for name, f in models.items():
        for o in origins:
            train, actual = y.iloc[: o + 1], y.iloc[o + 1 : o + 1 + h].to_numpy()
            pred = f(train, h)
            for i in range(h):
                rows.append((name, str(y.index[o]), i + 1, actual[i] - pred[i]))
    return Backtest(pd.DataFrame(rows, columns=["model", "origin", "horizon", "error"]))


def intervals(bt: Backtest, model: str, level: float = 0.8) -> tuple[pd.DataFrame, float]:
    """Split-conformal prediction intervals: yhat +/- the `level` quantile of |backtest error| per horizon.

    Symmetric bands on absolute errors are used because the sign of the error flips between
    periods (forecasts ran low in El Nino years and high afterwards); quantiles of signed errors
    from older origins covered only ~37% of newer errors for the trend model. To report honest
    coverage, the quantile is estimated on the older half of the origins and checked on the newer
    half; the returned band uses all origins."""
    e = bt.errors[bt.errors["model"] == model].assign(abs_err=lambda d: d["error"].abs())
    origins = sorted(e["origin"].unique())
    old = e[e["origin"].isin(origins[: len(origins) // 2])]
    new = e[e["origin"].isin(origins[len(origins) // 2 :])]
    q_old = old.groupby("horizon")["abs_err"].quantile(level).rename("q")
    joined = new.join(q_old, on="horizon")
    covered = float((joined["abs_err"] <= joined["q"]).mean())
    q = e.groupby("horizon")["abs_err"].quantile(level)
    return pd.DataFrame({"lo": -q, "hi": q}), covered


def choose_model(bt: Backtest) -> str:
    """Lowest mean absolute error across all horizons (1-24 months)."""
    return str(bt.errors.assign(a=bt.errors["error"].abs()).groupby("model")["a"].mean().idxmin())


def forecast(y: pd.Series, model: str, h: int, band: pd.DataFrame) -> pd.DataFrame:
    yhat = MODELS[model](y, h)
    idx = pd.period_range(y.index[-1] + 1, periods=h, freq="M")
    return pd.DataFrame(
        {
            "month": idx.astype(str),
            "yhat": yhat,
            "lo80": yhat + band["lo"].to_numpy()[:h],
            "hi80": yhat + band["hi"].to_numpy()[:h],
        }
    )


def warming_per_doubling(joined: pd.DataFrame) -> dict:
    """OLS of annual temperature anomaly on ln(annual CO2). slope * ln 2 is the warming associated
    with a doubling of CO2 over this record. Observational and correlational: it folds in every
    other forcing that moved with CO2 (aerosols, other greenhouse gases), so it is not a causal
    climate-sensitivity estimate."""
    j = joined.assign(year=joined["month"].str[:4])
    counts = j.groupby("year").size()
    annual = j[j["year"].isin(counts[counts == 12].index)].groupby("year")[["co2_ppm", "anomaly_c"]].mean()
    x, yv = np.log(annual["co2_ppm"].to_numpy()), annual["anomaly_c"].to_numpy()
    slope, intercept = np.polyfit(x, yv, 1)
    pred = slope * x + intercept
    r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return {
        "years": f"{annual.index.min()}-{annual.index.max()}",
        "n_years": int(len(annual)),
        "c_per_doubling": round(float(slope * np.log(2)), 2),
        "r2": round(float(r2), 3),
        "annual": annual.reset_index(),
    }
