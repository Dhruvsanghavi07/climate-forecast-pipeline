from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate import forecast as fc
from climate import warehouse
from climate.validate import parse_co2, parse_temp

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def frames():
    co2, _ = parse_co2((FIX / "co2.csv").read_text())
    temp, _ = parse_temp((FIX / "temp.csv").read_text())
    return co2, temp


@pytest.fixture()
def con(tmp_path, frames):
    c = warehouse.connect(tmp_path / "w.db")
    with warehouse.run(c) as run_id:
        warehouse.upsert_co2(c, frames[0], run_id)
        warehouse.upsert_temp(c, frames[1], run_id)
    return c


def synthetic(n=480, slope=0.18, amp=3.0, noise=0.0, seed=0):
    """Rising level + annual cycle, like CO2: ~2.2 ppm/yr growth and a +/-3 ppm season."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    v = 330 + slope * t + amp * np.sin(2 * np.pi * t / 12) + noise * rng.standard_normal(n)
    return pd.Series(v, index=pd.period_range("1980-01", periods=n, freq="M"))


def test_upsert_is_idempotent(con, frames):
    before = con.execute("SELECT COUNT(*) FROM co2_monthly").fetchone()[0]
    with warehouse.run(con) as run_id:
        warehouse.upsert_co2(con, frames[0], run_id)
    assert con.execute("SELECT COUNT(*) FROM co2_monthly").fetchone()[0] == before
    assert con.execute("SELECT COUNT(DISTINCT loaded_run_id) FROM co2_monthly").fetchone()[0] == 1


def test_failed_run_is_recorded_and_rolled_back(con):
    with pytest.raises(RuntimeError), warehouse.run(con) as run_id:
        con.execute("DELETE FROM co2_monthly")
        raise RuntimeError("boom")
    assert con.execute("SELECT COUNT(*) FROM co2_monthly").fetchone()[0] > 0
    status, msg = con.execute("SELECT status, message FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
    assert status == "failed" and "boom" in msg


def test_schema_rejects_impossible_values(con):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO co2_monthly (month, decimal_date, co2_ppm) VALUES ('2099-01', 2099.0, 9000)")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO co2_monthly (month, decimal_date, co2_ppm) VALUES ('Jan 2099', 2099.0, 400)")


def test_sql_views(con):
    growth = pd.read_sql("SELECT * FROM co2_annual_growth ORDER BY year", con)
    assert growth["growth_ppm"].iloc[1:].between(0, 5).all()  # CO2 has risen every year since 1980
    joined = pd.read_sql("SELECT * FROM co2_temp_joined", con)
    assert len(joined) > 400 and joined["anomaly_c"].notna().all()


def test_read_co2_interpolates_gaps_and_counts_them(con):
    con.execute("UPDATE co2_monthly SET co2_ppm = NULL WHERE month = '2000-06'")
    y = warehouse.read_co2(con)
    assert y.attrs["interpolated_months"] == 1 and not y.isna().any()


def test_seasonal_naive_repeats_last_year():
    y = synthetic(n=36)
    np.testing.assert_allclose(fc.seasonal_naive(y, 24)[:12], y.iloc[-12:].to_numpy())


def test_models_recover_a_clean_trend_with_season():
    y = synthetic(n=480)
    truth = synthetic(n=504).iloc[-24:].to_numpy()
    for name in ["seasonal_naive_drift", "trend_fourier", "growth_ridge"]:
        err = np.abs(fc.MODELS[name](y, 24) - truth).mean()
        assert err < 0.5, (name, err)


def test_backtest_has_no_lookahead():
    seen = []

    def spy(train, h):
        seen.append(train.index[-1])
        return np.zeros(h)

    y = synthetic(n=300)
    bt = fc.backtest(y, {"spy": spy, "seasonal_naive": fc.seasonal_naive}, h=12, years=5, step=6)
    for last, origin in zip(seen, bt.errors[bt.errors.model == "spy"]["origin"].unique(), strict=True):
        assert str(last) == origin  # the model saw data up to the origin and nothing later


def test_models_beat_naive_on_noisy_series():
    bt = fc.backtest(synthetic(noise=0.4), h=24, years=5, step=6)
    s = bt.summary()
    assert (s.drop(index="seasonal_naive")["skill_vs_naive_1_12"] > 0.5).all()
    assert fc.choose_model(bt) != "seasonal_naive"


def test_intervals_are_symmetric_and_widen_reasonably():
    bt = fc.backtest(synthetic(noise=0.4), h=24, years=5, step=6)
    band, coverage = fc.intervals(bt, "trend_fourier")
    assert (band["hi"] >= 0).all() and np.allclose(band["lo"], -band["hi"])
    assert 0.5 <= coverage <= 1.0


def test_forecast_frame_shape():
    y = synthetic()
    band = pd.DataFrame({"lo": -np.ones(24), "hi": np.ones(24)}, index=range(1, 25))
    f = fc.forecast(y, "trend_fourier", 24, band)
    assert list(f.columns) == ["month", "yhat", "lo80", "hi80"] and f["month"].iloc[0] == "2020-01"
    assert ((f["hi80"] - f["lo80"]) == 2).all()


def test_warming_per_doubling_on_known_relationship():
    months = pd.period_range("1960-01", "2019-12", freq="M").astype(str)
    co2 = np.linspace(316, 410, len(months))
    joined = pd.DataFrame({"month": months, "co2_ppm": co2, "anomaly_c": 2.0 * np.log(co2 / 280) / np.log(2)})
    r = fc.warming_per_doubling(joined)
    assert r["c_per_doubling"] == pytest.approx(2.0, abs=0.01) and r["r2"] > 0.99
