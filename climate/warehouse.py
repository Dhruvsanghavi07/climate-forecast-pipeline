"""SQLite warehouse: schema setup, run bookkeeping and idempotent upserts."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .validate import to_float_or_none

SCHEMA = Path(__file__).with_name("schema.sql")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA.read_text())
    return con


@contextmanager
def run(con: sqlite3.Connection, source_sha256: str = ""):
    """Record a pipeline run; mark it failed (and roll back its data) if the body raises."""
    cur = con.execute(
        "INSERT INTO pipeline_runs (started_at, status, source_sha256) VALUES (?, 'running', ?)", (now(), source_sha256)
    )
    con.commit()
    run_id = cur.lastrowid
    try:
        yield run_id
        con.execute("UPDATE pipeline_runs SET status='succeeded', finished_at=? WHERE run_id=?", (now(), run_id))
        con.commit()
    except Exception as e:
        con.rollback()
        con.execute(
            "UPDATE pipeline_runs SET status='failed', finished_at=?, message=? WHERE run_id=?",
            (now(), f"{type(e).__name__}: {e}"[:500], run_id),
        )
        con.commit()
        raise


def upsert_co2(con: sqlite3.Connection, df: pd.DataFrame, run_id: int) -> int:
    rows = [
        (
            r.month,
            float(r.decimal_date),
            to_float_or_none(r.co2_ppm),
            to_float_or_none(r.co2_deseasonalized_ppm),
            None if pd.isna(r.n_days) else int(r.n_days),
            to_float_or_none(r.uncertainty),
            run_id,
        )
        for r in df.itertuples()
    ]
    con.executemany(
        """INSERT INTO co2_monthly (month, decimal_date, co2_ppm, co2_deseasonalized_ppm, n_days, uncertainty,
                                    loaded_run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(month) DO UPDATE SET
               decimal_date=excluded.decimal_date, co2_ppm=excluded.co2_ppm,
               co2_deseasonalized_ppm=excluded.co2_deseasonalized_ppm, n_days=excluded.n_days,
               uncertainty=excluded.uncertainty, loaded_run_id=excluded.loaded_run_id""",
        rows,
    )
    _add_rows(con, run_id, len(rows))
    return len(rows)


def upsert_temp(con: sqlite3.Connection, df: pd.DataFrame, run_id: int) -> int:
    rows = [(r.month, to_float_or_none(r.anomaly_c), run_id) for r in df.itertuples()]
    con.executemany(
        """INSERT INTO temp_anomaly_monthly (month, anomaly_c, loaded_run_id) VALUES (?, ?, ?)
           ON CONFLICT(month) DO UPDATE SET anomaly_c=excluded.anomaly_c, loaded_run_id=excluded.loaded_run_id""",
        rows,
    )
    _add_rows(con, run_id, len(rows))
    return len(rows)


def _add_rows(con: sqlite3.Connection, run_id: int, n: int) -> None:
    con.execute("UPDATE pipeline_runs SET rows_upserted = rows_upserted + ? WHERE run_id = ?", (n, run_id))


def read_co2(con: sqlite3.Connection) -> pd.Series:
    """Monthly CO2 as a Series indexed by month-end timestamps; gaps are filled by linear interpolation
    (so models see a regular series) and the count of filled months is kept in .attrs."""
    df = pd.read_sql("SELECT month, co2_ppm FROM co2_monthly ORDER BY month", con)
    s = pd.Series(df["co2_ppm"].to_numpy(), index=pd.PeriodIndex(df["month"], freq="M"), name="co2_ppm")
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    filled = int(s.isna().sum())
    s = s.interpolate(limit_area="inside")
    s.attrs["interpolated_months"] = filled
    return s
