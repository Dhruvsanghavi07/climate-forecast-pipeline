"""Parse and validate the raw files.

Every check records what it found instead of silently fixing data, and the run fails
if a check marked `fatal` does not pass. Known quirks handled here:
- The CO2 mirror's header names 6 columns, but every data row has 7, and the names
  ("Interpolated", "Trend", ...) don't match what the columns hold. The values follow
  NOAA's current file layout: average, deseasonalized, number of days, std dev of days,
  uncertainty. Reading by header would mislabel every column after "Average", so
  columns are assigned by position and the row shape is checked first.
- NOAA uses sentinel values (-99.99 for a missing mean, -1 / -9.99 / -0.99 for missing
  day counts and uncertainties); these become NaN, never numbers.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CO2_COLUMNS = ["month", "decimal_date", "co2_ppm", "co2_deseasonalized_ppm", "n_days", "std_days", "uncertainty"]
SENTINELS = {
    "co2_ppm": [-99.99],
    "co2_deseasonalized_ppm": [-99.99],
    "n_days": [-1],
    "std_days": [-9.99, -0.99],
    "uncertainty": [-9.99, -0.99],
}
RANGES = {"co2_ppm": (250.0, 600.0), "anomaly_c": (-3.0, 3.0)}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    fatal: bool = True


@dataclass
class ValidationReport:
    dataset: str
    rows: int = 0
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks if c.fatal)

    def add(self, name: str, passed: bool, detail: str, fatal: bool = True) -> None:
        self.checks.append(Check(name, bool(passed), detail, fatal))

    def summary(self) -> str:
        lines = [f"{self.dataset}: {self.rows} rows, {'OK' if self.ok else 'FAILED'}"]
        lines += [
            f"  [{'pass' if c.passed else 'FAIL' if c.fatal else 'warn'}] {c.name}: {c.detail}" for c in self.checks
        ]
        return "\n".join(lines)


class ValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        super().__init__(report.summary())
        self.report = report


def _check_monthly_index(df: pd.DataFrame, report: ValidationReport) -> None:
    months = pd.PeriodIndex(df["month"], freq="M")
    dupes = int(months.duplicated().sum())
    report.add("no duplicate months", dupes == 0, f"{dupes} duplicates")
    full = pd.period_range(months.min(), months.max(), freq="M")
    missing = sorted(set(full) - set(months))
    report.add(
        "no gaps between first and last month",
        not missing,
        f"{len(missing)} missing" + (f", e.g. {missing[0]}" if missing else ""),
        fatal=False,
    )
    report.add("months in increasing order", bool(months.is_monotonic_increasing), "checked")


def parse_co2(text: str) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport("co2")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    header, rows = lines[0].split(","), lines[1:]
    widths = {len(r.split(",")) for r in rows}
    report.add("consistent row width", len(widths) == 1, f"row widths {sorted(widths)}")
    report.add(
        "row width matches schema",
        widths == {len(CO2_COLUMNS)},
        f"header has {len(header)} names, rows have {sorted(widths)} fields, schema expects {len(CO2_COLUMNS)}",
    )
    if not report.ok:
        raise ValidationError(report)

    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=CO2_COLUMNS)
    for col, bad in SENTINELS.items():
        df[col] = df[col].mask(df[col].isin(bad))
    report.rows = len(df)
    _check_monthly_index(df, report)
    lo, hi = RANGES["co2_ppm"]
    out = df["co2_ppm"].dropna()
    n_out = int(((out < lo) | (out > hi)).sum())
    report.add("co2 within physical range", n_out == 0, f"{n_out} values outside {lo}-{hi} ppm")
    n_nan = int(df["co2_ppm"].isna().sum())
    report.add("missing monthly means", n_nan <= 12, f"{n_nan} months missing", fatal=False)
    # A month-over-month jump this large has never happened at Mauna Loa; flag it for a human.
    jumps = int((df["co2_ppm"].diff().abs() > 10).sum())
    report.add("no implausible month-to-month jumps", jumps == 0, f"{jumps} jumps over 10 ppm")
    if not report.ok:
        raise ValidationError(report)
    return df, report


def parse_temp(text: str, source: str = "GCAG") -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport("temp")
    raw = pd.read_csv(io.StringIO(text))
    report.add("expected columns", list(raw.columns) == ["Source", "Year", "Mean"], f"columns {list(raw.columns)}")
    if not report.ok:
        raise ValidationError(report)
    df = raw[raw["Source"] == source].rename(columns={"Year": "month", "Mean": "anomaly_c"})[["month", "anomaly_c"]]
    df = df.sort_values("month").reset_index(drop=True)
    df["anomaly_c"] = pd.to_numeric(df["anomaly_c"], errors="coerce")
    report.rows = len(df)
    report.add(f"series {source} present", len(df) > 0, f"{len(df)} rows")
    if len(df):
        _check_monthly_index(df, report)
        lo, hi = RANGES["anomaly_c"]
        vals = df["anomaly_c"].dropna()
        n_out = int(((vals < lo) | (vals > hi)).sum())
        report.add("anomaly within plausible range", n_out == 0, f"{n_out} values outside {lo} to {hi} C")
    if not report.ok:
        raise ValidationError(report)
    df["anomaly_c"] = df["anomaly_c"].astype(float).round(4)
    return df, report


def to_float_or_none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
