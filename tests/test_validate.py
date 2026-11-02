from pathlib import Path

import numpy as np
import pytest

from climate.validate import ValidationError, parse_co2, parse_temp

FIX = Path(__file__).parent / "fixtures"
HEADER = "Date,Decimal Date,Average,Interpolated,Trend,Number of Days"


def co2_text(rows):
    return "\n".join([HEADER, *rows]) + "\n"


def row(month, value, frac=0.5):
    return f"{month},{int(month[:4]) + frac:.4f},{value},{value},25,0.40,0.15"


def test_real_fixture_parses_by_position_not_header():
    df, rep = parse_co2((FIX / "co2.csv").read_text())
    assert rep.ok and len(df) > 500
    assert list(df.columns)[:4] == ["month", "decimal_date", "co2_ppm", "co2_deseasonalized_ppm"]
    assert df["n_days"].dropna().between(0, 31).all()  # would hold std devs if read by header


def test_inconsistent_row_width_is_fatal():
    text = co2_text([row("2020-01", 413.0), "2020-02,2020.12,414.0,414.0"])
    with pytest.raises(ValidationError, match="consistent row width"):
        parse_co2(text)


def test_sentinels_become_missing_not_numbers():
    text = co2_text([row("2020-01", 413.0), row("2020-02", -99.99), row("2020-03", 415.0)])
    df, rep = parse_co2(text)
    assert np.isnan(df.loc[1, "co2_ppm"]) and rep.ok


def test_duplicate_month_is_fatal():
    with pytest.raises(ValidationError, match="no duplicate months"):
        parse_co2(co2_text([row("2020-01", 413.0), row("2020-01", 413.5)]))


def test_gap_is_a_warning_not_a_failure():
    df, rep = parse_co2(co2_text([row("2020-01", 413.0), row("2020-03", 414.0)]))
    gap = next(c for c in rep.checks if c.name.startswith("no gaps"))
    assert not gap.passed and not gap.fatal and rep.ok


def test_out_of_range_and_jump_are_fatal():
    with pytest.raises(ValidationError, match="physical range"):
        parse_co2(co2_text([row("2020-01", 413.0), row("2020-02", 4130.0)]))
    with pytest.raises(ValidationError, match="jumps"):
        parse_co2(co2_text([row("2020-01", 413.0), row("2020-02", 430.0)]))


def test_temp_selects_one_source_and_checks_columns():
    df, rep = parse_temp((FIX / "temp.csv").read_text())
    assert rep.ok and list(df.columns) == ["month", "anomaly_c"]
    with pytest.raises(ValidationError, match="expected columns"):
        parse_temp("Source,Date,Value\nGCAG,2020-01,1.0\n")


def test_report_summary_lists_every_check():
    _, rep = parse_co2((FIX / "co2.csv").read_text())
    assert rep.summary().count("[pass]") == len(rep.checks)
