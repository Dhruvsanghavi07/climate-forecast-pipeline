import json
import shutil
from pathlib import Path

from climate.pipeline import run

FIX = Path(__file__).parent / "fixtures"


def test_end_to_end_offline(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("co2", "temp"):
        shutil.copy(FIX / f"{name}.csv", raw / f"{name}.csv")
    out = tmp_path / "site"
    result = run(raw, tmp_path / "w.db", out, offline=True, horizon=24)

    assert result["rows_upserted"] > 1000 and result["last_month"].startswith("20")
    assert result["mae_1_12"] < 1.5  # ppm; the seasonal naive baseline is around 2.5
    page = (out / "index.html").read_text()
    assert "CO2 Forecast Dashboard" in page and "plotly" in page
    assert json.loads((out / "summary.json").read_text())["model"] == result["model"]

    again = run(raw, tmp_path / "w.db", out, offline=True, horizon=24)
    assert again["run_id"] == result["run_id"] + 1
