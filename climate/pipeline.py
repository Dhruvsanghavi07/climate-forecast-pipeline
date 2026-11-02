"""End-to-end run: extract -> validate -> load -> backtest -> forecast -> report.

python -m climate.pipeline                      # download, then run everything
python -m climate.pipeline --raw-dir tests/fixtures --offline   # use files already on disk
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import forecast as fc
from . import report, warehouse
from .sources import SOURCES, download
from .validate import parse_co2, parse_temp

log = logging.getLogger("climate")


def run(raw_dir: Path, db_path: Path, out_dir: Path, offline: bool = False, horizon: int = 24) -> dict:
    t0 = time.perf_counter()
    paths = {name: (raw_dir / f"{name}.csv") if offline else download(src, raw_dir) for name, src in SOURCES.items()}
    sha = ",".join(
        json.loads(p.with_suffix(".json").read_text())["sha256"][:12]
        for p in paths.values()
        if p.with_suffix(".json").exists()
    )

    co2, rep_co2 = parse_co2(paths["co2"].read_text())
    temp, rep_temp = parse_temp(paths["temp"].read_text())
    log.info("\n%s\n%s", rep_co2.summary(), rep_temp.summary())

    con = warehouse.connect(db_path)
    with warehouse.run(con, sha) as run_id:
        n = warehouse.upsert_co2(con, co2, run_id) + warehouse.upsert_temp(con, temp, run_id)
        y = warehouse.read_co2(con)
        bt = fc.backtest(y, h=horizon)
        model = fc.choose_model(bt)
        band, coverage = fc.intervals(bt, model)
        forecast = fc.forecast(y, model, horizon, band)
        metrics = bt.metrics()
        con.executemany(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?)",
            [(run_id, model, r.month, r.yhat, r.lo80, r.hi80) for r in forecast.itertuples()],
        )
        con.executemany(
            "INSERT INTO backtest_metrics VALUES (?, ?, ?, ?, ?)",
            [(run_id, r.model, r.horizon, r.mae, r.rmse) for r in metrics.itertuples()],
        )
        growth = pd.read_sql("SELECT * FROM co2_annual_growth ORDER BY year", con)
        warming = fc.warming_per_doubling(pd.read_sql("SELECT * FROM co2_temp_joined", con))
        summary = bt.summary()

    finished = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    page = report.build(
        out_dir / "index.html",
        y=y,
        fc=forecast,
        model=model,
        bt_metrics=metrics,
        bt_summary=summary,
        coverage=coverage,
        growth=growth,
        warming=warming,
        checks=rep_co2.checks + rep_temp.checks,
        run_info={"run_id": run_id, "finished": finished},
    )
    result = {
        "run_id": run_id,
        "rows_upserted": n,
        "last_month": str(y.index[-1]),
        "model": model,
        "mae_1_12": float(summary.loc[model, "mae_1_12"]),
        "mae_13_24": float(summary.loc[model, "mae_13_24"]),
        "interval_coverage_heldout": round(coverage, 3),
        "c_per_doubling": warming["c_per_doubling"],
        "warming_r2": warming["r2"],
        "next_12_months": forecast.iloc[11].to_dict(),
        "seconds": round(time.perf_counter() - t0, 1),
        "report": str(page),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    summary.to_csv(out_dir / "backtest_summary.csv")
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="climate-pipeline")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--db", type=Path, default=Path("data/warehouse.db"))
    p.add_argument("--out", type=Path, default=Path("site"))
    p.add_argument("--offline", action="store_true", help="use raw files already in --raw-dir")
    p.add_argument("--horizon", type=int, default=24)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(a.raw_dir, a.db, a.out, a.offline, a.horizon), indent=2, default=str))


if __name__ == "__main__":
    main()
