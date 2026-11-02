-- Warehouse schema (SQLite). Loads are idempotent: re-running a month updates it in place.

CREATE TABLE IF NOT EXISTS co2_monthly (
    month                  TEXT PRIMARY KEY CHECK (month GLOB '[12][0-9][0-9][0-9]-[01][0-9]'),
    decimal_date           REAL NOT NULL,
    co2_ppm                REAL CHECK (co2_ppm IS NULL OR co2_ppm BETWEEN 250 AND 600),
    co2_deseasonalized_ppm REAL,
    n_days                 INTEGER,
    uncertainty            REAL,
    loaded_run_id          INTEGER REFERENCES pipeline_runs(run_id)
);

CREATE TABLE IF NOT EXISTS temp_anomaly_monthly (
    month         TEXT PRIMARY KEY CHECK (month GLOB '[12][0-9][0-9][0-9]-[01][0-9]'),
    anomaly_c     REAL CHECK (anomaly_c IS NULL OR anomaly_c BETWEEN -3 AND 3),
    loaded_run_id INTEGER REFERENCES pipeline_runs(run_id)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    rows_upserted INTEGER DEFAULT 0,
    source_sha256 TEXT,
    message       TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id       INTEGER NOT NULL REFERENCES pipeline_runs(run_id),
    model        TEXT NOT NULL,
    month        TEXT NOT NULL,
    yhat         REAL NOT NULL,
    lo80         REAL NOT NULL,
    hi80         REAL NOT NULL,
    PRIMARY KEY (run_id, model, month)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id   INTEGER NOT NULL REFERENCES pipeline_runs(run_id),
    model    TEXT NOT NULL,
    horizon  INTEGER NOT NULL,
    mae      REAL NOT NULL,
    rmse     REAL NOT NULL,
    PRIMARY KEY (run_id, model, horizon)
);

-- Annual means, only for years with at least 11 measured months.
CREATE VIEW IF NOT EXISTS co2_annual AS
SELECT substr(month, 1, 4) AS year,
       ROUND(AVG(co2_ppm), 2) AS co2_ppm,
       COUNT(co2_ppm)         AS months_measured
FROM co2_monthly
GROUP BY year
HAVING COUNT(co2_ppm) >= 11;

-- Year-over-year growth with a window function.
CREATE VIEW IF NOT EXISTS co2_annual_growth AS
SELECT year, co2_ppm,
       ROUND(co2_ppm - LAG(co2_ppm) OVER (ORDER BY year), 2) AS growth_ppm
FROM co2_annual;

-- Months where both series exist, for the CO2 vs temperature analysis.
CREATE VIEW IF NOT EXISTS co2_temp_joined AS
SELECT c.month, c.co2_ppm, c.co2_deseasonalized_ppm, t.anomaly_c
FROM co2_monthly c
JOIN temp_anomaly_monthly t USING (month)
WHERE c.co2_ppm IS NOT NULL AND t.anomaly_c IS NOT NULL;
