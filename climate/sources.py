"""Data sources and download with caching.

Both series come from the Open Knowledge Foundation "datasets" mirrors on GitHub, which republish
the official files monthly under a public-domain license:
- co2: NOAA Global Monitoring Laboratory, Mauna Loa monthly mean CO2 (ppm), since 1958
- temp: NOAA global land+ocean temperature anomaly (GCAG, deg C vs the 1901-2000 average), since 1850
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = "https://raw.githubusercontent.com/datasets"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    description: str


SOURCES = {
    "co2": Source("co2", f"{BASE}/co2-ppm/main/data/co2-mm-mlo.csv", "NOAA GML Mauna Loa monthly mean CO2 (ppm)"),
    "temp": Source("temp", f"{BASE}/global-temp/main/data/monthly.csv", "NOAA global temperature anomaly (deg C)"),
}


def download(source: Source, raw_dir: Path, retries: int = 3, timeout: int = 30) -> Path:
    """Fetch a source to raw_dir/<name>.csv and write a sidecar with its SHA-256 and fetch time.

    Retries with exponential backoff; raises the last error if every attempt fails."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{source.name}.csv"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(source.url, timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            meta = {
                "url": source.url,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            dest.with_suffix(".json").write_text(json.dumps(meta, indent=2))
            return dest
        except OSError as e:  # URLError and timeouts are OSError subclasses
            last = e
            time.sleep(2**attempt)
    raise RuntimeError(f"could not download {source.name} from {source.url}") from last
