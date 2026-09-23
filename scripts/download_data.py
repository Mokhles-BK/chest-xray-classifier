# scripts/download_data.py
"""Download the arudaev/chest-xray-14 parquet shards in parallel with retries.

Each shard is a self-contained Apache Parquet file embedding raw PNG bytes under
the `image` column, so no separate image store is needed.  We download to
<data_root>/raw/<split>/<file>.parquet and verify the PAR1 footer before accepting.

Set CHESTXRAY_DATA_ROOT to a mounted Google Drive path in Colab so this only
needs to run once, ever — not once per session.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://huggingface.co/datasets/arudaev/chest-xray-14/resolve/main"
SPLITS = {
    "train": 12,
    "validation": 12,
    "test": 12,
}
MAX_WORKERS = 6
RETRIES = 4


def data_root() -> Path:
    env = os.environ.get("CHESTXRAY_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def url_for(split: str, idx: int, total: int) -> str:
    return f"{BASE}/data/{split}-{idx:05d}-of-{total:05d}.parquet"


def dest_for(root: Path, split: str, idx: int, total: int) -> Path:
    return root / "raw" / split / f"{split}-{idx:05d}-of-{total:05d}.parquet"


def download_one(root: Path, split: str, idx: int, total: int) -> tuple[str, int, int]:
    url = url_for(split, idx, total)
    dest = dest_for(root, split, idx, total)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        with open(dest, "rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        if head == b"PAR1" and tail == b"PAR1":
            return split, idx, dest.stat().st_size  # already downloaded, skip
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "chest-xray-classifier/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            with open(tmp, "rb") as f:
                head = f.read(4)
                f.seek(-4, 2)
                tail = f.read(4)
            if head != b"PAR1" or tail != b"PAR1":
                raise RuntimeError(f"bad parquet magic head={head!r} tail={tail!r}")
            os.replace(tmp, dest)
            return split, idx, dest.stat().st_size
        except Exception as e:  # noqa: BLE001
            last_err = e
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(2**attempt)
    raise RuntimeError(f"failed {url}: {last_err}")


def main() -> int:
    root = data_root()
    tasks = [(s, i, n) for s, n in SPLITS.items() for i in range(n)]
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(download_one, root, s, i, n): (s, i) for s, i, n in tasks}
        for fut in as_completed(futs):
            s, i = futs[fut]
            try:
                _, _, size = fut.result()
                ok += 1
                print(f"OK   {s}-{i:05d}  {size/1e6:.1f} MB", flush=True)
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"FAIL {s}-{i:05d}  {e}", flush=True)
    print(f"\nDONE ok={ok} fail={fail}  root={root}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())