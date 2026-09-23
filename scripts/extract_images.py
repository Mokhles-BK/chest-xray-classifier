# scripts/extract_images.py
"""Extract embedded PNGs from the arudaev parquet shards to disk + build manifests.

The parquet shards embed raw PNG bytes under the `image` column.  We materialise:
  <data_root>/images/<split>/<filename>.png
  <data_root>/manifest/<split>.csv          (split, filename, label_<name>...)

Skips a split entirely if its manifest already exists, so re-running this on a
Drive-backed data_root across Colab sessions is a no-op after the first run.
"""
from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from chestxray.labels import DEFAULT_LABELS  # noqa: E402


def data_root() -> Path:
    env = os.environ.get("CHESTXRAY_DATA_ROOT")
    if env:
        return Path(env)
    return ROOT / "data"


def extract_shard(shard: Path, images_root: Path) -> int:
    import pyarrow.parquet as pq

    split = shard.parent.name
    out_dir = images_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=1024):
        d = batch.to_pydict()
        imgs, labs, fns = d["image"], d["labels"], d["filename"]
        for img, lab, fn in zip(imgs, labs, fns):
            im = Image.open(io.BytesIO(img["bytes"]))
            im.save(out_dir / fn, format="PNG")
            n += 1
            rows.append({"split": split, "filename": fn, "labels": lab})
    if rows:
        df = pd.DataFrame(rows)
        for name in DEFAULT_LABELS:
            df[f"label_{name}"] = df["labels"].apply(
                lambda s, n=name: 1 if n in str(s).replace("|", ",").split(",") else 0
            )
        df.to_csv(images_root.parent / "manifest" / f"{split}.csv", index=False)
    return n


def main() -> int:
    root = data_root()
    raw = root / "raw"
    images = root / "images"
    manifest = root / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)

    shards = sorted(raw.glob("*/*.parquet"))
    splits_present = {s.parent.name for s in shards}
    already_done = {
        split for split in splits_present if (manifest / f"{split}.csv").exists()
    }
    if already_done:
        print(f"SKIP already extracted: {sorted(already_done)}", flush=True)
    todo = [s for s in shards if s.parent.name not in already_done]

    total = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futs = {ex.submit(extract_shard, s, images): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                n = fut.result()
                total += n
                print(f"OK   {s.parent.name}/{s.name}  {n} rows", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {s.parent.name}/{s.name}  {e}", flush=True)
    print(f"\nDONE total={total}  root={root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())