"""Canonical label set for the NIH ChestX-ray14 multi-label task.

The arudaev/chest-xray-14 parquet shards store each row's labels as a single
comma-separated string (e.g. "Infiltration, Mass, Nodule").  We derive the
canonical ordering from the data itself so the code never hardcodes a list
that could drift from the source.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Fallback ordering matching the official NIH ChestX-ray14 paper if the data
# cannot be inspected (e.g. offline manifest reuse).
DEFAULT_LABELS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def derive_labels_from_manifest(manifest: pd.DataFrame) -> list[str]:
    """Return the sorted canonical label list from a manifest's label columns."""
    cols = [c for c in manifest.columns if c.startswith("label_")]
    return sorted(c[len("label_") :] for c in cols)


def parse_labels(raw: str, labels: list[str]) -> list[float]:
    """Convert a label string to a multi-hot float vector.

    Accepts both comma- and pipe-separated encodings (arudaev uses
    "Atelectasis|Infiltration"; some upstream dumps use commas).
    """
    text = str(raw).replace("|", ",")
    present = {s.strip() for s in text.split(",") if s.strip()}
    return [1.0 if name in present else 0.0 for name in labels]


def label_index(labels: list[str], name: str) -> int:
    return labels.index(name)


def manifest_path(data_dir: Path, split: str) -> Path:
    return data_dir / "manifest" / f"{split}.csv"