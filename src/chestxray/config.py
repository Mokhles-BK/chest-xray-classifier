# config.py
"""Experiment configuration + JSON/CSV logging.

Every run writes a single JSON file with the exact config used and a CSV of
per-epoch metrics, so experiments are comparable later without MLflow.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device; pass explicit choices through."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def data_root() -> Path:
    """Root data directory. Override with CHESTXRAY_DATA_ROOT (e.g. a mounted
    Google Drive path in Colab) so downloaded/extracted data persists across
    ephemeral sessions instead of living next to the script."""
    env = os.environ.get("CHESTXRAY_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / "data"


@dataclass
class TrainConfig:
    backbone: str = "resnet18"
    image_size: int = 224
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    num_workers: int = 4
    label_smoothing: float = 0.0
    drop_rate: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    min_lr: float = 1e-5
    seed: int = 42
    early_stop_patience: int = 3
    device: str = "auto"
    notes: str = ""

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    run_id: str
    config: dict[str, Any]
    metrics: list[dict[str, Any]] = field(default_factory=list)
    best_metric: dict[str, Any] = field(default_factory=dict)
    thresholds: list[float] = field(default_factory=list)
    notes: str = ""

    def save(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(self.config, indent=2))
        (run_dir / "metrics.csv").write_text(
            pd.DataFrame(self.metrics).to_csv(index=False)
        )
        (run_dir / "run.json").write_text(json.dumps(asdict(self), indent=2, default=str))
        return run_dir / "run.json"


def make_run_dir(base: Path, run_id: str) -> Path:
    return base / run_id