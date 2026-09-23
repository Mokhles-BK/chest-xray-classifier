"""Re-evaluate an existing checkpoint with tuned per-class thresholds.

No retraining — loads a saved best_model.pt and reruns evaluation, useful after
fixing a reporting bug in train_baseline.py without paying the training cost again.

Usage:
    python scripts/reevaluate.py --run-dir runs/resnet18_20260923_023010_49ec6d --backbone resnet18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chestxray.config import data_root  # noqa: E402
from chestxray.dataset import ChestXrayDataset, collate  # noqa: E402
from chestxray.labels import DEFAULT_LABELS, derive_labels_from_manifest, manifest_path  # noqa: E402
from chestxray.metrics import compute_metrics, best_thresholds  # noqa: E402
from chestxray.model import MultiLabelClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Path to the run dir containing best_model.pt")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    root = data_root()
    images_root = root / "images"
    train_manifest = pd.read_csv(manifest_path(root, "train"))
    labels = derive_labels_from_manifest(train_manifest) or DEFAULT_LABELS

    val_df = pd.read_csv(manifest_path(root, "validation"))
    val_ds = ChestXrayDataset(val_df, labels, images_root / "validation", train=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=(device.type == "cuda"),
    )

    model = MultiLabelClassifier(args.backbone, num_classes=len(labels)).to(device)
    ckpt_path = Path(args.run_dir) / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    logits, targets = evaluate(model, val_loader, device)

    # Baseline: fixed 0.5 threshold, for comparison.
    preds_05 = (torch.sigmoid(logits) >= 0.5).float()
    metrics_05 = compute_metrics(targets, preds_05, labels)

    # Tuned: per-class thresholds that maximize F1 on this val set.
    thresholds = best_thresholds(targets, logits, labels)
    probs = torch.sigmoid(logits)
    preds_tuned = (probs >= torch.tensor(thresholds)).float()
    metrics_tuned = compute_metrics(targets, preds_tuned, labels)

    print("\n=== Per-class comparison: threshold=0.5 vs tuned ===", flush=True)
    print(f"{'Label':20s}  {'thr':>5s}  {'P@0.5':>6s}  {'F1@0.5':>6s}   {'P@tuned':>7s}  {'F1@tuned':>8s}  support", flush=True)
    for i, name in enumerate(labels):
        print(
            f"{name:20s}  {thresholds[i]:.3f}  "
            f"{metrics_05.precision[i]:.3f}  {metrics_05.f1[i]:.3f}   "
            f"{metrics_tuned.precision[i]:.3f}   {metrics_tuned.f1[i]:.3f}    {int(metrics_tuned.support[i])}",
            flush=True,
        )

    print(
        f"\nMacro F1 @0.5:   {metrics_05.macro_f1:.4f}"
        f"\nMacro F1 @tuned: {metrics_tuned.macro_f1:.4f}"
        f"\nWeighted F1 @0.5:   {metrics_05.weighted_f1:.4f}"
        f"\nWeighted F1 @tuned: {metrics_tuned.weighted_f1:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
