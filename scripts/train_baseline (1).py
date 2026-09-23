"""Train the ChestX-ray14 multi-label baseline and report honest metrics.

Usage (from repo root, with src/ on the path):
    python scripts/train_baseline.py --epochs 10 --batch-size 64
    python scripts/train_baseline.py --smoke-test          # ~200 images, 1 epoch, sanity check only

Reads data from $CHESTXRAY_DATA_ROOT (falls back to <repo>/data). Writes a
run directory under runs/<run_id>/ with config.json, metrics.csv, run.json —
comparable across runs without needing MLflow.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chestxray.config import TrainConfig, RunRecord, make_run_dir, data_root  # noqa: E402
from chestxray.dataset import ChestXrayDataset, collate  # noqa: E402
from chestxray.labels import DEFAULT_LABELS, derive_labels_from_manifest, manifest_path  # noqa: E402
from chestxray.loss import MultiLabelBCELoss  # noqa: E402
from chestxray.metrics import compute_metrics, best_thresholds  # noqa: E402
from chestxray.model import MultiLabelClassifier, count_trainable_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--pos-weight", action="store_true",
        help="Upweight rare-class positives in the loss (neg/pos ratio per class, capped at 20x) to fight class imbalance during training, not just at inference.",
    )
    p.add_argument("--pos-weight-cap", type=float, default=20.0)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--notes", default="")
    p.add_argument("--run-name", default=None, help="Defaults to <backbone>_<timestamp>")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Subsample ~200 train / 100 val images, run 1 epoch, verify the pipeline runs end-to-end.",
    )
    return p.parse_args()


def build_datasets(labels: list[str], images_root: Path, smoke: bool):
    train_df = pd.read_csv(manifest_path(images_root.parent, "train"))
    val_df = pd.read_csv(manifest_path(images_root.parent, "validation"))
    if smoke:
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=0)
        val_df = val_df.sample(n=min(100, len(val_df)), random_state=0)
    train_ds = ChestXrayDataset(train_df, labels, images_root / "train", train=True)
    val_ds = ChestXrayDataset(val_df, labels, images_root / "validation", train=False)
    return train_ds, val_ds


@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(targets)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = (torch.sigmoid(logits) >= 0.5).float()
    metrics = compute_metrics(targets, preds, labels)
    return metrics, logits, targets


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = TrainConfig(
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=1 if args.smoke_test else args.epochs,
        num_workers=args.num_workers,
        label_smoothing=args.label_smoothing,
        drop_rate=args.drop_rate,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        notes=args.notes,
    )
    device = torch.device(cfg.device)
    print(f"Device: {cfg.device}", flush=True)

    root = data_root()
    images_root = root / "images"
    train_manifest = pd.read_csv(manifest_path(root, "train"))
    labels = derive_labels_from_manifest(train_manifest) or DEFAULT_LABELS
    print(f"Labels ({len(labels)}): {labels}", flush=True)

    train_ds, val_ds = build_datasets(labels, images_root, args.smoke_test)
    print(f"Train examples: {len(train_ds)}  Val examples: {len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=(device.type == "cuda"),
    )

    model = MultiLabelClassifier(cfg.backbone, num_classes=len(labels), drop_rate=cfg.drop_rate).to(device)
    print(f"Model: {cfg.backbone}  trainable params: {count_trainable_parameters(model):,}", flush=True)

    pos_weight = None
    if args.pos_weight:
        target_cols = [f"label_{n}" for n in labels]
        pos_counts = train_manifest[target_cols].sum(axis=0).values
        neg_counts = len(train_manifest) - pos_counts
        ratios = neg_counts / pos_counts.clip(min=1)
        ratios = ratios.clip(max=args.pos_weight_cap)
        pos_weight = torch.tensor(ratios, dtype=torch.float32).to(device)
        print(f"pos_weight (capped at {args.pos_weight_cap}x): {dict(zip(labels, ratios.round(1)))}", flush=True)

    criterion = MultiLabelBCELoss(smoothing=cfg.label_smoothing, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    run_id = args.run_name or f"{cfg.backbone}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = make_run_dir(ROOT / "runs", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = RunRecord(run_id=run_id, config=cfg.to_dict(), notes=cfg.notes)

    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)

        val_metrics, val_logits, val_targets = evaluate(model, val_loader, device, labels)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch}/{cfg.epochs}  train_loss={train_loss:.4f}  "
            f"val_macro_f1={val_metrics.macro_f1:.4f}  val_macro_p={val_metrics.macro_precision:.4f}  "
            f"val_macro_r={val_metrics.macro_recall:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

        epoch_record = {"epoch": epoch, "train_loss": train_loss, **val_metrics.to_dict()}
        record.metrics.append(epoch_record)

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            epochs_without_improvement = 0
            record.best_metric = epoch_record
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {cfg.early_stop_patience} epochs)", flush=True)
                break

    # Load the actual best checkpoint before final reporting — the in-memory model
    # is whatever the last epoch trained, which may not be the best one saved.
    best_ckpt = run_dir / "best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print(f"Loaded best checkpoint for final report: {best_ckpt}", flush=True)

    # Final report: tune per-class thresholds on val, then RE-EVALUATE with them
    # (not the default 0.5) so the report reflects each class's optimal cutoff.
    _, raw_logits, raw_targets = evaluate(model, val_loader, device, labels)
    thresholds = best_thresholds(raw_targets, raw_logits, labels)
    record.thresholds = thresholds

    probs = torch.sigmoid(raw_logits)
    thresh_tensor = torch.tensor(thresholds)
    final_preds = (probs >= thresh_tensor).float()
    final_metrics = compute_metrics(raw_targets, final_preds, labels)

    print("\n=== Final per-class metrics (per-class tuned thresholds) ===", flush=True)
    for i, name in enumerate(labels):
        print(
            f"{name:20s}  P={final_metrics.precision[i]:.3f}  "
            f"R={final_metrics.recall[i]:.3f}  F1={final_metrics.f1[i]:.3f}  "
            f"support={int(final_metrics.support[i])}",
            flush=True,
        )
    print(
        f"\nMacro  P={final_metrics.macro_precision:.4f}  R={final_metrics.macro_recall:.4f}  "
        f"F1={final_metrics.macro_f1:.4f}  Weighted F1={final_metrics.weighted_f1:.4f}",
        flush=True,
    )

    run_path = record.save(run_dir)
    print(f"\nRun saved to {run_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
