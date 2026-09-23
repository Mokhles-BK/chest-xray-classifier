"""Multi-label classification metrics computed from scratch (no sklearn in the hot loop).

Everything here is per-class precision / recall / F1 plus the standard
multi-label aggregates, so an experiment's numbers are reproducible from a
single confusion-matrix source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class ConfusionMatrix:
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    tn: torch.Tensor

    @classmethod
    def from_targets(cls, y_true: torch.Tensor, y_pred: torch.Tensor) -> "ConfusionMatrix":
        """y_true / y_pred are (N, C) binary tensors."""
        y_true = y_true.bool()
        y_pred = y_pred.bool()
        tp = (y_true & y_pred).sum(dim=0)
        fp = (~y_true & y_pred).sum(dim=0)
        fn = (y_true & ~y_pred).sum(dim=0)
        tn = (~y_true & ~y_pred).sum(dim=0)
        return cls(tp, fp, fn, tn)


def _safe_div(n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return torch.where(d > 0, n / d.clamp_min(1), torch.zeros_like(n, dtype=torch.float32))


def per_class_prf(cm: ConfusionMatrix) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    precision = _safe_div(cm.tp, cm.tp + cm.fp)
    recall = _safe_div(cm.tp, cm.tp + cm.fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


@dataclass
class MultiLabelMetrics:
    labels: Sequence[str]
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    support: torch.Tensor
    subset_accuracy: float
    hamming_loss: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float

    def to_dict(self) -> dict:
        d = {
            "subset_accuracy": self.subset_accuracy,
            "hamming_loss": self.hamming_loss,
            "macro_precision": float(self.macro_precision),
            "macro_recall": float(self.macro_recall),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
        }
        for i, name in enumerate(self.labels):
            d[f"{name}__precision"] = float(self.precision[i])
            d[f"{name}__recall"] = float(self.recall[i])
            d[f"{name}__f1"] = float(self.f1[i])
            d[f"{name}__support"] = int(self.support[i])
        return d


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    labels: Sequence[str],
) -> MultiLabelMetrics:
    """Compute per-class + aggregate multi-label metrics from hard predictions."""
    cm = ConfusionMatrix.from_targets(y_true, y_pred)
    precision, recall, f1 = per_class_prf(cm)
    support = cm.tp + cm.fn
    subset_accuracy = float((y_true == y_pred).all(dim=1).float().mean()) if y_true.numel() else 0.0
    hamming_loss = float((y_true != y_pred).float().mean()) if y_true.numel() else 0.0
    macro_p, macro_r, macro_f = precision.mean(), recall.mean(), f1.mean()
    weighted_f = (f1 * support).sum() / support.sum().clamp_min(1)
    return MultiLabelMetrics(
        labels=tuple(labels),
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        subset_accuracy=subset_accuracy,
        hamming_loss=hamming_loss,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f),
        weighted_f1=float(weighted_f),
    )


def best_thresholds(y_true: torch.Tensor, logits: torch.Tensor, labels: Sequence[str]) -> list[float]:
    """Per-class threshold that maximizes F1 on a validation set (0.5 if undefined)."""
    probs = torch.sigmoid(logits)
    n = probs.shape[1]
    out: list[float] = []
    for c in range(n):
        best_t, best_f1 = 0.5, -1.0
        for t in [i / 40 for i in range(1, 40)]:
            pred = (probs[:, c] >= t).long()
            tp = ((y_true[:, c] == 1) & (pred == 1)).sum().item()
            fp = ((y_true[:, c] == 0) & (pred == 1)).sum().item()
            fn = ((y_true[:, c] == 1) & (pred == 0)).sum().item()
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        out.append(best_t)
    return out