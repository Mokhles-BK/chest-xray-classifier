"""Multi-label loss: sigmoid + binary cross-entropy, with optional label smoothing.

Multi-label pathology classification is a set-membership problem, so the loss
operates per-element of the label vector — never softmax over the class axis.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiLabelBCELoss(nn.Module):
    """BCEWithLogitsLoss with per-positive label smoothing (0 = no smoothing)."""

    def __init__(self, smoothing: float = 0.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(logits, targets)
        if self.smoothing > 0:
            # replace positive targets with (1 - smoothing) and negative with smoothing/2
            smoothed = targets * (1 - self.smoothing) + (1 - targets) * (self.smoothing / 2)
            loss = self.loss(logits, smoothed)
        return loss.mean()