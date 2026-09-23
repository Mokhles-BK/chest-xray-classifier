# model.py
"""Baseline multi-label classifier.

A pretrained backbone (timm) with a single linear head, sigmoid activations and
binary-cross-entropy loss — the honest multi-label baseline.  No softmax, no
accuracy-only reporting.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


def create_model(
    backbone: str = "resnet18",
    num_classes: int = 15,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Build a backbone + classification head model.

    timm models expose `forward_features`; we replace the final classification
    layer with a bare linear projection so the head outputs raw logits for the
    multi-label sigmoid + BCE setup.
    """
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=0,  # remove timm's own head; we add our own
        global_pool="avg",
    )
    in_features = model.num_features
    head = nn.Sequential(
        nn.Dropout(p=drop_rate),
        nn.Linear(in_features, num_classes),
    )
    model.head = head
    model.num_classes = num_classes
    return model


def _pool(feats: torch.Tensor) -> torch.Tensor:
    """Global-average-pool backbone features to (N, C).

    Handles both CNN backbones (feats: N,C,H,W -> pool over spatial dims) and
    transformer backbones (feats: N,tokens,C -> pool over the token dim), so
    the same wrapper works for ResNet/EfficientNet now and ViT later.
    """
    if feats.ndim == 4:
        return feats.mean(dim=(2, 3))
    if feats.ndim == 3:
        return feats.mean(dim=1)
    return feats


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = model.forward_features(x)
    return model.head(_pool(feats))


class MultiLabelClassifier(nn.Module):
    """Thin wrapper so training code can call model(x) -> logits directly."""

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.model = create_model(backbone, num_classes, pretrained, drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return forward_logits(self.model, x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)