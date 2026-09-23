# CLAUDE.md

## Project
Multi-label chest X-ray pathology classifier (NIH ChestX-ray14). Start with one strong baseline before comparing architectures. Ship as an API eventually, not a notebook.

## Stack
PyTorch, timm, FastAPI, Docker.

## Current state
Milestone 1 complete. Data pipeline (download + extraction, resumable) built.
Baseline trained: ResNet18, 10 epochs, GPU (local RTX 3050). Final result with
tuned per-class thresholds: macro F1 = 0.264, weighted F1 = 0.533 (see
RESULTS.md for full per-class breakdown).

Ablation tested: pos_weight-based class reweighting in the loss (to address
severe class imbalance) — recall improved substantially (0.16 → 0.44 macro)
but precision dropped correspondingly, netting a near-identical/slightly worse
overall F1 (0.254). Conclusion: for this dataset/backbone, per-class threshold
tuning at inference was more effective than reweighting the loss at train time.
Kept as a documented negative result rather than discarded.

Second architecture comparison (EfficientNet/ViT) scoped but not run —
deprioritized in favor of finishing this milestone cleanly. Not currently planned.

API wrapping (FastAPI) and Docker deployment: not yet done — next milestone if resumed.

## Conventions
- Multi-label = sigmoid + BCE, never softmax/accuracy alone.
- Every experiment must log its config and metrics somewhere I can compare later (start with a simple CSV/JSON log if MLflow feels like too much right now).
- Small, verifiable steps. Don't scaffold the whole system before one model actually trains and reports real metrics.