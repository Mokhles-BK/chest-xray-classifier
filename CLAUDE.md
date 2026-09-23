# CLAUDE.md

## Project
Multi-label chest X-ray pathology classifier (NIH ChestX-ray14). Start with one strong baseline before comparing architectures. Ship as an API eventually, not a notebook.

## Stack
PyTorch, timm, FastAPI, Docker.

## Current state
Nothing built yet. First milestone: working data pipeline + one trained baseline model with honest metrics (per-class precision/recall/F1, not accuracy).

## Conventions
- Multi-label = sigmoid + BCE, never softmax/accuracy alone.
- Every experiment must log its config and metrics somewhere I can compare later (start with a simple CSV/JSON log if MLflow feels like too much right now).
- Small, verifiable steps. Don't scaffold the whole system before one model actually trains and reports real metrics.