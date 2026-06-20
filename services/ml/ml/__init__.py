"""Geopolitic ML package — spatio-temporal GNN for country↔country link-label
prediction. See plans/03-ml-workflow.md. Heavy modules (dataset, model, train,
infer, explain, calibrate, metrics, losses) import torch / torch_geometric; the
config and timestep modules are dependency-free so they can be used (and tested)
without the ML stack installed."""

from .config import CLASS_NAMES, Config  # noqa: F401
