"""Evaluation figures: precision/recall vs. threshold and confusion matrices (DOC-02 §13).

Rendered with the non-interactive Agg backend and without software metadata, so the
PNG bytes are deterministic for the same inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_detection.evaluation.threshold import ThresholdChoice  # noqa: E402

logger = logging.getLogger(__name__)

FIGURE_DPI = 110


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    fig.tight_layout()
    fig.savefig(tmp, dpi=FIGURE_DPI, metadata={"Software": None})
    plt.close(fig)
    tmp.replace(path)
    logger.info("Saved figure %s", path)
    return path


def plot_threshold_curve(table: pd.DataFrame, choice: ThresholdChoice, model: str, path: Path) -> Path:
    """Precision, Recall and F2 against the threshold, marking r_min and the chosen threshold."""
    data = table.sort_values("threshold", kind="mergesort")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(data["threshold"], data["precision"], label="Precision", color="#4C72B0")
    ax.plot(data["threshold"], data["recall"], label="Recall", color="#C44E52")
    ax.plot(data["threshold"], data["f2"], label="F2", color="#55A868", linestyle="--")
    ax.axhline(choice.r_min, color="#8C8C8C", linestyle=":", label=f"r_min = {choice.r_min:.2f}")
    ax.axvline(choice.threshold, color="black", linewidth=1,
               label=f"chosen threshold = {choice.threshold:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Threshold (fraud if p >= threshold)")
    ax.set_ylabel("Validation metric")
    ax.set_title(f"{model}: precision / recall vs threshold ({choice.objective})")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def plot_confusion_matrix(matrix: list[list[int]], title: str, path: Path) -> Path:
    """2x2 confusion matrix with counts (rows: actual, columns: predicted)."""
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow([[1, 0], [0, 1]], cmap="Blues", vmin=0, vmax=3)  # fixed shading; counts carry the data
    labels = ["Legitimate", "Fraud"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    return _save(fig, path)
