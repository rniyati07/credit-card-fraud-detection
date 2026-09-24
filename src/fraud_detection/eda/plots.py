"""EDA figures (DOC-02 §4). The set is capped at :data:`MAX_FIGURES` (DOC-02: ≤ 8 figures).

Every function draws one figure from precomputed results or the dataset, saves it and
returns the path. The non-interactive Agg backend keeps rendering headless and
deterministic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from fraud_detection.config import EdaConfig, SchemaConfig  # noqa: E402
from fraud_detection.eda.analysis import AMOUNT_COLUMN, EdaResults  # noqa: E402

logger = logging.getLogger(__name__)

MAX_FIGURES = 8
LEGIT_COLOR = "#4C72B0"
FRAUD_COLOR = "#C44E52"
CLASS_PALETTE = {"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR}


def _save(fig: Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    logger.info("Saved figure %s", path)
    return path


def _class_labels(target: pd.Series, positive_label: int) -> pd.Series:
    return target.eq(positive_label).map({True: "Fraud", False: "Legitimate"})


def plot_class_distribution(results: EdaResults, path: Path, dpi: int) -> Path:
    """Bar chart of class counts on a log y-axis (item 3)."""
    dist = results.class_distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["Legitimate", "Fraud"]
    counts = [dist["legitimate_count"], dist["fraud_count"]]
    bars = ax.bar(labels, counts, color=[LEGIT_COLOR, FRAUD_COLOR])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(f"Class distribution: fraud = {dist['fraud_percentage']:.3f}%")
    for bar, count in zip(bars, counts, strict=True):
        ax.annotate(f"{count:,}", (bar.get_x() + bar.get_width() / 2, count),
                    ha="center", va="bottom", fontsize=10)
    return _save(fig, path, dpi)


def plot_amount_distribution(
    df: pd.DataFrame, schema: SchemaConfig, eda: EdaConfig, path: Path, dpi: int
) -> Path:
    """Amount histogram on a log x-axis, overall (counts) and per class (share) (item 5).

    Per-class bars are the share of that class's transactions in each bin. A ``density``
    normalisation would divide by bin width and inflate the narrow low-amount log bins.
    """
    amount = df[AMOUNT_COLUMN]
    positive = amount[amount > 0]
    edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), eda.histogram_bins)
    is_fraud = df[schema.target] == schema.positive_label

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(positive, bins=edges, color="#7F7F7F")
    axes[0].set_title(f"All transactions (Amount > 0; {int((amount == 0).sum()):,} zero amounts)")
    axes[0].set_ylabel("Transactions")
    for label, mask, color in (("Legitimate", ~is_fraud, LEGIT_COLOR), ("Fraud", is_fraud, FRAUD_COLOR)):
        values = amount[mask & (amount > 0)]
        weights = np.full(len(values), 100 / max(len(values), 1))
        axes[1].hist(values, bins=edges, weights=weights, histtype="step", linewidth=2,
                     color=color, label=label)
    axes[1].set_title("By class (Amount > 0; share of each class per bin)")
    axes[1].set_ylabel("Share of class (%)")
    axes[1].legend()
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Amount (log scale)")
    return _save(fig, path, dpi)


def plot_amount_by_class(df: pd.DataFrame, schema: SchemaConfig, path: Path, dpi: int) -> Path:
    """Boxplot of ``log10(1 + Amount)`` by class (item 5)."""
    values = np.log10(1 + df[AMOUNT_COLUMN])
    labels = _class_labels(df[schema.target], schema.positive_label)
    order = ["Legitimate", "Fraud"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    boxes = ax.boxplot([values[labels == name] for name in order], tick_labels=order,
                       patch_artist=True, flierprops={"markersize": 1.5})
    for patch, name in zip(boxes["boxes"], order, strict=True):
        patch.set_facecolor(CLASS_PALETTE[name])
        patch.set_alpha(0.7)
    ax.set_ylabel("log10(1 + Amount)")
    ax.set_title("Amount by class (outliers retained)")
    return _save(fig, path, dpi)


def plot_time_by_class(
    df: pd.DataFrame, results: EdaResults, schema: SchemaConfig, eda: EdaConfig, path: Path, dpi: int
) -> Path:
    """Density of elapsed time by class and fraud rate per time bin (item 6; descriptive only)."""
    hours = df[schema.ordering_column] / 3600
    is_fraud = df[schema.target] == schema.positive_label
    edges = np.arange(0, hours.max() + eda.time_bin_seconds / 3600, eda.time_bin_seconds / 3600)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, mask, color in (("Legitimate", ~is_fraud, LEGIT_COLOR), ("Fraud", is_fraud, FRAUD_COLOR)):
        axes[0].hist(hours[mask], bins=edges, density=True, histtype="step", linewidth=2,
                     color=color, label=label)
    axes[0].set_title("Elapsed time by class (density)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    rate = results.fraud_rate_over_time
    axes[1].plot(rate.index, 100 * rate["fraud_rate"], marker="o", markersize=3, color=FRAUD_COLOR)
    axes[1].axhline(results.class_distribution["fraud_percentage"], linestyle="--", color="#555555",
                    label="Overall fraud %")
    axes[1].set_title(f"Fraud rate per {eda.time_bin_seconds / 3600:g}-hour bin")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Hours since first transaction")
    return _save(fig, path, dpi)


def plot_mean_differences(results: EdaResults, path: Path, dpi: int) -> Path:
    """Ranked standardized class-mean differences of V1-V28 (item 7)."""
    diffs = results.mean_differences["std_mean_diff"]
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in diffs]
    ax.barh(diffs.index[::-1], diffs.to_numpy()[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("(mean fraud - mean legitimate) / overall std")
    ax.set_title("Class separation by V-feature (ranked by |effect|)")
    return _save(fig, path, dpi)


def plot_top_feature_kde(
    df: pd.DataFrame, results: EdaResults, schema: SchemaConfig, eda: EdaConfig, path: Path, dpi: int
) -> Path:
    """KDE small multiples of the most separating V-features by class (item 8)."""
    top = list(results.mean_differences.index[: eda.top_k_features])
    n_cols = min(3, len(top))
    n_rows = int(np.ceil(len(top) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)
    labels = _class_labels(df[schema.target], schema.positive_label)
    for ax, feature in zip(axes.flat, top, strict=False):
        sns.kdeplot(x=df[feature], hue=labels, hue_order=["Legitimate", "Fraud"], palette=CLASS_PALETTE,
                    common_norm=False, fill=True, alpha=0.3, ax=ax, warn_singular=False)
        ax.set_title(feature)
        ax.set_xlabel("")
    for ax in list(axes.flat)[len(top):]:
        ax.set_visible(False)
    fig.suptitle(f"Top {len(top)} separating V-features: density by class (each class normalised)")
    return _save(fig, path, dpi)


def plot_target_correlation(results: EdaResults, path: Path, dpi: int) -> Path:
    """Correlation of each feature with the target, ranked by |r| (item 9)."""
    corr = results.target_correlation
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in corr]
    ax.barh(corr.index[::-1], corr.to_numpy()[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with Class")
    ax.set_title("Feature correlation with fraud label")
    return _save(fig, path, dpi)


def plot_correlation_heatmap(results: EdaResults, path: Path, dpi: int) -> Path:
    """Feature-feature Pearson correlation heatmap (item 10)."""
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(results.feature_correlation, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.2, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature correlation matrix (V1-V28, Amount)")
    return _save(fig, path, dpi)


def generate_figures(
    df: pd.DataFrame, results: EdaResults, schema: SchemaConfig, eda: EdaConfig, figures_dir: Path
) -> dict[str, Path]:
    """Render every EDA figure into ``figures_dir``; returns ``{name: path}`` in report order."""
    dpi = eda.figure_dpi
    plotters: dict[str, Callable[[Path], Path]] = {
        "class_distribution": lambda p: plot_class_distribution(results, p, dpi),
        "amount_distribution": lambda p: plot_amount_distribution(df, schema, eda, p, dpi),
        "amount_by_class": lambda p: plot_amount_by_class(df, schema, p, dpi),
        "time_by_class": lambda p: plot_time_by_class(df, results, schema, eda, p, dpi),
        "feature_mean_differences": lambda p: plot_mean_differences(results, p, dpi),
        "top_feature_distributions": lambda p: plot_top_feature_kde(df, results, schema, eda, p, dpi),
        "correlation_with_class": lambda p: plot_target_correlation(results, p, dpi),
        "correlation_heatmap": lambda p: plot_correlation_heatmap(results, p, dpi),
    }
    if len(plotters) > MAX_FIGURES:
        raise RuntimeError(f"EDA is limited to {MAX_FIGURES} figures (DOC-02 §4)")
    sns.set_theme(style="whitegrid")
    return {name: plot(figures_dir / f"{name}.png") for name, plot in plotters.items()}
