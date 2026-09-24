"""Render the EDA Markdown report from computed results.

Every number and data-dependent statement is derived from :class:`EdaResults`; the
report contains no generation timestamp so re-runs on the same data are byte-identical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import SchemaConfig
from fraud_detection.eda.analysis import AMOUNT_COLUMN, EdaResults

# Design fractions from DOC-02 §6 / §21, used only to estimate fraud counts per partition.
DESIGN_TEMPORAL_HOLDOUT_FRACTION = 0.15
DESIGN_VAL_TEST_FRACTION = 0.15
LOW_COLLINEARITY_ABS_R = 0.3


def _int(value: float) -> str:
    return f"{int(value):,}"


def _num(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _pct(fraction: float, digits: int = 3) -> str:
    return f"{100 * fraction:.{digits}f}%"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table (no external dependency)."""
    lines = [
        "| " + " | ".join(map(str, headers)) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def _figure(figures: Mapping[str, Path], name: str, caption: str, report_dir: Path) -> str:
    rel = Path(figures[name]).relative_to(report_dir).as_posix()
    return f"![{caption}]({rel})\n\n*{caption}*"


def _v_pair_max_abs_corr(corr: pd.DataFrame) -> tuple[float, str, str]:
    v_cols = [c for c in corr.columns if c != AMOUNT_COLUMN]
    sub = corr.loc[v_cols, v_cols].to_numpy(copy=True)
    np.fill_diagonal(sub, 0.0)
    i, j = np.unravel_index(np.abs(sub).argmax(), sub.shape)
    return float(abs(sub[i, j])), v_cols[i], v_cols[j]


def _amount_corr(corr: pd.DataFrame) -> pd.Series:
    s = corr[AMOUNT_COLUMN].drop(AMOUNT_COLUMN)
    return s.reindex(s.abs().sort_values(ascending=False, kind="mergesort").index)


# -------------------------------------------------------------------------------- sections


def _executive_summary(r: EdaResults) -> str:
    cd, amt, ts = r.class_distribution, r.amount_statistics, r.time_summary
    top3 = ", ".join(r.mean_differences.index[:3])
    max_r, a, b = _v_pair_max_abs_corr(r.feature_correlation)
    fraud_med, legit_med = amt.loc["fraud", "median"], amt.loc["legitimate", "median"]
    relation = "lower" if fraud_med < legit_med else "higher" if fraud_med > legit_med else "equal"
    quality = r.data_quality
    quality_line = (
        "Data quality is confirmed against the M1 validation report: no duplicates, missing "
        "or infinite values remain."
        if quality.get("matches_m1_report")
        and quality["duplicate_rows"] == 0
        and quality["missing_values_total"] == 0
        and quality["infinite_values_total"] == 0
        else "**Data quality does not fully reconcile with M1. See §2 before proceeding.**"
    )
    return (
        f"The deduplicated dataset contains **{_int(cd['total'])} transactions** spanning "
        f"**{ts['span_hours']:.1f} hours**, of which **{_int(cd['fraud_count'])} are fraudulent "
        f"({cd['fraud_percentage']:.4f}%)**, about 1 fraud per {cd['imbalance_ratio']:,.0f} "
        f"legitimate transactions. A model predicting \"legitimate\" for everything would reach "
        f"{_pct(cd['all_legitimate_accuracy'])} accuracy while catching no fraud, so accuracy "
        f"is unusable for model selection. `Amount` is strongly right-skewed "
        f"(skewness {_num(amt.loc['all', 'skewness'], 1)}); the median fraud amount "
        f"({_num(fraud_med, 2)}) is {relation} than the median legitimate amount "
        f"({_num(legit_med, 2)}). The anonymised features {top3} separate the classes most "
        f"strongly, and the PCA components are nearly uncorrelated with each other "
        f"(max |r| = {max_r:.3f}, {a}–{b}). {quality_line}"
    )


def _key_findings(r: EdaResults) -> list[str]:
    cd, amt, ts, var = r.class_distribution, r.amount_statistics, r.time_summary, r.variance
    md = r.mean_differences
    strong = md[md["std_mean_diff"].abs() >= 0.5]
    tc = r.target_correlation
    max_r, a, b = _v_pair_max_abs_corr(r.feature_correlation)
    amount_corr = _amount_corr(r.feature_correlation)
    return [
        f"**Extreme imbalance:** {_int(cd['fraud_count'])} frauds vs "
        f"{_int(cd['legitimate_count'])} legitimate (ratio 1 : {cd['imbalance_ratio']:,.0f}); "
        f"the no-skill PR-AUC baseline equals the prevalence, {cd['fraud_rate']:.5f}.",
        f"**Amount is heavy-tailed:** median {_num(amt.loc['all', 'median'], 2)}, 99th percentile "
        f"{_num(amt.loc['all', 'q99'], 2)}, max {_num(amt.loc['all', 'max'], 2)}; skewness "
        f"{_num(amt.loc['all', 'skewness'], 1)} falls to {_num(amt.loc['all', 'log1p_skewness'], 2)} "
        f"after `log1p`.",
        f"**Fraud amounts differ in shape, not just level:** fraud median "
        f"{_num(amt.loc['fraud', 'median'], 2)} vs legitimate {_num(amt.loc['legitimate', 'median'], 2)}; "
        f"fraud mean {_num(amt.loc['fraud', 'mean'], 2)} vs {_num(amt.loc['legitimate', 'mean'], 2)}; "
        f"{_pct(amt.loc['fraud', 'zero_share'], 1)} of frauds have `Amount = 0` "
        f"(legitimate: {_pct(amt.loc['legitimate', 'zero_share'], 1)}).",
        f"**Strong class signal in V-features:** {len(strong)} of {len(md)} V-features have a "
        f"standardized mean difference ≥ 0.5 in magnitude; the largest is {md.index[0]} "
        f"({md['std_mean_diff'].iloc[0]:+.2f} σ). Highest |correlation| with `Class`: "
        f"{tc.index[0]} (r = {tc.iloc[0]:+.3f}), {tc.index[1]} (r = {tc.iloc[1]:+.3f}), "
        f"{tc.index[2]} (r = {tc.iloc[2]:+.3f}).",
        f"**Low collinearity among PCA components:** max |r| between V-features is {max_r:.3f} "
        f"({a}–{b}). `Amount` is the only notably correlated variable (strongest with "
        f"{amount_corr.index[0]}, r = {amount_corr.iloc[0]:+.3f}).",
        f"**Scale differs sharply:** `Amount` variance is {var['amount_to_max_v_variance_ratio']:,.0f}× "
        f"the largest V-feature variance ({var['max_feature']}); V-feature variances span a "
        f"{var['max_to_min_ratio']:.1f}× range.",
        f"**Fraud rate varies over the capture window:** per-bin fraud rate ranges from "
        f"{_pct(ts['bin_fraud_rate_min'])} to {_pct(ts['bin_fraud_rate_max'])} "
        f"(median {_pct(ts['bin_fraud_rate_median'])}) across {ts['n_bins']} bins, so a "
        f"chronological holdout's prevalence may differ from the overall rate.",
    ]


def _overview_section(r: EdaResults) -> str:
    ov = r.overview
    rows = [
        ["Rows", _int(ov["rows"])],
        ["Columns", _int(ov["columns"])],
        ["Data types", ", ".join(f"{t} × {n}" for t, n in ov["dtype_counts"].items())],
        ["In-memory size (deep)", f"{ov['memory_mb']:.1f} MB"],
    ]
    dtype_rows = [[c, t] for c, t in ov["dtypes"].items()]
    return (
        markdown_table(["Property", "Value"], rows)
        + "\n\n<details><summary>Column data types</summary>\n\n"
        + markdown_table(["Column", "dtype"], dtype_rows)
        + "\n\n</details>"
    )


def _quality_section(r: EdaResults) -> str:
    q = r.data_quality
    rows = [
        ["Duplicate rows remaining", _int(q["duplicate_rows"])],
        ["Missing values", _int(q["missing_values_total"])],
        ["Infinite values", _int(q["infinite_values_total"])],
    ]
    if q["validation_report_available"]:
        rows += [
            ["M1 validation status", q["m1_overall_status"]],
            ["M1 raw rows / frauds", f"{_int(q['m1_raw_rows'])} / {_int(q['m1_raw_fraud_count'])}"],
            ["M1 duplicates removed", _int(q["m1_duplicates_removed"])],
            ["M1 clean rows / frauds", f"{_int(q['m1_clean_rows'])} / {_int(q['m1_clean_fraud_count'])}"],
            ["Clean rows match M1 report", "yes" if q["matches_m1_report"] else "**NO**"],
            ["Raw data SHA-256", f"`{q['m1_raw_sha256']}`"],
        ]
        note = (
            "Duplicates were removed in M1 before any split (DOC-02 §3 duplicate policy), which "
            "prevents identical rows straddling train and test. Outliers were deliberately kept."
        )
    else:
        note = "**The M1 validation report was not found; M1 results could not be reconciled.**"
    return markdown_table(["Check", "Result"], rows) + "\n\n" + note


def _class_section(r: EdaResults, figures: Mapping[str, Path], report_dir: Path) -> str:
    cd = r.class_distribution
    pool_frauds = cd["fraud_count"] * (1 - DESIGN_TEMPORAL_HOLDOUT_FRACTION)
    per_split = pool_frauds * DESIGN_VAL_TEST_FRACTION
    table = markdown_table(
        ["Metric", "Value"],
        [
            ["Fraud (Class = 1)", _int(cd["fraud_count"])],
            ["Legitimate (Class = 0)", _int(cd["legitimate_count"])],
            ["Fraud percentage", f"{cd['fraud_percentage']:.4f}%"],
            ["Imbalance ratio (legitimate : fraud)", f"{cd['imbalance_ratio']:,.1f} : 1"],
            ["Accuracy of always predicting legitimate", _pct(cd["all_legitimate_accuracy"])],
        ],
    )
    insights = [
        f"Accuracy is uninformative: the trivial classifier scores {_pct(cd['all_legitimate_accuracy'])}. "
        "Model selection uses PR-AUC, with Precision/Recall/F1/F2 at a tuned threshold (DOC-02 §11–13).",
        f"The PR-AUC of a random classifier equals the prevalence ({cd['fraud_rate']:.5f}); "
        "reported PR-AUC must be read against this baseline (DOC-01 SC-06).",
        "The threshold must be tuned on validation data: with this prior, predicted probabilities "
        "concentrate near 0 and the default 0.50 is unlikely to be a good operating point.",
        f"Stratification is essential. If prevalence were uniform, the development pool (earlier "
        f"{_pct(1 - DESIGN_TEMPORAL_HOLDOUT_FRACTION, 0)}) would hold ≈ {pool_frauds:.0f} frauds and "
        f"each {_pct(DESIGN_VAL_TEST_FRACTION, 0)} validation/test split ≈ {per_split:.0f}, so one "
        f"fraud ≈ {100 / per_split:.1f} pp of recall.",
    ]
    return (
        table
        + "\n\n"
        + _figure(figures, "class_distribution", "Class counts (log scale)", report_dir)
        + "\n\n**Fraud imbalance insights**\n\n"
        + "\n".join(f"- {i}" for i in insights)
    )


def _amount_section(r: EdaResults, figures: Mapping[str, Path], report_dir: Path) -> str:
    amt = r.amount_statistics
    stats = [
        ("count", "Count", 0), ("mean", "Mean", 2), ("std", "Std", 2), ("min", "Min", 2),
        ("q25", "25%", 2), ("median", "Median", 2), ("q75", "75%", 2), ("q95", "95%", 2),
        ("q99", "99%", 2), ("max", "Max", 2), ("skewness", "Skewness", 2),
        ("log1p_skewness", "Skewness of log1p", 2),
    ]
    rows = [[label] + [_num(amt.loc[g, key], d) for g in ("all", "legitimate", "fraud")]
            for key, label, d in stats]
    rows.append(["Share with Amount = 0"] + [_pct(amt.loc[g, "zero_share"], 2)
                                             for g in ("all", "legitimate", "fraud")])
    rows.append(["Share with Amount ≤ 1"] + [_pct(amt.loc[g, "le_one_share"], 2)
                                             for g in ("all", "legitimate", "fraud")])
    fraud, legit = amt.loc["fraud"], amt.loc["legitimate"]
    insights = [
        f"`Amount` is heavily right-skewed (skewness {_num(amt.loc['all', 'skewness'], 1)}) with a long "
        f"tail up to {_num(amt.loc['all', 'max'], 2)}. This supports `RobustScaler` for `Amount` "
        "(DOC-02 §7, TD-04); outliers stay in the data (TD-02).",
        f"`log1p` reduces skewness to {_num(amt.loc['all', 'log1p_skewness'], 2)}. Whether it helps "
        "Logistic Regression remains an empirical question for validation (TD-05, default off).",
        f"Fraud amounts are {'more' if fraud['std'] > legit['std'] else 'less'} dispersed "
        f"(std {_num(fraud['std'], 2)} vs {_num(legit['std'], 2)}), with median {_num(fraud['median'], 2)} "
        f"vs {_num(legit['median'], 2)} and 95th percentile {_num(fraud['q95'], 2)} vs "
        f"{_num(legit['q95'], 2)}.",
        f"Very small amounts are over-represented in fraud: {_pct(fraud['le_one_share'], 1)} of "
        f"frauds have `Amount ≤ 1` vs {_pct(legit['le_one_share'], 1)} of legitimate transactions "
        "(consistent with, but not proof of, card-testing behaviour; descriptive only, not a derived feature).",
        f"`Amount`'s linear correlation with `Class` is "
        f"r = {r.target_correlation[AMOUNT_COLUMN]:+.4f}, so it "
        f"{'is a weak signal on its own' if abs(r.target_correlation[AMOUNT_COLUMN]) < 0.1 else 'carries some linear signal'}.",
    ]
    return (
        markdown_table(["Statistic", "All", "Legitimate", "Fraud"], rows)
        + "\n\n"
        + _figure(figures, "amount_distribution", "Amount distribution: overall and by class (log x-axis)", report_dir)
        + "\n\n"
        + _figure(figures, "amount_by_class", "log10(1 + Amount) by class", report_dir)
        + "\n\n**Amount behaviour insights**\n\n"
        + "\n".join(f"- {i}" for i in insights)
    )


def _time_section(r: EdaResults, figures: Mapping[str, Path], report_dir: Path, schema: SchemaConfig) -> str:
    ts = r.time_summary
    rows = [
        ["Span", f"{ts['span_hours']:.1f} h ({_num(ts['min_seconds'], 0)} – {_num(ts['max_seconds'], 0)} s)"],
        ["Rows sorted by time in file", "yes" if ts["is_sorted_in_file"] else "no"],
        ["Median time, legitimate / fraud", f"{ts['legitimate_median_hours']:.1f} h / {ts['fraud_median_hours']:.1f} h"],
        ["Per-bin fraud rate (min / median / max)",
         f"{_pct(ts['bin_fraud_rate_min'])} / {_pct(ts['bin_fraud_rate_median'])} / {_pct(ts['bin_fraud_rate_max'])}"],
        ["Bins without any fraud", f"{ts['bins_without_fraud']} of {ts['n_bins']}"],
    ]
    return (
        f"`{schema.ordering_column}` is an **ordering key only** and is not a model feature "
        "(DOC-02 §21.2, D-11). No hour-of-day or other time features are derived. This section "
        "only describes the chronology that the temporal holdout (M3) will inherit.\n\n"
        + markdown_table(["Property", "Value"], rows)
        + "\n\n"
        + _figure(figures, "time_by_class", "Elapsed time by class, and fraud rate per time bin", report_dir)
    )


def _feature_section(r: EdaResults, figures: Mapping[str, Path], report_dir: Path, eda_top_k: int) -> str:
    fs = r.feature_summary
    summary_rows = [
        [f, _num(fs.loc[f, "mean"], 3), _num(fs.loc[f, "std"], 3), _num(fs.loc[f, "variance"], 3),
         _num(fs.loc[f, "min"], 2), _num(fs.loc[f, "median"], 3), _num(fs.loc[f, "max"], 2),
         _num(fs.loc[f, "skewness"], 2), _num(fs.loc[f, "kurtosis"], 1)]
        for f in fs.index
    ]
    var = r.variance
    md = r.mean_differences.head(eda_top_k)
    md_rows = [[f, _num(row["mean_legitimate"], 3), _num(row["mean_fraud"], 3), f"{row['std_mean_diff']:+.3f}"]
               for f, row in md.iterrows()]
    tc = r.target_correlation
    tc_rows = [[f, f"{v:+.4f}"] for f, v in tc.head(eda_top_k).items()]
    pair_rows = [[p.feature_a, p.feature_b, f"{p.correlation:+.3f}"] for p in r.top_correlated_pairs.itertuples()]
    max_r, a, b = _v_pair_max_abs_corr(r.feature_correlation)
    collinearity = "low" if max_r < LOW_COLLINEARITY_ABS_R else "non-negligible"

    observations = [
        f"V-feature means are ≈ 0 (max |mean| = {fs['mean'].abs().max():.3g}), consistent with "
        "centred PCA outputs. Heavy scaling is unnecessary, and `StandardScaler` is harmless for trees "
        "and useful for Logistic Regression (DOC-02 §7).",
        f"Variance decreases from {var['max_feature']} ({var['variances'][var['max_feature']]:.3f}) to "
        f"{var['min_feature']} ({var['variances'][var['min_feature']]:.4f}); the ordering is "
        f"{'monotonic, as expected for PCA components' if var['monotonic_non_increasing'] else 'not strictly monotonic after deduplication'}.",
        f"Many V-features are heavy-tailed (kurtosis > 10 for "
        f"{int((fs['kurtosis'] > 10).sum())} of {len(fs)}). Extreme values are retained, since they "
        "may be the fraud signal (TD-02).",
        f"Separation is concentrated in a subset of features: the top {len(md)} have |effect| between "
        f"{md['std_mean_diff'].abs().min():.2f} σ and {md['std_mean_diff'].abs().max():.2f} σ, while "
        f"{int((r.mean_differences['std_mean_diff'].abs() < 0.1).sum())} V-features have |effect| < 0.1 σ. "
        "The KDE grid shows that fraud differs in spread and shape, not only in location. This "
        "supports including tree models (Random Forest, XGBoost) alongside Logistic Regression "
        "(DOC-02 §9).",
        f"Collinearity among V-features is {collinearity} (max |r| = {max_r:.3f}, {a}–{b}), so "
        "Logistic Regression coefficients are interpretable without decorrelation.",
    ]
    return (
        "### 6.1 Summary statistics (V1–V28)\n\n<details><summary>Per-feature statistics</summary>\n\n"
        + markdown_table(["Feature", "Mean", "Std", "Variance", "Min", "Median", "Max", "Skew", "Kurtosis"],
                         summary_rows)
        + "\n\n</details>\n\n### 6.2 Variance analysis\n\n"
        + markdown_table(
            ["Metric", "Value"],
            [
                ["Largest variance", f"{var['max_feature']} ({var['variances'][var['max_feature']]:.3f})"],
                ["Smallest variance", f"{var['min_feature']} ({var['variances'][var['min_feature']]:.4f})"],
                ["Max / min variance ratio", f"{var['max_to_min_ratio']:.1f}"],
                ["Monotonically non-increasing (V1 → V28)", "yes" if var["monotonic_non_increasing"] else "no"],
                ["`Amount` variance", _num(var["amount_variance"], 1)],
            ],
        )
        + "\n\n### 6.3 Class separation\n\n"
        + markdown_table(["Feature", "Mean (legitimate)", "Mean (fraud)", "Std. mean diff"], md_rows)
        + "\n\n"
        + _figure(figures, "feature_mean_differences", "Standardized class-mean difference per V-feature", report_dir)
        + "\n\n"
        + _figure(figures, "top_feature_distributions", "Density by class of the most separating V-features", report_dir)
        + "\n\n### 6.4 Correlation analysis\n\n**Correlation with `Class` (top by |r|)**\n\n"
        + markdown_table(["Feature", "r"], tc_rows)
        + "\n\n"
        + _figure(figures, "correlation_with_class", "Pearson correlation of each feature with Class", report_dir)
        + "\n\n**Strongest feature–feature correlations**\n\n"
        + markdown_table(["Feature A", "Feature B", "r"], pair_rows)
        + "\n\n"
        + _figure(figures, "correlation_heatmap", "Feature correlation matrix", report_dir)
        + "\n\n**Feature observations**\n\n"
        + "\n".join(f"- {o}" for o in observations)
    )


def _risks(r: EdaResults) -> list[str]:
    cd, ts = r.class_distribution, r.time_summary
    return [
        f"**Few positives.** {_int(cd['fraud_count'])} frauds in total means validation/test metrics "
        "move in coarse steps and are noisy; small differences between models may not be meaningful.",
        "**Anonymised features.** V1–V28 have no semantic meaning; findings cannot be turned into "
        "domain features, and feature importances are not business-interpretable (DOC-02 §2).",
        f"**Short time window.** {ts['span_hours']:.0f} hours of data; any time-related pattern is "
        "short-horizon and must not be read as production drift (DOC-01 L-01, D-13).",
        f"**Prevalence shifts over time.** Per-bin fraud rate ranges from {_pct(ts['bin_fraud_rate_min'])} "
        f"to {_pct(ts['bin_fraud_rate_max'])}; the temporal holdout's prevalence will differ from the "
        "random test's and must be reported alongside PR-AUC (DOC-01 L-03).",
        "**Linear, marginal statistics.** Correlations and mean differences capture only univariate, "
        "linear association; low |r| does not imply a feature is useless to nonlinear models.",
        "**Descriptive only.** EDA is computed on the full deduplicated dataset (before splitting) "
        "and must not drive fitted choices such as feature selection or scaling parameters; doing so "
        "would leak test information (DOC-02 §4–5).",
    ]


def _recommendations(r: EdaResults, schema: SchemaConfig) -> list[str]:
    cd = r.class_distribution
    return [
        f"Carve the temporal holdout **before** the primary split: stable-sort by `{schema.ordering_column}` "
        "(mergesort), cut at `floor(0.85 · n)`, move boundary ties into the holdout, and do not "
        "stratify the holdout (DOC-02 §21.3).",
        "Record the holdout's fraud count and prevalence; flag low confidence if it holds < 20 frauds "
        "(DOC-02 §21.3 rule 6). The per-bin fraud-rate variation above makes this check important.",
        f"Use a **stratified** 70/15/15 split of the development pool with seed 42. With prevalence "
        f"{cd['fraud_percentage']:.3f}%, unstratified splits could leave too few frauds in validation/test.",
        f"Exclude `{schema.ordering_column}` and `{schema.target}` from the feature set; use exactly the "
        "29 configured features (V1–V28, Amount).",
        "Build the preprocessing `ColumnTransformer` exactly as designed: median imputer + "
        "`StandardScaler` for V1–V28 and median imputer + `RobustScaler` for `Amount`, fitted on "
        "train only. Keep `log1p(Amount)` off by default (TD-05).",
        "Do not remove outliers, and do not apply feature selection based on this EDA.",
        "Add split tests: ratios, stratified prevalence within tolerance, `max(Time_pool) ≤ "
        "min(Time_holdout)`, boundary ties, and zero row overlap across the four partitions (DOC-05 M3).",
    ]


def render_report(
    results: EdaResults,
    figures: Mapping[str, Path],
    report_dir: Path,
    schema: SchemaConfig,
    source_path: Path,
    top_k_features: int,
) -> str:
    """Assemble the full Markdown EDA report."""
    sections = [
        "# Exploratory Data Analysis: Credit Card Fraud Detection",
        f"> Descriptive analysis of the validated, deduplicated dataset `{source_path.as_posix()}`. "
        "Nothing in this report is fitted or used as a pipeline input (DOC-02 §4, DOC-03 MD-16). "
        "Regenerate with `python -m fraud_detection.eda`.",
        "## Executive summary",
        _executive_summary(results),
        "## Key findings",
        "\n".join(f"{i}. {f}" for i, f in enumerate(_key_findings(results), start=1)),
        "## 1. Dataset overview",
        _overview_section(results),
        "## 2. Data quality (M1 reconciliation)",
        _quality_section(results),
        "## 3. Class distribution and fraud imbalance",
        _class_section(results, figures, report_dir),
        "## 4. Transaction amount behaviour",
        _amount_section(results, figures, report_dir),
        f"## 5. Chronology (`{schema.ordering_column}`, descriptive only)",
        _time_section(results, figures, report_dir, schema),
        "## 6. Feature observations",
        _feature_section(results, figures, report_dir, top_k_features),
        "## 7. Risks and limitations",
        "\n".join(f"- {x}" for x in _risks(results)),
        "## 8. Recommendations for M3 (split and preprocessing)",
        "These follow the design already fixed in DOC-02/DOC-05. The EDA supports them; it does not change them.",
        "\n".join(f"{i}. {x}" for i, x in enumerate(_recommendations(results, schema), start=1)),
    ]
    return "\n\n".join(sections) + "\n"
