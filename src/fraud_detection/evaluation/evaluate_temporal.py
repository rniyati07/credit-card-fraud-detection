"""Temporal robustness comparison, interpretation and final ``model_config.json`` (DOC-02 §15, §21).

The frozen model and threshold are evaluated once on the chronologically latest
transactions and compared with the primary random test (DOC-02 §21.5):
absolute Δ = temporal − test, relative Δ = (temporal − test) / test (omitted when the test
value is 0). Interpretation follows DOC-02 §21.6:

* a decrease is evidence of *possible short-horizon temporal/distribution shift* within the
  2-day capture window, never proof of drift in a live deployment;
* differences within the bootstrap CIs (intervals overlap) or within a few fraud cases are
  inconclusive;
* recall is also stated in fraud-case terms;
* results never feed back into model choice, threshold or training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fraud_detection.config import Config
from fraud_detection.evaluation.evaluate import EvaluationError, FrozenModel, r_min_check

logger = logging.getLogger(__name__)

SHIFT_WORDING = "possible short-horizon temporal/distribution shift"
FULL_DELTA_METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "f2")
COUNT_METRICS = ("tp", "fp", "tn", "fn")
RATE_METRICS = ("fp_rate", "fn_rate")
CI_METRICS = ("pr_auc", "recall", "precision")
NO_DECREASE = "no decrease"
INCONCLUSIVE = "decrease within sampling noise (inconclusive)"
SHIFT = f"decrease beyond sampling noise: evidence of {SHIFT_WORDING}"


def load_test_metrics(path: Path, frozen: FrozenModel) -> dict[str, Any]:
    """Primary test results written by ``pipeline.evaluate``; must belong to the same frozen model."""
    if not path.is_file():
        raise EvaluationError(
            f"Test metrics not found at '{path}'. Run `python -m fraud_detection.pipeline.evaluate` first."
        )
    test = json.loads(path.read_text(encoding="utf-8"))
    if test.get("model_sha256") != frozen.metadata["model_sha256"] or test.get("threshold") != frozen.metadata["threshold"]:
        raise EvaluationError(f"'{path}' was produced for a different frozen model/threshold; rerun pipeline.evaluate")
    return test


def load_split_summary(path: Path) -> dict[str, Any] | None:
    """Temporal carve-out summary from M3 (pool/holdout counts, prevalence, time ranges)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Temporal split summary not available at %s", path)
        return None


def _value(result: dict[str, Any], metric: str) -> float:
    if metric in result["at_threshold"]:
        return float(result["at_threshold"][metric])
    if metric in result["rates"]:
        return float(result["rates"][metric])
    return float(result[metric])


def _relative(delta: float, base: float) -> float | None:
    return delta / base if base != 0 else None


def comparison_rows(temporal: dict[str, Any], test: dict[str, Any]) -> list[dict[str, Any]]:
    """The DOC-02 §21.5 comparison table (temporal holdout vs primary random test)."""
    rows: list[dict[str, Any]] = []

    def add(metric: str, abs_delta: bool, rel_delta: bool, note: str = "") -> None:
        t, r = _value(temporal, metric), _value(test, metric)
        delta = t - r
        row = {
            "metric": metric,
            "temporal_holdout": t,
            "random_test": r,
            "abs_delta": delta if abs_delta else None,
            "rel_delta": _relative(delta, r) if rel_delta else None,
            "note": note,
        }
        if metric in CI_METRICS and "bootstrap" in temporal and "bootstrap" in test:
            tci, rci = temporal["bootstrap"][metric], test["bootstrap"][metric]
            row |= {
                "temporal_ci_low": tci["low"], "temporal_ci_high": tci["high"],
                "test_ci_low": rci["low"], "test_ci_high": rci["high"],
                "cis_overlap": tci["low"] <= rci["high"] and rci["low"] <= tci["high"],
            }
        rows.append(row)

    for metric in FULL_DELTA_METRICS:
        add(metric, True, True)
    for metric in COUNT_METRICS:
        add(metric, False, False, "sizes differ: counts reported side by side")
    for metric in RATE_METRICS:
        add(metric, True, False)
    add("prevalence", False, False, "context: read PR-AUC against each set's prevalence")
    add("pr_auc_lift", True, False, "PR-AUC / prevalence (lift over no-skill)")
    return rows


def interpret(
    temporal: dict[str, Any], rows: list[dict[str, Any]], inconclusive_fraud_cases: int
) -> dict[str, Any]:
    """Per-metric verdicts under the DOC-02 §21.6 rules, plus recall in fraud-case terms."""
    by_metric = {r["metric"]: r for r in rows}
    n_fraud_temporal = temporal["r_min_check"]["frauds_total"]
    verdicts: dict[str, dict[str, Any]] = {}
    for metric in CI_METRICS:
        row = by_metric[metric]
        delta = row["abs_delta"]
        reasons = []
        if delta >= 0:
            verdict = NO_DECREASE
        else:
            if row.get("cis_overlap"):
                reasons.append("bootstrap CIs overlap")
            if metric == "recall" and abs(delta) * n_fraud_temporal <= inconclusive_fraud_cases + 1e-9:
                reasons.append(f"within {inconclusive_fraud_cases} fraud cases")
            verdict = INCONCLUSIVE if reasons else SHIFT
        verdicts[metric] = {"abs_delta": delta, "verdict": verdict, "reasons": reasons}
    shift_metrics = [m for m, v in verdicts.items() if v["verdict"] == SHIFT]
    decreased = [m for m, v in verdicts.items() if v["abs_delta"] < 0]
    if shift_metrics:
        overall = f"decrease beyond sampling noise in {', '.join(shift_metrics)}: evidence of {SHIFT_WORDING}"
    elif decreased:
        overall = (
            f"lower point estimates on the holdout for {', '.join(decreased)}, but every decrease is within "
            "sampling noise (inconclusive at this sample size): no evidence beyond noise of "
            f"{SHIFT_WORDING}"
        )
    else:
        overall = "no decrease: holdout metrics are equal to or higher than on the random test"
    return {
        "verdicts": verdicts,
        "recall_delta_in_holdout_fraud_cases": by_metric["recall"]["abs_delta"] * n_fraud_temporal,
        "metrics_decreased": decreased,
        "metrics_with_shift_evidence": shift_metrics,
        "overall": overall,
    }


def prevalence_context(temporal: dict[str, Any], test: dict[str, Any], split_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Holdout fraud prevalence against the development pool and the random test (DOC-01 L-03)."""
    out: dict[str, Any] = {
        "holdout_prevalence": temporal["prevalence"],
        "test_prevalence": test["prevalence"],
        "holdout_minus_test": temporal["prevalence"] - test["prevalence"],
    }
    if split_summary is not None:
        pool = split_summary["development_pool"]["prevalence"]
        out |= {
            "development_pool_prevalence": pool,
            "holdout_minus_pool": temporal["prevalence"] - pool,
            "holdout_to_pool_ratio": temporal["prevalence"] / pool if pool else None,
            "holdout_low_confidence": split_summary.get("low_confidence"),
            "min_fraud_warning": split_summary.get("min_fraud_warning"),
        }
    return out


# ------------------------------------------------------------------------------ report


def _fmt(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:+.{digits}f}%"


def _cases(check: dict[str, Any]) -> str:
    return (
        f"detected {check['frauds_detected']} of {check['frauds_total']} frauds "
        f"(missed {check['frauds_missed']}); r_min {check['r_min']} needs {check['frauds_required_for_r_min']} "
        f"-> margin {check['margin_in_fraud_cases']:+d} fraud case(s)"
    )


def render_report(
    frozen: FrozenModel,
    temporal: dict[str, Any],
    test: dict[str, Any],
    rows: list[dict[str, Any]],
    interpretation: dict[str, Any],
    prevalence: dict[str, Any],
    split_summary: dict[str, Any] | None,
    config: Config,
) -> str:
    """``temporal_robustness.md``: facts from computed metrics only (deterministic)."""
    m = frozen.metadata
    val_check = m["validation_metrics"]["at_threshold"]
    val_required_margin = r_min_check(val_check, float(m["r_min"]))["margin_in_fraud_cases"]
    tc, hc = test["r_min_check"], temporal["r_min_check"]
    verdicts = interpretation["verdicts"]
    lines = [
        "# Temporal Robustness Assessment",
        "",
        f"> Secondary, evaluation-only experiment (DOC-02 §21). The frozen `{m['model_name']}` "
        f"(version `{m['model_version']}`) and its frozen threshold {m['threshold']:.10f} were evaluated once on the "
        "chronologically latest transactions and compared with the primary random test. Nothing was retrained, "
        "retuned or reselected; these results never feed back into the model, threshold or r_min "
        "(DOC-02 §21.6). Regenerate with `python -m fraud_detection.pipeline.evaluate_temporal`.",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Model / version | `{m['model_name']}` / `{m['model_version']}` |",
        f"| Frozen threshold | {m['threshold']:.10f} ({m['threshold_objective']}) |",
        f"| r_min | {m['r_min']} ({m.get('r_min_status', 'n/a')}) |",
        f"| Temporal holdout | {temporal['n_rows']:,} rows, {temporal['n_fraud']} frauds, prevalence {temporal['prevalence']:.5f} |",
        f"| Primary random test | {test['n_rows']:,} rows, {test['n_fraud']} frauds, prevalence {test['prevalence']:.5f} |",
    ]
    if split_summary is not None:
        pool, hold = split_summary["development_pool"], split_summary["temporal_holdout"]
        lines += [
            f"| Development pool (M3) | {pool['rows']:,} rows, {pool['fraud_count']} frauds, prevalence {pool['prevalence']:.5f}, Time {pool['time_min']:.0f}-{pool['time_max']:.0f} s |",
            f"| Holdout time range | {hold['time_min']:.0f}-{hold['time_max']:.0f} s (strictly after the pool) |",
            f"| Low-confidence flag (< {split_summary.get('min_fraud_warning')} frauds) | {'yes' if split_summary.get('low_confidence') else 'no'} |",
        ]
    lines += [
        "",
        "## Comparison (DOC-02 §21.5)",
        "",
        "| Metric | Temporal holdout | Random test | Abs Δ | Rel Δ | Note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        count = r["metric"] in COUNT_METRICS
        fmt = (lambda v: f"{int(v):,}") if count else (lambda v: _fmt(v, 5 if r["metric"] in ("prevalence", "fp_rate") else 4))
        lines.append(
            f"| {r['metric']} | {fmt(r['temporal_holdout'])} | {fmt(r['random_test'])} | "
            f"{_fmt(r['abs_delta'], 5) if r['abs_delta'] is not None else '-'} | "
            f"{_pct(r['rel_delta']) if r['rel_delta'] is not None else '-'} | {r['note']} |"
        )
    if "bootstrap" in temporal:
        b = temporal["bootstrap"]
        lines += [
            "",
            f"Bootstrap {int(b['confidence'] * 100)}% CIs ({b['n']} seeded resamples per set):",
            "",
            "| Metric | Temporal holdout CI | Random test CI | Overlap |",
            "|---|---|---|---|",
        ]
        for r in rows:
            if "temporal_ci_low" in r:
                lines.append(
                    f"| {r['metric']} | [{r['temporal_ci_low']:.4f}, {r['temporal_ci_high']:.4f}] | "
                    f"[{r['test_ci_low']:.4f}, {r['test_ci_high']:.4f}] | {'yes' if r['cis_overlap'] else 'no'} |"
                )
    lines += [
        "",
        "Confusion matrices at the frozen threshold (rows: actual legitimate/fraud; columns: predicted legitimate/fraud):",
        "",
        f"- Temporal holdout: {temporal['at_threshold']['confusion_matrix']}",
        f"- Random test: {test['at_threshold']['confusion_matrix']}",
        "",
        "## Answers",
        "",
        "**1. Does the model generalise to future transactions?**",
        "",
        f"On the latest 15% of this 2-day capture: PR-AUC {temporal['pr_auc']:.4f} vs {test['pr_auc']:.4f} on the random test "
        f"(lift over no-skill {temporal['pr_auc_lift']:.0f}x vs {test['pr_auc_lift']:.0f}x), recall {temporal['at_threshold']['recall']:.4f} "
        f"vs {test['at_threshold']['recall']:.4f}, precision {temporal['at_threshold']['precision']:.4f} vs "
        f"{test['at_threshold']['precision']:.4f}. Overall: {interpretation['overall']}. Holdout recall is "
        f"{'at or above' if temporal['r_min_check']['meets_r_min'] else 'below'} r_min = {m['r_min']} "
        f"(margin {temporal['r_min_check']['margin_in_fraud_cases']:+d} fraud case(s)). This covers hours, not "
        "months, so it indicates short-horizon behaviour only (DOC-01 L-01).",
        "",
        "**2. How much did performance change?** (temporal − test)",
        "",
    ]
    for metric in CI_METRICS:
        v = verdicts[metric]
        rel = next(r["rel_delta"] for r in rows if r["metric"] == metric)
        lines.append(f"- {metric}: {v['abs_delta']:+.4f} ({_pct(rel)}) -> {v['verdict']}"
                     + (f" ({'; '.join(v['reasons'])})" if v["reasons"] else ""))
    lines += [
        f"- Recall change expressed in holdout fraud cases: {interpretation['recall_delta_in_holdout_fraud_cases']:+.1f}.",
        "",
        "**3. Is the degradation acceptable according to the project requirements?**",
        "",
        "DOC-01 SC-10 mandates no minimum temporal performance; the requirement is that the holdout is evaluated "
        "once with the frozen model and threshold and that the comparison, prevalence and interpretation are "
        "reported. Those requirements are met by this report. Whether the observed change is acceptable for a "
        "deployment is a business decision outside the documented criteria.",
        "",
        "**4. Did recall stay near the M6 target?**",
        "",
        f"- Validation (M6): recall {val_check['recall']:.4f}; detected {val_check['tp']} of "
        f"{val_check['tp'] + val_check['fn']}; margin over r_min {val_required_margin:+d} fraud case(s).",
        f"- Random test: recall {tc['recall']:.4f}; {_cases(tc)}; "
        f"{'meets' if tc['meets_r_min'] else 'below'} r_min.",
        f"- Temporal holdout: recall {hc['recall']:.4f}; {_cases(hc)}; "
        f"{'meets' if hc['meets_r_min'] else 'below'} r_min.",
        "",
        "**5. Does the M6 plateau-edge limitation appear in practice?**",
        "",
        "M6 recorded that the frozen threshold sits at the upper edge of the validation recall plateau (validation "
        f"margin over r_min: {val_required_margin:+d} fraud case(s)), so unseen data could fall below r_min by one "
        "or two frauds (DOC-05 §23 DEC-01). "
        + ("It does **not** appear on either set: both meet r_min." if tc["meets_r_min"] and hc["meets_r_min"]
           else "It **does** appear: " + "; ".join(
               f"{name} recall {c['recall']:.4f} is {-c['margin_in_fraud_cases']} fraud case(s) short of r_min"
               for name, c in (("random test", tc), ("temporal holdout", hc)) if not c["meets_r_min"]
           ) + ". Reported as-is; the threshold is not retuned (DOC-02 §14)."),
        "",
        "## Prevalence context (DOC-01 L-03)",
        "",
        f"- Holdout prevalence {prevalence['holdout_prevalence']:.5f} vs random test {prevalence['test_prevalence']:.5f} "
        f"(difference {prevalence['holdout_minus_test']:+.5f}).",
    ]
    if "development_pool_prevalence" in prevalence:
        lines.append(
            f"- Holdout prevalence vs development pool {prevalence['development_pool_prevalence']:.5f}: difference "
            f"{prevalence['holdout_minus_pool']:+.5f} (ratio {prevalence['holdout_to_pool_ratio']:.2f}). The no-skill "
            "PR-AUC baseline differs accordingly, so PR-AUC is read together with its lift."
        )
    lines += [
        "",
        "## Interpretation rules and limitations",
        "",
        f"- A decrease on the temporal holdout is treated as evidence of {SHIFT_WORDING} within the capture window, "
        "not as proof of drift in a live deployment (DOC-02 §21.6, D-13).",
        f"- Decreases whose {int(config.evaluation.bootstrap_confidence * 100)}% bootstrap CIs overlap, or recall changes "
        f"within {config.evaluation.inconclusive_fraud_cases} fraud cases, are treated as inconclusive. Overlapping "
        "intervals are a conservative criterion: 'inconclusive' means the data cannot distinguish the change from "
        "sampling noise, not that no change occurred.",
        f"- Few frauds ({temporal['n_fraud']} in the holdout, {test['n_fraud']} in the test set) make every metric "
        "high-variance (DOC-01 L-02)."
        + (f" The holdout has fewer than {split_summary.get('min_fraud_warning')} frauds, so its results are "
           "flagged **low-confidence** (DOC-02 §21.3 rule 6)."
           if split_summary is not None and split_summary.get("low_confidence") else ""),
        "- `Time` was used only to order transactions; it is not a model feature (DOC-02 §21.2).",
        "- The model, threshold and r_min were frozen before this evaluation and are not changed by it.",
        "",
    ]
    return "\n".join(lines)


def build_model_config(
    frozen: FrozenModel,
    test_metrics: dict[str, Any],
    temporal_metrics: dict[str, Any],
    trained_at: str | None,
    evaluation_lineage: dict[str, Any],
) -> dict[str, Any]:
    """Final ``model_config.json``: DOC-02 §15 schema + DOC-03 §7 lineage keys.

    Identity, threshold and feature order are copied verbatim from ``train_metadata.json``;
    nothing is recomputed (DOC-03 §4.2).
    """
    m = frozen.metadata
    return {
        "model_name": m["model_name"],
        "model_version": m["model_version"],
        "threshold": m["threshold"],
        "threshold_objective": m["threshold_objective"],
        "r_min": m["r_min"],
        "feature_order": m["feature_order"],
        "ordering_column": m["ordering_column"],
        "temporal_holdout_fraction": m["temporal_holdout_fraction"],
        "target": m["target"],
        "positive_label": m["positive_label"],
        "trained_at": trained_at,
        "random_seed": m["random_seed"],
        "data_hash": m["data_hash"],
        "library_versions": m["library_versions"],
        "validation_metrics": m["validation_metrics"],
        "test_metrics": test_metrics,
        "temporal_holdout_metrics": temporal_metrics,
        # Lineage (DOC-03 §7)
        "git_commit": m["git_commit"],
        "mlflow_run_id": m["mlflow_run_id"],
        "data_dvc_md5": m["data_dvc_md5"],
        "config_hash": m["config_hash"],
        "python_version": m["python_version"],
        "model_sha256": m["model_sha256"],
        "evaluation_lineage": evaluation_lineage,
    }
