"""Rule-based model selection on validation results (DOC-02 §12; DOC-05 M6).

Ordered framework (never a manual choice, DOC-03 MD-11):

1. Eligibility: if the primary threshold objective is used by any candidate, only
   models that reach Recall >= r_min at some validation threshold remain.
2. Primary ranking: validation PR-AUC (descending).
3. Operating-point check: each model's own tuned threshold is compared (Precision at the
   required recall under the primary objective, F2 under the fallback, and FP count).
   It is recorded for every candidate and applied through the tie-break.
4. Tie-break: among models whose PR-AUC is within ``pr_auc_tie_tolerance`` of the leader,
   prefer fewer false positives at the operating point, then the simpler model
   (LR < RF = XGB), then the smaller artifact.
5. Sanity: the selected model's PR-AUC must exceed the Logistic Regression baseline's,
   otherwise Logistic Regression is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fraud_detection.evaluation.threshold import OBJECTIVE_PRIMARY

BASELINE_MODEL = "logistic_regression"
# DOC-02 §12: LR < RF ≈ XGB in complexity.
COMPLEXITY_RANK = {"logistic_regression": 0, "random_forest": 1, "xgboost": 1}
_FLOAT_EPS = 1e-12


@dataclass(frozen=True)
class CandidateSummary:
    """Validation facts about one tuned candidate that the framework needs."""

    name: str
    val_pr_auc: float
    reached_r_min: bool
    objective: str
    threshold: float
    precision: float
    recall: float
    f2: float
    fp: int
    artifact_bytes: int

    @property
    def operating_point_score(self) -> float:
        """Precision at the required recall (primary objective) or F2 (fallback)."""
        return self.precision if self.objective == OBJECTIVE_PRIMARY else self.f2


@dataclass(frozen=True)
class SelectionResult:
    """The selected model plus a record of every step of the framework."""

    selected: str
    eligible: list[str]
    ranking: list[str]
    pr_auc_leader: str
    tie_group: list[str]
    tie_break_applied: bool
    operating_point: dict[str, dict[str, Any]]
    sanity_passed: bool | None
    baseline_fallback: bool
    steps: list[dict[str, str]] = field(default_factory=list)


def _tie_break_key(c: CandidateSummary) -> tuple[int, int, int, str]:
    return (c.fp, COMPLEXITY_RANK.get(c.name, 1), c.artifact_bytes, c.name)


def select_model(candidates: dict[str, CandidateSummary], pr_auc_tie_tolerance: float) -> SelectionResult:
    """Apply DOC-02 §12 to the candidates and return the selection with its rationale.

    Raises:
        ValueError: If there are no candidates.
    """
    if not candidates:
        raise ValueError("No candidates to select from")
    steps: list[dict[str, str]] = []

    # 1. Eligibility
    primary_used = any(c.reached_r_min for c in candidates.values())
    if primary_used:
        eligible = sorted(n for n, c in candidates.items() if c.reached_r_min)
        excluded = sorted(set(candidates) - set(eligible))
        outcome = f"Eligible (reach recall >= r_min): {eligible}" + (f"; excluded: {excluded}" if excluded else "")
    else:
        eligible = sorted(candidates)
        outcome = "No model reaches r_min; F2 fallback applies to all and eligibility is not applied"
    steps.append({"step": "1. Eligibility", "outcome": outcome})

    # 2. Primary ranking by validation PR-AUC
    ranking = sorted(eligible, key=lambda n: (-candidates[n].val_pr_auc, n))
    leader = ranking[0]
    steps.append({
        "step": "2. Primary ranking (validation PR-AUC)",
        "outcome": ", ".join(f"{n} ({candidates[n].val_pr_auc:.4f})" for n in ranking),
    })

    # 3. Operating-point check
    operating_point = {
        n: {
            "objective": candidates[n].objective,
            "threshold": candidates[n].threshold,
            "score": candidates[n].operating_point_score,
            "score_name": "precision" if candidates[n].objective == OBJECTIVE_PRIMARY else "f2",
            "fp": candidates[n].fp,
        }
        for n in ranking
    }
    best_op = max(ranking, key=lambda n: (candidates[n].operating_point_score, -candidates[n].fp, n))
    steps.append({
        "step": "3. Operating-point check",
        "outcome": (
            f"Best operating point: {best_op}; PR-AUC leader: {leader}"
            + ("" if best_op == leader else " (they differ; decided by the tie-break if within tolerance)")
        ),
    })

    # 4. Tie-break among models within the PR-AUC tolerance of the leader
    top = candidates[leader].val_pr_auc
    tie_group = [n for n in ranking if top - candidates[n].val_pr_auc <= pr_auc_tie_tolerance + _FLOAT_EPS]
    if len(tie_group) > 1:
        selected = min(tie_group, key=lambda n: _tie_break_key(candidates[n]))
        outcome = (
            f"PR-AUC within {pr_auc_tie_tolerance} of the leader: {tie_group}; "
            f"fewer FP -> simpler -> smaller artifact selects {selected}"
        )
    else:
        selected = leader
        outcome = f"No other model within {pr_auc_tie_tolerance} PR-AUC of {leader}; no tie-break needed"
    steps.append({"step": "4. Tie-break", "outcome": outcome})

    # 5. Sanity against the Logistic Regression baseline
    baseline_fallback = False
    if BASELINE_MODEL not in candidates:
        sanity_passed = None
        outcome = "Logistic Regression baseline not among candidates; check not applicable"
    elif selected == BASELINE_MODEL:
        sanity_passed = True
        outcome = "Selected model is the Logistic Regression baseline"
    else:
        sanity_passed = candidates[selected].val_pr_auc > candidates[BASELINE_MODEL].val_pr_auc
        if sanity_passed:
            outcome = (
                f"{selected} PR-AUC {candidates[selected].val_pr_auc:.4f} > LR baseline "
                f"{candidates[BASELINE_MODEL].val_pr_auc:.4f}"
            )
        else:
            outcome = f"{selected} does not exceed the LR baseline PR-AUC; Logistic Regression selected"
            selected, baseline_fallback = BASELINE_MODEL, True
    steps.append({"step": "5. LR-baseline sanity", "outcome": outcome})

    return SelectionResult(
        selected=selected,
        eligible=eligible,
        ranking=ranking,
        pr_auc_leader=leader,
        tie_group=tie_group,
        tie_break_applied=len(tie_group) > 1,
        operating_point=operating_point,
        sanity_passed=sanity_passed,
        baseline_fallback=baseline_fallback,
        steps=steps,
    )
