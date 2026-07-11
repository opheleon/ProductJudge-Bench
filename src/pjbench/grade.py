"""Deterministic grading: AnswerSpec vs oracle-verified gold. Zero LLM calls.

Grading semantics (per the design review):
- Exactly one principal check per scenario; a wrong branch (decide vs clarify)
  fails the principal check once.
- Downstream checks (labels, sequencing, coverage) are *conditional
  diagnostics*: they are graded only when the model took the gold branch,
  and are otherwise marked not-applicable rather than failed — a branch error
  is not multiply counted across dimensions.
- Selection answers additionally get feasibility and normalized regret,
  reported as diagnostics next to the binary exact-optimum result.
- Cluster labels are scored with balanced accuracy (macro over classes) so
  "label everything noise" cannot exploit class imbalance.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .oracle import selection_regret, solve_selection_for
from .schema import (
    AnswerSpec,
    ClusterLabelsCheck,
    CoverageCheck,
    DecisionStatus,
    GoldCheck,
    PrincipalAction,
    PrincipalClarify,
    PrincipalSelection,
    PrincipalSlot,
    Scenario,
    SelectionSpec,
    SequenceCheck,
    PRINCIPAL_CHECKS,
)


class ItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gold_id: str
    check: str
    dimension: str
    weight: float
    applicable: bool          # False = conditional diagnostic skipped (branch mismatch)
    score: float              # 0..1; binary checks are 0.0/1.0
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.applicable and self.score >= 1.0 - 1e-9


class ScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: str
    family: str
    variant: str
    format_compliant: bool
    branch_correct: bool          # decide-vs-clarify matched gold
    items: list[ItemResult]
    feasible: bool | None = None  # selection scenarios only
    regret: float | None = None   # normalized; None when infeasible/N-A


def _gold_branch(scenario: Scenario) -> DecisionStatus:
    principal = scenario.principal_check
    if isinstance(principal, PrincipalClarify):
        return DecisionStatus.CLARIFY
    return DecisionStatus.DECIDE


def _grade_principal(
    scenario: Scenario, gold: GoldCheck, answer: AnswerSpec
) -> tuple[ItemResult, bool | None, float | None]:
    """Returns (item, feasible, regret) — the latter two for selections only."""
    feasible: bool | None = None
    regret: float | None = None

    if isinstance(gold, PrincipalClarify):
        asked = set(answer.questions)
        required, allowed = set(gold.required_questions), set(gold.allowed_questions)
        ok = (
            answer.decision_status == DecisionStatus.CLARIFY
            and required <= asked <= allowed
        )
        detail = (
            f"asked={sorted(asked)} required={sorted(required)} allowed={sorted(allowed)}"
        )
    elif isinstance(gold, PrincipalSelection):
        selected = frozenset(answer.selected)
        optimal = {frozenset(s) for s in gold.optimal_sets}
        ok = (
            answer.decision_status == DecisionStatus.DECIDE
            and answer.action == gold.action
            and selected in optimal
        )
        detail = f"selected={sorted(selected)} optimal={[sorted(s) for s in optimal]}"
        if answer.decision_status == DecisionStatus.DECIDE:
            spec = scenario.decision_spec
            assert isinstance(spec, SelectionSpec)
            solution = solve_selection_for(scenario)
            feasible, regret = selection_regret(solution, selected, spec.objective.sense)
            detail += f" feasible={feasible} regret={regret}"
    elif isinstance(gold, PrincipalAction):
        ok = (
            answer.decision_status == DecisionStatus.DECIDE
            and answer.action in set(gold.acceptable_actions)
        )
        detail = f"action={answer.action} acceptable={[a.value for a in gold.acceptable_actions]}"
    elif isinstance(gold, PrincipalSlot):
        got = answer.slots.get(gold.slot)
        ok = (
            answer.decision_status == DecisionStatus.DECIDE
            and answer.action == gold.action
            and got in set(gold.acceptable)
        )
        detail = f"slot[{gold.slot}]={got!r} acceptable={gold.acceptable}"
    else:  # pragma: no cover - discriminated union is exhaustive
        raise TypeError(f"not a principal check: {gold.check}")

    item = ItemResult(
        gold_id=gold.id,
        check=gold.check,
        dimension=gold.dimension.value,
        weight=gold.weight,
        applicable=True,
        score=1.0 if ok else 0.0,
        detail=detail,
    )
    return item, feasible, regret


def _balanced_accuracy(expected: list[str], got: list[str | None]) -> float:
    """Macro-averaged per-class recall over the classes present in gold."""
    classes = sorted(set(expected))
    recalls = []
    for cls in classes:
        idx = [i for i, e in enumerate(expected) if e == cls]
        recalls.append(sum(1 for i in idx if got[i] == cls) / len(idx))
    return sum(recalls) / len(recalls) if recalls else 1.0


def _grade_cluster_labels(gold: ClusterLabelsCheck, answer: AnswerSpec) -> ItemResult:
    dims = sorted({d for labels in gold.expected.values() for d in labels})
    per_dim: list[float] = []
    exact = 0
    total = 0
    for dim in dims:
        expected = []
        got: list[str | None] = []
        for cluster_id, labels in sorted(gold.expected.items()):
            expected.append(labels[dim])
            model_labels = answer.cluster_labels.get(cluster_id)
            got.append(
                getattr(model_labels, dim).value if model_labels is not None else None
            )
        per_dim.append(_balanced_accuracy(expected, got))
        exact += sum(1 for e, g in zip(expected, got) if e == g)
        total += len(expected)
    score = sum(per_dim) / len(per_dim) if per_dim else 1.0
    return ItemResult(
        gold_id=gold.id,
        check=gold.check,
        dimension=gold.dimension.value,
        weight=gold.weight,
        applicable=True,
        score=score,
        detail=(
            f"balanced_accuracy={score:.3f} per_dim="
            f"{dict(zip(dims, [round(x, 3) for x in per_dim]))} "
            f"exact={exact}/{total}"
        ),
    )


def _grade_sequence(
    scenario: Scenario, gold: SequenceCheck, answer: AnswerSpec
) -> ItemResult:
    selected = set(answer.selected)
    by_id = {c.id: c for c in scenario.candidates}
    edges = [
        (dep, cid)
        for cid in sorted(selected & set(by_id))
        for dep in by_id[cid].depends_on
        if dep in selected
    ]
    base = dict(
        gold_id=gold.id, check=gold.check, dimension=gold.dimension.value,
        weight=gold.weight,
    )
    if not edges:
        return ItemResult(
            **base, applicable=False, score=0.0,
            detail="no dependency edges among the model's selected set",
        )
    if sorted(answer.sequence) != sorted(selected):
        return ItemResult(
            **base, applicable=True, score=0.0,
            detail="sequence is not a permutation of the selected set",
        )
    position = {cid: i for i, cid in enumerate(answer.sequence)}
    respected = sum(1 for dep, cid in edges if position[dep] < position[cid])
    return ItemResult(
        **base, applicable=True, score=respected / len(edges),
        detail=f"{respected}/{len(edges)} dependency edges respected",
    )


def _grade_coverage(gold: CoverageCheck, answer: AnswerSpec) -> ItemResult:
    selected = set(answer.selected)
    satisfied = 0
    problems = []
    for criterion, acceptable in sorted(gold.criteria.items()):
        claimed = answer.coverage.get(criterion, [])
        valid = set(acceptable) & selected
        if not claimed:
            problems.append(f"{criterion}: no coverage claimed")
        elif not set(claimed) <= valid:
            problems.append(
                f"{criterion}: claims {sorted(set(claimed) - valid)} "
                f"(not acceptable or not selected)"
            )
        else:
            satisfied += 1
    score = satisfied / len(gold.criteria) if gold.criteria else 1.0
    return ItemResult(
        gold_id=gold.id,
        check=gold.check,
        dimension=gold.dimension.value,
        weight=gold.weight,
        applicable=True,
        score=score,
        detail="; ".join(problems) if problems else "all criteria covered",
    )


def grade_scenario(scenario: Scenario, answer: AnswerSpec | None) -> ScenarioResult:
    gold_branch = _gold_branch(scenario)

    if answer is None:  # format-noncompliant: every item is a real failure
        items = [
            ItemResult(
                gold_id=g.id, check=g.check, dimension=g.dimension.value,
                weight=g.weight, applicable=True, score=0.0,
                detail="no valid answer spec",
            )
            for g in scenario.gold
        ]
        return ScenarioResult(
            instance_id=scenario.instance_id, family=scenario.family,
            variant=scenario.variant.value, format_compliant=False,
            branch_correct=False, items=items,
        )

    branch_correct = answer.decision_status == gold_branch
    items: list[ItemResult] = []
    feasible: bool | None = None
    regret: float | None = None

    for gold in scenario.gold:
        if gold.check in PRINCIPAL_CHECKS:
            item, feasible, regret = _grade_principal(scenario, gold, answer)
            items.append(item)
            continue
        if not branch_correct:
            items.append(
                ItemResult(
                    gold_id=gold.id, check=gold.check, dimension=gold.dimension.value,
                    weight=gold.weight, applicable=False, score=0.0,
                    detail=f"conditional diagnostic skipped: model chose "
                    f"{answer.decision_status.value}, gold is {gold_branch.value}",
                )
            )
            continue
        if isinstance(gold, ClusterLabelsCheck):
            items.append(_grade_cluster_labels(gold, answer))
        elif isinstance(gold, SequenceCheck):
            items.append(_grade_sequence(scenario, gold, answer))
        elif isinstance(gold, CoverageCheck):
            items.append(_grade_coverage(gold, answer))
        else:  # pragma: no cover
            raise TypeError(f"unhandled check {gold.check}")

    return ScenarioResult(
        instance_id=scenario.instance_id,
        family=scenario.family,
        variant=scenario.variant.value,
        format_compliant=True,
        branch_correct=branch_correct,
        items=items,
        feasible=feasible,
        regret=regret,
    )
