"""Deterministic baseline strategies.

Every baseline emits a full AnswerSpec per scenario and flows through the
exact grading path a model does. Two purposes:
- `oracle` is the harness self-test: it must score 100% on every dimension;
  anything less is a bug in authoring or grading, never in the baseline.
- The trivial strategies are the floor: a model must beat all of them before
  any capability claim means anything.
"""
from __future__ import annotations

from typing import Callable

from .oracle import derive_gold, solve_selection_for
from .schema import (
    Action,
    ActionChoiceSpec,
    AnswerSpec,
    ClusterLabels,
    ConfidenceLabel,
    DecisionStatus,
    MaterialityLabel,
    Scenario,
    SelectionSpec,
    SignalType,
    SlotDecisionSpec,
)


def _topo_sequence(scenario: Scenario, selected: list[str]) -> list[str]:
    """Dependency-respecting order over the selected set (Kahn, deterministic)."""
    chosen = set(selected)
    by_id = {c.id: c for c in scenario.candidates}
    deps = {
        cid: sorted(set(by_id[cid].depends_on) & chosen) if cid in by_id else []
        for cid in sorted(chosen)
    }
    order: list[str] = []
    remaining = dict(deps)
    while remaining:
        ready = sorted(c for c, d in remaining.items() if all(x in order for x in d))
        if not ready:  # cycle — emit deterministically anyway
            ready = sorted(remaining)
        order.append(ready[0])
        del remaining[ready[0]]
    return order


def _labels(signal: SignalType, conf: ConfidenceLabel, mat: MaterialityLabel,
            scenario: Scenario) -> dict[str, ClusterLabels]:
    return {
        c.id: ClusterLabels(signal_type=signal, confidence=conf, materiality=mat)
        for c in scenario.clusters
    }


def _decide_common(scenario: Scenario, selected: list[str]) -> dict:
    coverage = {
        sc.id: sorted(
            {c.id for c in scenario.candidates if sc.id in c.addresses}
            & set(selected)
        )
        for sc in scenario.success_criteria
    }
    return {
        "selected": sorted(selected),
        "sequence": _topo_sequence(scenario, selected),
        "coverage": {k: v for k, v in coverage.items() if v},
    }


def oracle_answer(scenario: Scenario) -> AnswerSpec:
    derived = derive_gold(scenario)
    if derived.must_clarify:
        return AnswerSpec(
            decision_status=DecisionStatus.CLARIFY,
            questions=sorted(derived.required_questions),
        )
    outcome = derived.outcome
    assert outcome is not None
    labels = {
        cid: ClusterLabels(
            signal_type=SignalType(dims["signal_type"]),
            confidence=ConfidenceLabel(dims["confidence"]),
            materiality=MaterialityLabel(dims["materiality"]),
        )
        for cid, dims in derived.cluster_labels.items()
    }
    if outcome[0] == "selection":
        _, action, optimal_sets = outcome
        selected = sorted(min(optimal_sets, key=lambda s: sorted(s)))
        return AnswerSpec(
            decision_status=DecisionStatus.DECIDE,
            action=Action(action),
            cluster_labels=labels,
            **_decide_common(scenario, selected),
        )
    if outcome[0] == "action":
        return AnswerSpec(
            decision_status=DecisionStatus.DECIDE,
            action=Action(sorted(outcome[1])[0]),
            cluster_labels=labels,
        )
    _, action, slot, label = outcome
    return AnswerSpec(
        decision_status=DecisionStatus.DECIDE,
        action=Action(action),
        slots={slot: label},
        cluster_labels=labels,
    )


def _default_action(scenario: Scenario, fallback: Action) -> Action:
    spec = scenario.decision_spec
    if isinstance(spec, SelectionSpec):
        return spec.action
    if isinstance(spec, SlotDecisionSpec):
        return spec.action
    return fallback


def always_clarify(scenario: Scenario) -> AnswerSpec:
    return AnswerSpec(
        decision_status=DecisionStatus.CLARIFY,
        questions=sorted(scenario.question_ids),
    )


def always_decline(scenario: Scenario) -> AnswerSpec:
    return AnswerSpec(decision_status=DecisionStatus.DECIDE, action=Action.DECLINE)


def _slot_default(scenario: Scenario) -> dict[str, str]:
    spec = scenario.decision_spec
    if isinstance(spec, SlotDecisionSpec):
        return {spec.slot: spec.labels[0]}
    return {}


def select_all(scenario: Scenario) -> AnswerSpec:
    return AnswerSpec(
        decision_status=DecisionStatus.DECIDE,
        action=_default_action(scenario, Action.BUILD),
        slots=_slot_default(scenario),
        cluster_labels=_labels(
            SignalType.BROAD_TREND, ConfidenceLabel.SUFFICIENT,
            MaterialityLabel.REVENUE, scenario,
        ),
        **_decide_common(scenario, [c.id for c in scenario.candidates]),
    )


def all_noise(scenario: Scenario) -> AnswerSpec:
    answer = select_all(scenario)
    return answer.model_copy(
        update={
            "cluster_labels": _labels(
                SignalType.DUPLICATE_NOISE, ConfidenceLabel.INSUFFICIENT,
                MaterialityLabel.LOW, scenario,
            )
        }
    )


def greedy_value(scenario: Scenario) -> AnswerSpec:
    """The canonical-framework reflex: rank by value density, fill capacity,
    ignore gates and dependencies."""
    spec = scenario.decision_spec
    if not isinstance(spec, SelectionSpec):
        return select_all(scenario)
    def density(c) -> float:
        value = float(c.facts.get(spec.objective.value_fact, 0))
        if spec.capacity is None:
            return value
        cost = float(c.facts.get(spec.capacity.cost_fact, 1)) or 1.0
        return value / cost
    ranked = sorted(scenario.candidates, key=lambda c: (-density(c), c.id))
    selected: list[str] = []
    spent = 0.0
    for cand in ranked:
        cost = (
            float(cand.facts.get(spec.capacity.cost_fact, 0))
            if spec.capacity is not None
            else 0.0
        )
        if spec.capacity is None or spent + cost <= spec.capacity.budget + 1e-9:
            selected.append(cand.id)
            spent += cost
    if spec.objective.sense == "minimize":
        # greedy minimizer takes nothing unless coverage forces it; take nothing
        selected = []
    return AnswerSpec(
        decision_status=DecisionStatus.DECIDE,
        action=spec.action,
        cluster_labels=_labels(
            SignalType.BROAD_TREND, ConfidenceLabel.SUFFICIENT,
            MaterialityLabel.REVENUE, scenario,
        ),
        **_decide_common(scenario, selected),
    )


def always_build(scenario: Scenario) -> AnswerSpec:
    spec = scenario.decision_spec
    if isinstance(spec, ActionChoiceSpec):
        return AnswerSpec(decision_status=DecisionStatus.DECIDE, action=Action.BUILD)
    return select_all(scenario)


BASELINES: dict[str, Callable[[Scenario], AnswerSpec]] = {
    "oracle": oracle_answer,
    "always-clarify": always_clarify,
    "always-decline": always_decline,
    "always-build": always_build,
    "select-all": select_all,
    "greedy-value": greedy_value,
    "all-noise": all_noise,
}
