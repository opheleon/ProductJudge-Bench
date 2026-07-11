"""The decision oracle: derives gold answers from each scenario's decision_spec.

This module is the source of truth for what a correct answer is. Hand-authored
gold blocks in scenario YAML exist for human auditability only — `validate`
re-derives everything here and fails on any disagreement.

All computation is exact enumeration over small spaces (candidate subsets are
capped at 2^15) with the exprs.py policy language; there is no randomness, no
heuristic, and no LLM anywhere.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from .exprs import ExprError, evaluate
from .schema import (
    Action,
    ActionChoiceSpec,
    Candidate,
    Rule,
    Scenario,
    SelectionSpec,
    SlotDecisionSpec,
)

MAX_CANDIDATES = 15
MAX_UNKNOWN_ASSIGNMENTS = 512


class OracleError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Rule evaluation (shared by action_choice, slot_decision, classification)
# ---------------------------------------------------------------------------


def first_match(rules: list[Rule], facts: dict[str, Any], context: str) -> Rule:
    for rule in rules:
        try:
            if evaluate(rule.when, facts):
                return rule
        except ExprError as exc:
            raise OracleError(f"{context}: {exc}") from exc
    raise OracleError(
        f"{context}: no rule matched — rule lists must end with a catch-all 'True'"
    )


# ---------------------------------------------------------------------------
# Selection (knapsack / set-cover under gates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionSolution:
    optimal_sets: frozenset[frozenset[str]]
    optimal_objective: float
    # objective value per feasible set, for regret computation
    feasible_objectives: dict[frozenset[str], float]


def _item_facts(scenario_facts: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    return {**scenario_facts, **candidate.facts}


def solve_selection(
    spec: SelectionSpec,
    facts: dict[str, Any],
    candidates: list[Candidate],
    criteria_map: dict[str, set[str]],
) -> SelectionSolution:
    """Enumerate every subset, apply gates, return all optimal sets.

    criteria_map: success-criterion id -> candidate ids addressing it
    (used by must_cover gates).
    """
    if len(candidates) > MAX_CANDIDATES:
        raise OracleError(f"too many candidates ({len(candidates)} > {MAX_CANDIDATES})")
    by_id = {c.id: c for c in candidates}

    check_deps = any(g.kind == "requires_dependencies" for g in spec.gates)
    must_cover = [c for g in spec.gates if g.kind == "must_cover" for c in g.criteria]

    # predicate gates resolve to concrete item sets here, at solve time
    mandatory: set[str] = set()
    excluded: set[str] = set()
    for gate in spec.gates:
        if gate.kind not in ("mandatory_when", "excluded_when"):
            continue
        target = mandatory if gate.kind == "mandatory_when" else excluded
        for cand in candidates:
            try:
                if evaluate(gate.when, _item_facts(facts, cand)):
                    target.add(cand.id)
            except ExprError as exc:
                raise OracleError(f"gate {gate.when!r} on {cand.id}: {exc}") from exc
    if mandatory & excluded:
        raise OracleError(
            f"items both mandatory and excluded: {sorted(mandatory & excluded)}"
        )

    feasible: dict[frozenset[str], float] = {}
    ids = [c.id for c in candidates]
    for size in range(len(ids) + 1):
        for combo in itertools.combinations(ids, size):
            chosen = frozenset(combo)
            if not mandatory <= chosen or chosen & excluded:
                continue
            if check_deps and any(
                not set(by_id[i].depends_on) <= chosen for i in chosen
            ):
                continue
            if spec.capacity is not None:
                cost = sum(
                    float(by_id[i].facts[spec.capacity.cost_fact]) for i in chosen
                )
                if cost > spec.capacity.budget + 1e-9:
                    continue
            if any(not (criteria_map.get(cr, set()) & chosen) for cr in must_cover):
                continue
            value = sum(
                float(by_id[i].facts[spec.objective.value_fact]) for i in chosen
            )
            feasible[chosen] = value

    if not feasible:
        raise OracleError("no feasible selection exists — the scenario is broken")

    best = (max if spec.objective.sense == "maximize" else min)(feasible.values())
    optimal = frozenset(s for s, v in feasible.items() if abs(v - best) < 1e-9)
    return SelectionSolution(
        optimal_sets=optimal, optimal_objective=best, feasible_objectives=feasible
    )


def selection_regret(
    solution: SelectionSolution, selected: frozenset[str], sense: str
) -> tuple[bool, float | None]:
    """(feasible, normalized regret). Infeasible answers have no regret value."""
    if selected not in solution.feasible_objectives:
        return False, None
    achieved = solution.feasible_objectives[selected]
    best = solution.optimal_objective
    if abs(best) < 1e-9:
        return True, 0.0 if abs(achieved - best) < 1e-9 else 1.0
    if sense == "maximize":
        return True, max(0.0, (best - achieved) / abs(best))
    return True, max(0.0, (achieved - best) / abs(best))


# ---------------------------------------------------------------------------
# Principal outcome per unknown-assignment (drives clarify derivation)
# ---------------------------------------------------------------------------


def _criteria_map(scenario: Scenario) -> dict[str, set[str]]:
    return {
        sc.id: {c.id for c in scenario.candidates if sc.id in c.addresses}
        for sc in scenario.success_criteria
    }


def principal_outcome(scenario: Scenario, assignment: dict[str, Any]) -> tuple:
    """A hashable canonical representation of the optimal principal decision
    under one assignment of unknown facts."""
    facts = {**scenario.facts, **assignment}
    spec = scenario.decision_spec
    if isinstance(spec, SelectionSpec):
        sol = solve_selection(spec, facts, scenario.candidates, _criteria_map(scenario))
        return ("selection", spec.action.value, sol.optimal_sets)
    if isinstance(spec, ActionChoiceSpec):
        rule = first_match(spec.rules, facts, scenario.instance_id)
        if not rule.actions:
            raise OracleError(f"{scenario.instance_id}: matched rule has no actions")
        return ("action", frozenset(a.value for a in rule.actions))
    if isinstance(spec, SlotDecisionSpec):
        rule = first_match(spec.rules, facts, scenario.instance_id)
        if rule.label not in spec.labels:
            raise OracleError(
                f"{scenario.instance_id}: rule label {rule.label!r} not in slot menu"
            )
        return ("slot", spec.action.value, spec.slot, rule.label)
    raise OracleError(f"unknown spec kind {type(spec).__name__}")


@dataclass(frozen=True)
class ClarifyDerivation:
    must_clarify: bool
    required_facts: frozenset[str]   # decision-changing unknowns
    outcome: tuple | None            # the single outcome when not clarifying


def derive_clarify(scenario: Scenario) -> ClarifyDerivation:
    """Value-of-information rule: clarify iff plausible values of the unknowns
    produce different optimal principal decisions."""
    if not scenario.unknowns:
        return ClarifyDerivation(False, frozenset(), principal_outcome(scenario, {}))

    facts_list = [u.fact for u in scenario.unknowns]
    value_lists = [u.admissible_values for u in scenario.unknowns]
    n_assignments = 1
    for v in value_lists:
        n_assignments *= len(v)
    if n_assignments > MAX_UNKNOWN_ASSIGNMENTS:
        raise OracleError(
            f"{scenario.instance_id}: {n_assignments} unknown assignments exceeds cap"
        )

    outcomes: dict[tuple, tuple] = {}
    for values in itertools.product(*value_lists):
        assignment = dict(zip(facts_list, values))
        outcomes[values] = principal_outcome(scenario, assignment)

    distinct = set(outcomes.values())
    if len(distinct) == 1:
        return ClarifyDerivation(False, frozenset(), distinct.pop())

    # a fact is decision-changing iff two assignments differing only in it
    # yield different outcomes
    changing: set[str] = set()
    for i, fact in enumerate(facts_list):
        for values in outcomes:
            for alt in value_lists[i]:
                if alt == values[i]:
                    continue
                other = values[:i] + (alt,) + values[i + 1 :]
                if outcomes[values] != outcomes[other]:
                    changing.add(fact)
                    break
            if fact in changing:
                break
    return ClarifyDerivation(True, frozenset(changing), None)


# ---------------------------------------------------------------------------
# Full gold derivation (what validate compares against the YAML gold block)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedGold:
    must_clarify: bool
    required_questions: frozenset[str]
    allowed_questions: frozenset[str]
    outcome: tuple | None                       # decide-scenario principal outcome
    cluster_labels: dict[str, dict[str, str]]   # cluster -> label dim -> label
    coverage: dict[str, frozenset[str]]         # criterion -> acceptable items


def classify_clusters(scenario: Scenario) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for cluster in scenario.clusters:
        facts = {**scenario.facts, **cluster.facts}
        labels[cluster.id] = {}
        for dim, rules in scenario.label_rules.items():
            rule = first_match(rules, facts, f"{scenario.instance_id}/{cluster.id}/{dim}")
            labels[cluster.id][dim] = rule.label
    return labels


def derive_gold(scenario: Scenario) -> DerivedGold:
    clarify = derive_clarify(scenario)
    fact_to_question = {q.fact: q.id for q in scenario.questions}
    for fact in clarify.required_facts:
        if fact not in fact_to_question:
            raise OracleError(
                f"{scenario.instance_id}: decision-changing unknown {fact!r} has no "
                f"question in the menu"
            )
    required = frozenset(fact_to_question[f] for f in clarify.required_facts)
    # allowed = questions about any genuinely unknown fact; asking about a
    # stated fact means the model didn't read the prompt.
    unknown_facts = {u.fact for u in scenario.unknowns}
    allowed = frozenset(
        q.id for q in scenario.questions if q.fact in unknown_facts
    )
    return DerivedGold(
        must_clarify=clarify.must_clarify,
        required_questions=required,
        allowed_questions=allowed,
        outcome=clarify.outcome,
        cluster_labels=classify_clusters(scenario),
        coverage={k: frozenset(v) for k, v in _criteria_map(scenario).items()},
    )


def solve_selection_for(scenario: Scenario) -> SelectionSolution:
    """Selection solution for a decide-scenario (no unknowns unresolved)."""
    spec = scenario.decision_spec
    if not isinstance(spec, SelectionSpec):
        raise OracleError(f"{scenario.instance_id} is not a selection scenario")
    return solve_selection(
        spec, dict(scenario.facts), scenario.candidates, _criteria_map(scenario)
    )
