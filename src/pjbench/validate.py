"""Authoring-bar linter.

The core rule: the YAML gold block is a human-audit artifact, the oracle is
the truth. `validate_scenario` re-derives everything from the decision_spec
and fails on ANY disagreement — a scenario whose hand-authored gold does not
match its own executable policy is broken by definition (this is the check
that would have caught SysDesign's delivery-promise contradiction).

Also enforced: no gold/trap/unknown material leaks into the rendered prompt,
haystack facts really appear in the evidence, rule lists terminate, and
matched pairs behave (counterfactual pairs flip, comparable variants agree).
"""
from __future__ import annotations

from collections import defaultdict

from .exprs import ExprError, expr_fact_names
from .oracle import OracleError, derive_gold, principal_outcome
from .prompt import _fmt_value, build_prompt
from .schema import (
    ActionChoiceSpec,
    ClusterLabelsCheck,
    CoverageCheck,
    PrincipalAction,
    PrincipalClarify,
    PrincipalSelection,
    PrincipalSlot,
    Rule,
    Scenario,
    SelectionSpec,
    SequenceCheck,
    SlotDecisionSpec,
    Variant,
)

MIN_GOLD_RATIONALE_CHARS = 200


def _rule_lists(scenario: Scenario) -> list[tuple[str, list[Rule]]]:
    out: list[tuple[str, list[Rule]]] = []
    spec = scenario.decision_spec
    if isinstance(spec, (ActionChoiceSpec, SlotDecisionSpec)):
        out.append(("decision_spec.rules", spec.rules))
    for dim, rules in scenario.label_rules.items():
        out.append((f"label_rules.{dim}", rules))
    return out


def validate_scenario(scenario: Scenario) -> list[str]:
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"{scenario.instance_id}: {msg}")

    # --- structural ---------------------------------------------------------
    try:
        principal = scenario.principal_check
    except ValueError as exc:
        return [str(exc)]

    if len(scenario.candidate_ids) != len(scenario.candidates):
        err("duplicate candidate ids")
    if len(scenario.cluster_ids) != len(scenario.clusters):
        err("duplicate cluster ids")
    if len(scenario.question_ids) != len(scenario.questions):
        err("duplicate question ids")
    gold_ids = [g.id for g in scenario.gold]
    if len(set(gold_ids)) != len(gold_ids):
        err("duplicate gold ids")

    constraint_ids = {c.id for c in scenario.constraints}
    criteria_ids = {sc.id for sc in scenario.success_criteria}
    unknown_facts = {u.fact for u in scenario.unknowns}
    known_fact_names = (
        set(scenario.facts) | unknown_facts | set(scenario.fact_definitions)
    )

    for q in scenario.questions:
        if q.fact not in known_fact_names:
            err(f"question {q.id}: fact {q.fact!r} is neither stated, defined, "
                f"nor a declared unknown")
    for u in scenario.unknowns:
        if u.fact in scenario.facts:
            err(f"unknown {u.fact!r} is also a stated fact")

    if isinstance(scenario.decision_spec, SelectionSpec):
        for gate in scenario.decision_spec.gates:
            if gate.kind in ("mandatory_when", "excluded_when") and not gate.basis:
                err(f"gate `{gate.when}` needs a basis constraint id")
            if gate.basis and gate.basis not in constraint_ids:
                err(f"gate basis {gate.basis!r} is not a constraint id")
            for cr in gate.criteria:
                if cr not in criteria_ids:
                    err(f"must_cover criterion {cr!r} unknown")
    for name, rules in _rule_lists(scenario):
        if not rules:
            err(f"{name}: empty rule list")
        elif rules[-1].when.strip() != "True":
            err(f"{name}: must end with a catch-all 'True' rule")
        for rule in rules:
            try:
                expr_fact_names(rule.when)
            except (ExprError, SyntaxError) as exc:
                err(f"{name}: {exc}")

    # --- oracle equality (the load-bearing check) ---------------------------
    try:
        derived = derive_gold(scenario)
    except (OracleError, ExprError) as exc:
        err(f"oracle failure: {exc}")
        return errors

    if derived.must_clarify:
        if not isinstance(principal, PrincipalClarify):
            err(
                f"oracle says the unknowns are decision-changing (clarify), but the "
                f"principal check is {principal.check}"
            )
        else:
            if set(principal.required_questions) != set(derived.required_questions):
                err(
                    f"required_questions mismatch: gold "
                    f"{sorted(principal.required_questions)} vs oracle "
                    f"{sorted(derived.required_questions)}"
                )
            if set(principal.allowed_questions) != set(derived.allowed_questions):
                err(
                    f"allowed_questions mismatch: gold "
                    f"{sorted(principal.allowed_questions)} vs oracle "
                    f"{sorted(derived.allowed_questions)}"
                )
    else:
        outcome = derived.outcome
        if isinstance(principal, PrincipalClarify):
            err("principal check is clarify but the oracle says the decision is "
                "determined for every admissible unknown value")
        elif isinstance(principal, PrincipalSelection):
            if outcome is None or outcome[0] != "selection":
                err("principal_selection on a non-selection decision_spec")
            else:
                _, action, optimal_sets = outcome
                if principal.action.value != action:
                    err(f"gold action {principal.action.value!r} != spec action {action!r}")
                gold_sets = {frozenset(s) for s in principal.optimal_sets}
                if gold_sets != set(optimal_sets):
                    err(
                        f"optimal_sets mismatch: gold "
                        f"{sorted(sorted(s) for s in gold_sets)} vs oracle "
                        f"{sorted(sorted(s) for s in optimal_sets)}"
                    )
        elif isinstance(principal, PrincipalAction):
            if outcome is None or outcome[0] != "action":
                err("principal_action on a non-action_choice decision_spec")
            elif {a.value for a in principal.acceptable_actions} != set(outcome[1]):
                err(
                    f"acceptable_actions mismatch: gold "
                    f"{sorted(a.value for a in principal.acceptable_actions)} vs "
                    f"oracle {sorted(outcome[1])}"
                )
        elif isinstance(principal, PrincipalSlot):
            if outcome is None or outcome[0] != "slot":
                err("principal_slot on a non-slot_decision decision_spec")
            else:
                _, action, slot, label = outcome
                if principal.action.value != action:
                    err(f"gold action {principal.action.value!r} != spec action {action!r}")
                if principal.slot != slot:
                    err(f"gold slot {principal.slot!r} != spec slot {slot!r}")
                if set(principal.acceptable) != {label}:
                    err(
                        f"slot acceptable mismatch: gold {sorted(principal.acceptable)} "
                        f"vs oracle [{label!r}]"
                    )

    for gold in scenario.gold:
        if isinstance(gold, ClusterLabelsCheck):
            if gold.expected != derived.cluster_labels:
                err(
                    f"cluster_labels mismatch: gold {gold.expected} vs oracle "
                    f"{derived.cluster_labels}"
                )
        elif isinstance(gold, CoverageCheck):
            # coverage is only gradable over criteria every optimal set must
            # cover (must_cover gates) — anything wider would fail answers the
            # objective never required to cover those criteria
            if isinstance(scenario.decision_spec, SelectionSpec):
                covered = {
                    c
                    for g in scenario.decision_spec.gates
                    if g.kind == "must_cover"
                    for c in g.criteria
                }
                for criterion in set(gold.criteria) - covered:
                    err(f"coverage check criterion {criterion!r} is not in a "
                        f"must_cover gate — optimal answers need not cover it")
            for criterion, acceptable in gold.criteria.items():
                derived_ids = sorted(derived.coverage.get(criterion, frozenset()))
                if sorted(acceptable) != derived_ids:
                    err(
                        f"coverage mismatch for {criterion!r}: gold "
                        f"{sorted(acceptable)} vs oracle {derived_ids}"
                    )
        elif isinstance(gold, SequenceCheck):
            if not any(c.depends_on for c in scenario.candidates):
                err("sequence check but no candidate has dependencies")

    # --- prompt integrity ---------------------------------------------------
    prompt = build_prompt(scenario)
    if scenario.gold_rationale.strip() and scenario.gold_rationale.strip() in prompt:
        err("gold_rationale leaks into the rendered prompt")
    for trap in scenario.traps:
        if trap.description.strip() and trap.description.strip() in prompt:
            err(f"trap {trap.id} description leaks into the rendered prompt")
    for u in scenario.unknowns:
        for v in u.admissible_values:
            token = f"{u.fact}: {_fmt_value(v)}"
            if token in prompt:
                err(f"unknown fact {u.fact!r} renders a value into the prompt")

    if scenario.facts_in_prose:
        fact_sources: list[tuple[str, dict]] = [
            ("facts", scenario.facts),
            *[(c.id, c.facts) for c in scenario.candidates],
            *[(c.id, c.facts) for c in scenario.clusters],
        ]
        for holder_id, holder_facts in fact_sources:
            for name, value in holder_facts.items():
                if isinstance(value, bool):
                    continue  # booleans are expressed semantically in prose
                if _fmt_value(value) not in scenario.evidence:
                    err(
                        f"facts_in_prose: {holder_id}.{name}={_fmt_value(value)} "
                        f"does not appear in the evidence text"
                    )

    if len(scenario.gold_rationale.strip()) < MIN_GOLD_RATIONALE_CHARS:
        err(
            f"gold_rationale too short "
            f"({len(scenario.gold_rationale.strip())} chars) — it is the human "
            f"audit artifact for the decision_spec"
        )

    return errors


def validate_set(scenarios: list[Scenario]) -> list[str]:
    """Cross-scenario checks: matched-pair behavior and id uniqueness."""
    errors: list[str] = []
    ids = [s.instance_id for s in scenarios]
    for dup in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"duplicate instance_id {dup!r}")

    pairs: dict[str, list[Scenario]] = defaultdict(list)
    for s in scenarios:
        pairs[s.pair_id].append(s)

    for pair_id, members in sorted(pairs.items()):
        if len(members) < 2:
            errors.append(f"pair {pair_id!r} has only {len(members)} member(s)")
            continue
        base = next((s for s in members if s.variant == Variant.BASE), None)
        if base is None:
            errors.append(f"pair {pair_id!r} has no base variant")
            continue
        try:
            base_outcome = _pair_outcome(base)
        except (OracleError, ExprError):
            continue  # per-scenario validation already reports this
        for other in members:
            if other is base:
                continue
            try:
                other_outcome = _pair_outcome(other)
            except (OracleError, ExprError):
                continue
            comparable = (
                other.candidate_ids == base.candidate_ids
                and other.cluster_ids == base.cluster_ids
            )
            if other.variant == Variant.COUNTERFACTUAL:
                if comparable and other_outcome == base_outcome:
                    errors.append(
                        f"pair {pair_id!r}: counterfactual variant "
                        f"{other.instance_id} does not flip the decision"
                    )
            elif other.variant in (Variant.HAYSTACK, Variant.PRECOMPUTED,
                                   Variant.INVARIANCE):
                if comparable and other_outcome != base_outcome:
                    errors.append(
                        f"pair {pair_id!r}: {other.variant.value} variant "
                        f"{other.instance_id} changes the decision but must not"
                    )
    return errors


def _pair_outcome(scenario: Scenario) -> tuple:
    derived = derive_gold(scenario)
    if derived.must_clarify:
        return ("clarify", derived.required_questions)
    return derived.outcome  # type: ignore[return-value]
