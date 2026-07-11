"""Canonical prompt template.

One fixed template for every scenario and model — part of the pinned protocol.
The decision policy section is rendered *mechanically from the decision_spec*
(the same expressions the oracle executes), so the policy the model reads and
the policy the grader applies are one artifact and cannot diverge. Gold,
traps, unknowns, and rationale never render.
"""
from __future__ import annotations

from .schema import (
    ActionChoiceSpec,
    Rule,
    Scenario,
    SelectionSpec,
    SlotDecisionSpec,
)

RESPONSE_INSTRUCTIONS = """\
## Your response

Respond with ONLY one JSON object (no prose before or after) with this shape:

{
  "decision_status": "decide" | "clarify",

  // decision_status == "clarify": the question ids (from the menu) covering the
  // facts you need before deciding responsibly. Ask about a fact only if its
  // value could change the correct decision AND it is not stated anywhere in
  // this prompt. Asking about stated facts, or clarifying when the decision is
  // already determined, counts against you.
  "questions": ["<question id>", ...],

  // decision_status == "decide":
  "action": "build" | "configure" | "partner" | "experiment" | "defer" | "decline",
  "selected": ["<candidate id>", ...],
  "sequence": ["<selected ids in execution order, respecting dependencies>", ...],
  "slots": {"<slot name>": "<value>", ...},
  "cluster_labels": {"<cluster id>": {"signal_type": "...", "confidence": "...",
                                       "materiality": "..."}, ...},
  "coverage": {"<success criterion id>": ["<selected item id>", ...], ...},

  // optional, never graded
  "notes": "<free text>"
}

Rules:
- Pick exactly one decision_status. Apply the stated decision policy exactly —
  it is the grading policy, verbatim. Your reasoning is not graded; your
  decision is.
- Fill only the fields the scenario calls for (a selection task needs
  "selected"; a slot task needs "slots"; label tasks need "cluster_labels").
- Ids are case-sensitive. Do not invent ids.
"""


def _fmt_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _rules_lines(rules: list[Rule], what: str) -> list[str]:
    lines = [f"Apply the FIRST matching rule ({what}):"]
    for i, rule in enumerate(rules, 1):
        outcome = (
            " or ".join(a.value for a in rule.actions) if rule.actions else rule.label
        )
        cond = "otherwise" if rule.when.strip() == "True" else f"if `{rule.when}`"
        lines.append(f"{i}. {cond} -> {outcome}")
    return lines


def _policy_section(scenario: Scenario) -> str:
    spec = scenario.decision_spec
    lines: list[str] = ["## Decision policy (this is the grading policy, verbatim)", ""]
    if isinstance(spec, SelectionSpec):
        lines.append(
            f"Select the set of candidate items to execute; report "
            f'`action: "{spec.action.value}"` and the selected ids.'
        )
        obj = spec.objective
        unit = f" ({obj.unit})" if obj.unit else ""
        lines.append(
            f"- Objective: {obj.sense} the total `{obj.value_fact}`{unit} "
            f"summed over the selected items."
        )
        if spec.capacity is not None:
            cunit = f" {spec.capacity.unit}" if spec.capacity.unit else ""
            lines.append(
                f"- Capacity: total `{spec.capacity.cost_fact}` of selected items "
                f"must not exceed {_fmt_value(spec.capacity.budget)}{cunit}."
            )
        for gate in spec.gates:
            basis = f" [{gate.basis}]" if gate.basis else ""
            if gate.kind == "mandatory_when":
                lines.append(
                    f"- Hard rule: every item where `{gate.when}` MUST be selected.{basis}"
                )
            elif gate.kind == "excluded_when":
                lines.append(
                    f"- Hard rule: no item where `{gate.when}` may be selected.{basis}"
                )
            elif gate.kind == "requires_dependencies":
                lines.append(
                    f"- Hard rule: an item may be selected only if every item it "
                    f"depends on is also selected.{basis}"
                )
            elif gate.kind == "must_cover":
                lines.append(
                    f"- Hard rule: the selected set must address every success "
                    f"criterion in {gate.criteria}.{basis}"
                )
        lines.append(
            "- Ties: any selection achieving the optimal objective value is correct."
        )
    elif isinstance(spec, ActionChoiceSpec):
        lines.append(
            "Choose exactly one action for this request and report it as `action`."
        )
        lines += _rules_lines(spec.rules, "top to bottom")
    elif isinstance(spec, SlotDecisionSpec):
        lines.append(
            f'Report `action: "{spec.action.value}"` and set '
            f'`slots.{spec.slot}` to one of {spec.labels}.'
        )
        lines += _rules_lines(spec.rules, f"value of {spec.slot}")

    if scenario.label_rules:
        lines += ["", "For every feedback cluster, assign labels by these rules:"]
        for dim, rules in scenario.label_rules.items():
            lines.append(f"- `{dim}`:")
            for i, rule in enumerate(rules, 1):
                cond = (
                    "otherwise"
                    if rule.when.strip() == "True"
                    else f"if `{rule.when}`"
                )
                lines.append(f"  {i}. {cond} -> {rule.label}")
    return "\n".join(lines)


def _facts_table(facts: dict, indent: str = "") -> list[str]:
    return [f"{indent}- {k}: {_fmt_value(v)}" for k, v in facts.items()]


def _scenario_section(scenario: Scenario) -> str:
    lines: list[str] = [f"## Situation: {scenario.title}", "",
                        scenario.company_context.strip()]

    if scenario.constraints:
        lines += ["", "## Constraints"]
        for c in scenario.constraints:
            lines.append(f"- {c.id}: {c.text}")

    if scenario.success_criteria:
        lines += ["", "## Success criteria"]
        for sc in scenario.success_criteria:
            lines.append(f"- {sc.id}: {sc.text}")

    if scenario.fact_definitions:
        lines += ["", "## Fact glossary (fact names used by the decision policy)"]
        for name, text in scenario.fact_definitions.items():
            lines.append(f"- `{name}`: {text}")

    if scenario.facts and not scenario.facts_in_prose:
        lines += ["", "## Stated facts"]
        lines += _facts_table(scenario.facts)

    if scenario.candidates:
        lines += ["", "## Candidate items"]
        for cand in scenario.candidates:
            lines.append(f"- {cand.id}: {cand.title}")
            if cand.addresses:
                lines.append(f"  - addresses: {cand.addresses}")
            if cand.depends_on:
                lines.append(f"  - depends on: {cand.depends_on}")
            if cand.facts and not scenario.facts_in_prose:
                lines += _facts_table(cand.facts, indent="  ")

    if scenario.clusters:
        lines += ["", "## Feedback clusters"]
        for cluster in scenario.clusters:
            lines.append(f"- {cluster.id}: {cluster.title}")
            if cluster.facts and not scenario.facts_in_prose:
                lines += _facts_table(cluster.facts, indent="  ")

    if scenario.evidence.strip():
        lines += ["", "## Evidence documents", "", scenario.evidence.strip()]

    if scenario.questions:
        lines += ["", "## Clarification menu (only relevant if you must clarify)"]
        for q in scenario.questions:
            lines.append(f"- {q.id}: {q.text}")

    return "\n".join(lines)


def build_prompt(scenario: Scenario) -> str:
    header = (
        "You are the product decision-maker for the situation below. Read the "
        "context, evidence, and stated decision policy, then respond with a "
        "single structured JSON decision."
    )
    return "\n\n".join(
        [
            header,
            _scenario_section(scenario),
            _policy_section(scenario),
            RESPONSE_INSTRUCTIONS,
        ]
    )
