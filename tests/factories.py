"""Hand-computable scenario factories for unit tests.

The selection factory's arithmetic (documented inline) is the golden reference
for the oracle: budget 9, maximize value, item d contractually mandatory,
b depends on a, and the selection must cover criterion S1 (addressed only by
a and b). Optimum: {a,b,d}=1000 (cost 9). {c,d} fails S1 coverage, {a,c,d}
busts the budget at cost 10; the best feasible alternative is {a,d}=600.
"""
from __future__ import annotations

from pjbench.schema import (
    Action,
    Candidate,
    Capacity,
    Cluster,
    ClusterLabelsCheck,
    Constraint,
    CoverageCheck,
    Difficulty,
    Dimension,
    Gate,
    Objective,
    PrincipalAction,
    PrincipalClarify,
    PrincipalSelection,
    PrincipalSlot,
    Question,
    Rule,
    Scenario,
    SelectionSpec,
    SequenceCheck,
    SlotDecisionSpec,
    Split,
    SuccessCriterion,
    Unknown,
    Variant,
)

RATIONALE = (
    "Worked math: budget 9 engineer-weeks, maximize summed value. d is mandatory "
    "(contractual=true, C1) at cost 2, leaving 7. b requires a (dependency rule), "
    "and the selection must cover S1, addressed only by a or b. {a,b,d} costs "
    "3+4+2=9 and scores 1000; {c,d} fails S1 coverage; {a,c,d} costs 10, over "
    "budget; best feasible alternative {a,d}=600. Unique optimum: {a,b,d}=1000."
)


def selection_scenario(**overrides) -> Scenario:
    base = dict(
        instance_id="sel-001",
        family="capacity-prioritization",
        pair_id="sel-pair-001",
        variant=Variant.BASE,
        domain="b2b-saas",
        difficulty=Difficulty.MEDIUM,
        version="0.1.0",
        split=Split.PUBLIC,
        title="Quarter planning",
        company_context="Plan the quarter under a fixed engineering budget.",
        constraints=[
            Constraint(id="C1", text="Contractually committed items must ship this quarter."),
        ],
        success_criteria=[
            SuccessCriterion(id="S1", text="Retain the enterprise segment."),
        ],
        candidates=[
            Candidate(id="a", title="API v2", facts={"cost": 3, "value": 500, "contractual": False},
                      addresses=["S1"]),
            Candidate(id="b", title="SSO", facts={"cost": 4, "value": 400, "contractual": False},
                      addresses=["S1"], depends_on=["a"]),
            Candidate(id="c", title="Analytics", facts={"cost": 5, "value": 700, "contractual": False}),
            Candidate(id="d", title="Data residency", facts={"cost": 2, "value": 100, "contractual": True}),
        ],
        decision_spec=SelectionSpec(
            kind="selection",
            action=Action.BUILD,
            capacity=Capacity(cost_fact="cost", budget=9, unit="engineer-weeks"),
            objective=Objective(sense="maximize", value_fact="value"),
            gates=[
                Gate(kind="mandatory_when", when="contractual", basis="C1"),
                Gate(kind="requires_dependencies"),
                Gate(kind="must_cover", criteria=["S1"]),
            ],
        ),
        gold=[
            PrincipalSelection(
                id="g1", dimension=Dimension.DECISION, check="principal_selection",
                action=Action.BUILD, optimal_sets=[["a", "b", "d"]],
            ),
            SequenceCheck(id="g2", dimension=Dimension.SEQUENCING, check="sequence"),
            CoverageCheck(
                id="g3", dimension=Dimension.COVERAGE, check="coverage",
                criteria={"S1": ["a", "b"]},
            ),
        ],
        gold_rationale=RATIONALE,
    )
    base.update(overrides)
    return Scenario.model_validate(base)


def clarify_scenario(admissible=(500, 5000), **overrides) -> Scenario:
    """Build iff volume >= 1000. Admissible (500, 5000) straddles the threshold
    -> clarify with q-volume required. Admissible (1500, 5000) -> decide/build."""
    must_clarify = min(admissible) < 1000 <= max(admissible)
    principal = (
        PrincipalClarify(
            id="g1", dimension=Dimension.CLARIFICATION, check="principal_clarify",
            required_questions=["q-volume"], allowed_questions=["q-volume"],
        )
        if must_clarify
        else PrincipalAction(
            id="g1", dimension=Dimension.DECISION, check="principal_action",
            acceptable_actions=[Action.BUILD],
        )
    )
    base = dict(
        instance_id="clar-001",
        family="clarify-vs-decide",
        pair_id="clar-pair-001",
        variant=Variant.BASE,
        domain="b2b-saas",
        difficulty=Difficulty.MEDIUM,
        version="0.1.0",
        split=Split.PUBLIC,
        title="Export feature request",
        company_context="A customer asked for bulk export.",
        fact_definitions={"volume": "expected export rows per day"},
        questions=[
            Question(id="q-volume", text="How many rows per day?", fact="volume"),
        ],
        unknowns=[Unknown(fact="volume", admissible_values=list(admissible))],
        decision_spec={
            "kind": "action_choice",
            "rules": [
                Rule(when="volume >= 1000", actions=[Action.BUILD]),
                Rule(when="True", actions=[Action.DECLINE]),
            ],
        },
        gold=[principal],
        gold_rationale=(
            "Worked math: policy builds iff volume >= 1000 rows/day. The admissible "
            "range straddles (or clears) that threshold, so the optimal action flips "
            "(or does not flip) with the unknown; the oracle enumerates both values "
            "and derives clarify/decide plus the required question set from that."
        ),
    )
    base.update(overrides)
    return Scenario.model_validate(base)


def slot_scenario(**overrides) -> Scenario:
    """A/B readout: extend if underpowered, ship if delta >= 1.0, else hold.
    Facts: samples 6000/6000, delta 1.4 -> ship."""
    base = dict(
        instance_id="slot-001",
        family="ab-readout",
        pair_id="slot-pair-001",
        variant=Variant.BASE,
        domain="b2c",
        difficulty=Difficulty.EASY,
        version="0.1.0",
        split=Split.PUBLIC,
        title="Checkout experiment readout",
        company_context="Decide the checkout A/B test.",
        facts={"sample_a": 6000, "sample_b": 6000, "delta_pct": 1.4},
        decision_spec=SlotDecisionSpec(
            kind="slot_decision",
            action=Action.EXPERIMENT,
            slot="ab_decision",
            labels=["ship", "hold", "extend"],
            rules=[
                Rule(when="sample_a < 5000 or sample_b < 5000", label="extend"),
                Rule(when="delta_pct >= 1.0", label="ship"),
                Rule(when="True", label="hold"),
            ],
        ),
        gold=[
            PrincipalSlot(
                id="g1", dimension=Dimension.DECISION, check="principal_slot",
                action=Action.EXPERIMENT, slot="ab_decision", acceptable=["ship"],
            ),
        ],
        gold_rationale=(
            "Worked math: pre-registered rule is extend if either arm has under "
            "5000 samples; both arms have 6000, so the test is powered. Ship iff "
            "delta_pct >= 1.0; measured delta is 1.4, so the entailed readout is "
            "ship. Hold is unreachable given these facts."
        ),
    )
    base.update(overrides)
    return Scenario.model_validate(base)


def cluster_scenario(**overrides) -> Scenario:
    """Two clusters with rule-derived labels; principal is an action_choice."""
    label_rules = {
        "signal_type": [
            Rule(when="distinct_accounts <= 1", label="duplicate-noise"),
            Rule(when="distinct_accounts <= 3 and arr_pct >= 20", label="concentrated-account-risk"),
            Rule(when="distinct_accounts >= 5 and exposure_pct >= 10", label="broad-trend"),
            Rule(when="True", label="isolated-request"),
        ],
        "confidence": [
            Rule(when="independent_sources >= 2", label="sufficient"),
            Rule(when="True", label="insufficient"),
        ],
        "materiality": [
            Rule(when="arr_pct >= 20", label="revenue"),
            Rule(when="True", label="low"),
        ],
    }
    base = dict(
        instance_id="clus-001",
        family="signal-analysis",
        pair_id="clus-pair-001",
        variant=Variant.BASE,
        domain="b2b-saas",
        difficulty=Difficulty.MEDIUM,
        version="0.1.0",
        split=Split.PUBLIC,
        title="Feedback triage",
        company_context="Label the feedback clusters, then decide.",
        clusters=[
            Cluster(id="cl-loud", title="Dashboard slow",
                    facts={"distinct_accounts": 1, "arr_pct": 2,
                           "exposure_pct": 1, "independent_sources": 1}),
            Cluster(id="cl-quiet", title="SSO gaps",
                    facts={"distinct_accounts": 3, "arr_pct": 40,
                           "exposure_pct": 8, "independent_sources": 3}),
        ],
        facts={"top_cluster_is_material": True},
        decision_spec={
            "kind": "action_choice",
            "rules": [
                Rule(when="top_cluster_is_material", actions=[Action.BUILD]),
                Rule(when="True", actions=[Action.DEFER]),
            ],
        },
        label_rules=label_rules,
        gold=[
            PrincipalAction(
                id="g1", dimension=Dimension.DECISION, check="principal_action",
                acceptable_actions=[Action.BUILD],
            ),
            ClusterLabelsCheck(
                id="g2", dimension=Dimension.SIGNAL_ANALYSIS, check="cluster_labels",
                expected={
                    "cl-loud": {"signal_type": "duplicate-noise",
                                "confidence": "insufficient", "materiality": "low"},
                    "cl-quiet": {"signal_type": "concentrated-account-risk",
                                 "confidence": "sufficient", "materiality": "revenue"},
                },
            ),
        ],
        gold_rationale=(
            "Worked labels: cl-loud has one distinct account, so the first "
            "signal_type rule fires (duplicate-noise); one source -> insufficient; "
            "2% ARR -> low materiality. cl-quiet has 3 accounts and 40% ARR: rule 2 "
            "fires (concentrated-account-risk); 3 sources -> sufficient; revenue."
        ),
    )
    base.update(overrides)
    return Scenario.model_validate(base)
