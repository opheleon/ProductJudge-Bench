"""Oracle golden tests against hand-computed optima (see factories.py)."""
from __future__ import annotations

import pytest

from factories import clarify_scenario, cluster_scenario, selection_scenario, slot_scenario
from pjbench.exprs import ExprError, evaluate, expr_fact_names
from pjbench.oracle import (
    OracleError,
    derive_clarify,
    derive_gold,
    principal_outcome,
    selection_regret,
    solve_selection_for,
)


class TestExprs:
    def test_arithmetic_and_comparison(self):
        assert evaluate("a + b * 2 >= 10", {"a": 4, "b": 3}) is True
        assert evaluate("a / b < 1", {"a": 1, "b": 2}) is True

    def test_bool_ops_and_membership(self):
        facts = {"tier": "enterprise", "n": 5}
        assert evaluate("tier in ['enterprise', 'mid'] and n >= 5", facts) is True
        assert evaluate("not (n > 10)", facts) is True

    def test_unknown_fact_raises(self):
        with pytest.raises(ExprError, match="unknown fact"):
            evaluate("missing > 1", {"present": 1})

    def test_disallowed_syntax_rejected(self):
        for expr in ["__import__('os')", "a.b", "f(1)", "x[0]", "(lambda: 1)()"]:
            with pytest.raises(ExprError):
                evaluate(expr, {"a": 1, "x": [1], "f": min})

    def test_fact_names(self):
        assert expr_fact_names("a + b >= c and d in [1]") == {"a", "b", "c", "d"}


class TestSelection:
    def test_golden_optimum(self):
        solution = solve_selection_for(selection_scenario())
        assert solution.optimal_sets == frozenset({frozenset({"a", "b", "d"})})
        assert solution.optimal_objective == 1000

    def test_mandatory_gate_binds(self):
        # every feasible set contains d (contractual=true)
        solution = solve_selection_for(selection_scenario())
        assert all("d" in s for s in solution.feasible_objectives)

    def test_dependency_gate_binds(self):
        solution = solve_selection_for(selection_scenario())
        assert all("a" in s for s in solution.feasible_objectives if "b" in s)

    def test_regret(self):
        solution = solve_selection_for(selection_scenario())
        feasible, regret = selection_regret(solution, frozenset({"a", "d"}), "maximize")
        assert feasible is True
        assert regret == pytest.approx(0.4)  # (1000-600)/1000

    def test_infeasible_answer_has_no_regret(self):
        solution = solve_selection_for(selection_scenario())
        # {c,d} fails the must_cover S1 gate; {a,c,d} busts the budget (10 > 9)
        for bad in [frozenset({"c", "d"}), frozenset({"a", "c", "d"})]:
            feasible, regret = selection_regret(solution, bad, "maximize")
            assert feasible is False and regret is None

    def test_optimum_regret_is_zero(self):
        solution = solve_selection_for(selection_scenario())
        feasible, regret = selection_regret(solution, frozenset({"a", "b", "d"}), "maximize")
        assert feasible is True and regret == 0.0

    def test_conflicting_gates_rejected(self):
        from pjbench.schema import Gate
        scenario = selection_scenario()
        spec = scenario.decision_spec.model_copy(
            update={"gates": [
                Gate(kind="mandatory_when", when="contractual", basis="C1"),
                Gate(kind="excluded_when", when="contractual", basis="C1"),
            ]}
        )
        broken = selection_scenario(decision_spec=spec)
        with pytest.raises(OracleError, match="mandatory and excluded"):
            solve_selection_for(broken)


class TestClarify:
    def test_straddling_threshold_requires_clarify(self):
        derivation = derive_clarify(clarify_scenario(admissible=(500, 5000)))
        assert derivation.must_clarify is True
        assert derivation.required_facts == frozenset({"volume"})

    def test_inert_unknown_decides(self):
        derivation = derive_clarify(clarify_scenario(admissible=(1500, 5000)))
        assert derivation.must_clarify is False
        assert derivation.outcome == ("action", frozenset({"build"}))

    def test_derive_gold_maps_questions(self):
        derived = derive_gold(clarify_scenario(admissible=(500, 5000)))
        assert derived.must_clarify is True
        assert derived.required_questions == frozenset({"q-volume"})
        assert derived.allowed_questions == frozenset({"q-volume"})


class TestSlotAndLabels:
    def test_slot_golden(self):
        outcome = principal_outcome(slot_scenario(), {})
        assert outcome == ("slot", "experiment", "ab_decision", "ship")

    def test_slot_underpowered_extends(self):
        scenario = slot_scenario(
            facts={"sample_a": 3000, "sample_b": 6000, "delta_pct": 1.4}
        )
        assert principal_outcome(scenario, {})[-1] == "extend"

    def test_cluster_labels_derived(self):
        derived = derive_gold(cluster_scenario())
        assert derived.cluster_labels == {
            "cl-loud": {"signal_type": "duplicate-noise", "confidence": "insufficient",
                        "materiality": "low"},
            "cl-quiet": {"signal_type": "concentrated-account-risk",
                         "confidence": "sufficient", "materiality": "revenue"},
        }

    def test_missing_catchall_raises(self):
        from pjbench.schema import Rule, Action
        scenario = clarify_scenario(
            admissible=(1500, 5000),
            decision_spec={
                "kind": "action_choice",
                "rules": [Rule(when="volume >= 999999999", actions=[Action.BUILD])],
            },
        )
        with pytest.raises(OracleError, match="catch-all"):
            derive_clarify(scenario)
