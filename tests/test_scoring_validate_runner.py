"""Scoring aggregation, validator (oracle equality + lints), runner cache and
prediction-integrity behavior, and baseline strategies."""
from __future__ import annotations

import pytest

from factories import clarify_scenario, cluster_scenario, selection_scenario, slot_scenario
from pjbench.baselines import BASELINES, oracle_answer
from pjbench.grade import grade_scenario
from pjbench.runner import (
    PredictionIntegrityError,
    PredictionRecord,
    load_cached,
    match_predictions,
    store_cached,
)
from pjbench.schema import (
    Action,
    AnswerSpec,
    DecisionStatus,
    PrincipalSelection,
    Dimension,
    Variant,
)
from pjbench.scoring import score
from pjbench.validate import validate_scenario, validate_set

ALL_SCENARIOS = [
    selection_scenario(), clarify_scenario(), slot_scenario(), cluster_scenario(),
]


class TestScoring:
    def test_oracle_answers_score_100(self):
        results = [grade_scenario(s, oracle_answer(s)) for s in ALL_SCENARIOS]
        run = score(results, model="baseline/oracle", benchmark_version="test")
        assert run.overall == pytest.approx(1.0)
        assert run.decision == pytest.approx(1.0)
        assert run.branch_accuracy == 1.0
        for dim, ds in run.dimensions.items():
            assert ds.fraction == pytest.approx(1.0), dim

    def test_family_weighting_equalizes(self):
        # two scenarios in family A (one pass, one fail) + one pass in family B:
        # per-family decision = A 0.5, B 1.0 -> 0.75, not item-mean 2/3
        s_pass = selection_scenario()
        s_fail = selection_scenario(instance_id="sel-002")
        s_other = slot_scenario()  # family ab-readout
        good = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.BUILD,
            selected=["a", "b", "d"], sequence=["a", "b", "d"],
            coverage={"S1": ["a", "b"]},
        )
        bad = good.model_copy(update={"selected": ["a", "d"], "sequence": ["a", "d"],
                                      "coverage": {"S1": ["a"]}})
        slot_good = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.EXPERIMENT,
            slots={"ab_decision": "ship"},
        )
        results = [
            grade_scenario(s_pass, good),
            grade_scenario(s_fail, bad),
            grade_scenario(s_other, slot_good),
        ]
        run = score(results, model="m", benchmark_version="test")
        assert run.decision == pytest.approx(0.75)

    def test_regret_and_feasibility_aggregate(self):
        s = selection_scenario()
        bad = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.BUILD,
            selected=["a", "d"], sequence=["a", "d"], coverage={"S1": ["a"]},
        )
        run = score([grade_scenario(s, bad)], model="m", benchmark_version="test")
        assert run.feasibility_rate == 1.0
        assert run.mean_regret == pytest.approx(0.4)


class TestValidate:
    def test_factories_validate_clean(self):
        for scenario in ALL_SCENARIOS:
            assert validate_scenario(scenario) == [], scenario.instance_id

    def test_gold_oracle_disagreement_detected(self):
        scenario = selection_scenario()
        broken_gold = [
            g if not isinstance(g, PrincipalSelection) else g.model_copy(
                update={"optimal_sets": [["c", "d"]]})
            for g in scenario.gold
        ]
        errors = validate_scenario(selection_scenario(gold=broken_gold))
        assert any("optimal_sets mismatch" in e for e in errors)

    def test_clarify_mislabeled_as_decide_detected(self):
        from pjbench.schema import PrincipalAction
        wrong = clarify_scenario(
            gold=[PrincipalAction(
                id="g1", dimension=Dimension.DECISION, check="principal_action",
                acceptable_actions=[Action.BUILD])],
        )
        errors = validate_scenario(wrong)
        assert any("decision-changing" in e for e in errors)

    def test_missing_catchall_detected(self):
        from pjbench.schema import Rule
        scenario = cluster_scenario()
        rules = dict(scenario.label_rules)
        rules["confidence"] = [Rule(when="independent_sources >= 2", label="sufficient")]
        errors = validate_scenario(cluster_scenario(label_rules=rules))
        assert any("catch-all" in e for e in errors)

    def test_facts_in_prose_requires_literal_presence(self):
        scenario = cluster_scenario(
            facts_in_prose=True, evidence="The SSO cluster spans 3 accounts."
        )
        errors = validate_scenario(scenario)
        assert any("does not appear in the evidence" in e for e in errors)

    def test_counterfactual_pair_must_flip(self):
        base = selection_scenario()
        same = selection_scenario(
            instance_id="sel-001b", variant=Variant.COUNTERFACTUAL
        )
        errors = validate_set([base, same])
        assert any("does not flip" in e for e in errors)

    def test_invariance_pair_must_not_flip(self):
        base = clarify_scenario()
        flipped = clarify_scenario(
            instance_id="clar-001b", variant=Variant.INVARIANCE,
            admissible=(1500, 5000),
        )
        errors = validate_set([base, flipped])
        assert any("must not" in e for e in errors)


class TestRunnerIntegrity:
    def _record(self, scenario, **overrides) -> PredictionRecord:
        base = dict(
            instance_id=scenario.instance_id,
            model="m/x",
            reasoning="high",
            scenario_hash=scenario.content_hash(),
            raw_output="{}",
            spec=None,
            format_compliant=False,
            repair_attempted=False,
        )
        base.update(overrides)
        return PredictionRecord(**base)

    def test_cache_roundtrip_and_hash_invalidation(self, tmp_path):
        scenario = selection_scenario()
        record = self._record(scenario)
        store_cached(tmp_path, record)
        assert load_cached(tmp_path, "m/x", "high", scenario) is not None
        # edit the scenario -> same instance_id, different hash -> miss
        edited = selection_scenario(title="Quarter planning v2")
        assert load_cached(tmp_path, "m/x", "high", edited) is None

    def test_provider_failures_never_cached(self, tmp_path):
        scenario = selection_scenario()
        record = self._record(scenario, error="provider error: timeout")
        store_cached(tmp_path, record)
        assert load_cached(tmp_path, "m/x", "high", scenario) is None

    def test_match_predictions_rejects_missing_and_duplicates(self):
        s1, s2 = selection_scenario(), slot_scenario()
        r1 = self._record(s1)
        with pytest.raises(PredictionIntegrityError, match="missing prediction"):
            match_predictions([s1, s2], [r1])
        with pytest.raises(PredictionIntegrityError, match="duplicate prediction"):
            match_predictions([s1], [r1, r1])

    def test_match_predictions_rejects_stale_hash(self):
        scenario = selection_scenario()
        stale = self._record(scenario, scenario_hash="deadbeef0000")
        with pytest.raises(PredictionIntegrityError, match="stale prediction"):
            match_predictions([scenario], [stale])

    def test_match_predictions_accepts_exact_set(self):
        scenario = selection_scenario()
        record = self._record(scenario)
        assert match_predictions([scenario], [record]) == {scenario.instance_id: record}


class TestBaselines:
    def test_every_strategy_produces_valid_answers(self):
        for name, fn in BASELINES.items():
            for scenario in ALL_SCENARIOS:
                answer = fn(scenario)
                result = grade_scenario(scenario, answer)
                assert result.format_compliant, f"{name} on {scenario.instance_id}"

    def test_trivial_baselines_do_not_saturate(self):
        for name, fn in BASELINES.items():
            if name == "oracle":
                continue
            results = [grade_scenario(s, fn(s)) for s in ALL_SCENARIOS]
            run = score(results, model=f"baseline/{name}", benchmark_version="test")
            assert run.decision < 1.0, f"{name} passes every principal decision"


class TestDeterminism:
    def test_grading_is_reproducible(self):
        answers = [oracle_answer(s) for s in ALL_SCENARIOS]
        first = score(
            [grade_scenario(s, a) for s, a in zip(ALL_SCENARIOS, answers)],
            model="m", benchmark_version="test",
        ).model_dump_json()
        second = score(
            [grade_scenario(s, a) for s, a in zip(ALL_SCENARIOS, answers)],
            model="m", benchmark_version="test",
        ).model_dump_json()
        assert first == second
