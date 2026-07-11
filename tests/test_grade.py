"""Grading semantics: principal checks, conditional diagnostics, balanced
accuracy, sequence, coverage, regret plumbing."""
from __future__ import annotations

import pytest

from factories import clarify_scenario, cluster_scenario, selection_scenario, slot_scenario
from pjbench.grade import grade_scenario
from pjbench.schema import (
    Action,
    AnswerSpec,
    ClusterLabels,
    ConfidenceLabel,
    DecisionStatus,
    MaterialityLabel,
    SignalType,
)


def _optimal_answer() -> AnswerSpec:
    return AnswerSpec(
        decision_status=DecisionStatus.DECIDE,
        action=Action.BUILD,
        selected=["a", "b", "d"],
        sequence=["a", "b", "d"],
        coverage={"S1": ["a", "b"]},
    )


def _item(result, gold_id):
    return next(i for i in result.items if i.gold_id == gold_id)


class TestPrincipalSelection:
    def test_optimal_passes_with_zero_regret(self):
        result = grade_scenario(selection_scenario(), _optimal_answer())
        assert _item(result, "g1").passed
        assert result.feasible is True and result.regret == 0.0

    def test_suboptimal_fails_with_regret(self):
        answer = _optimal_answer().model_copy(
            update={"selected": ["a", "d"], "sequence": ["a", "d"],
                    "coverage": {"S1": ["a"]}}
        )
        result = grade_scenario(selection_scenario(), answer)
        assert not _item(result, "g1").passed
        assert result.feasible is True and result.regret == pytest.approx(0.4)

    def test_infeasible_fails_without_regret(self):
        answer = _optimal_answer().model_copy(
            update={"selected": ["a", "c", "d"], "sequence": ["a", "c", "d"]}
        )
        result = grade_scenario(selection_scenario(), answer)
        assert not _item(result, "g1").passed
        assert result.feasible is False and result.regret is None

    def test_wrong_action_fails(self):
        answer = _optimal_answer().model_copy(update={"action": Action.DEFER})
        result = grade_scenario(selection_scenario(), answer)
        assert not _item(result, "g1").passed


class TestConditionalDiagnostics:
    def test_wrong_branch_fails_principal_once_and_skips_downstream(self):
        answer = AnswerSpec(decision_status=DecisionStatus.CLARIFY, questions=[])
        result = grade_scenario(selection_scenario(), answer)
        assert not result.branch_correct
        principal = _item(result, "g1")
        assert principal.applicable and principal.score == 0.0
        for gold_id in ("g2", "g3"):
            item = _item(result, gold_id)
            assert not item.applicable, "downstream must be skipped, not failed"

    def test_matching_branch_grades_downstream(self):
        result = grade_scenario(selection_scenario(), _optimal_answer())
        assert all(_item(result, g).applicable for g in ("g2", "g3"))


class TestSequence:
    def test_dependency_order_respected(self):
        result = grade_scenario(selection_scenario(), _optimal_answer())
        assert _item(result, "g2").passed

    def test_dependency_order_violated(self):
        answer = _optimal_answer().model_copy(update={"sequence": ["b", "a", "d"]})
        result = grade_scenario(selection_scenario(), answer)
        item = _item(result, "g2")
        assert item.applicable and item.score == 0.0

    def test_sequence_not_permutation_scores_zero(self):
        answer = _optimal_answer().model_copy(update={"sequence": ["a", "b"]})
        result = grade_scenario(selection_scenario(), answer)
        assert _item(result, "g2").score == 0.0

    def test_no_edges_among_selected_is_not_applicable(self):
        answer = _optimal_answer().model_copy(
            update={"selected": ["a", "d"], "sequence": ["a", "d"],
                    "coverage": {"S1": ["a"]}}
        )
        result = grade_scenario(selection_scenario(), answer)
        assert not _item(result, "g2").applicable


class TestCoverage:
    def test_valid_edges_pass(self):
        result = grade_scenario(selection_scenario(), _optimal_answer())
        assert _item(result, "g3").passed

    def test_unselected_item_claim_fails(self):
        answer = _optimal_answer().model_copy(
            update={"selected": ["a", "b", "d"], "coverage": {"S1": ["c"]}}
        )
        result = grade_scenario(selection_scenario(), answer)
        assert _item(result, "g3").score == 0.0

    def test_missing_claim_fails(self):
        answer = _optimal_answer().model_copy(update={"coverage": {}})
        result = grade_scenario(selection_scenario(), answer)
        assert _item(result, "g3").score == 0.0


class TestClarifyGrading:
    def test_exact_required_set_passes(self):
        answer = AnswerSpec(
            decision_status=DecisionStatus.CLARIFY, questions=["q-volume"]
        )
        result = grade_scenario(clarify_scenario(), answer)
        assert _item(result, "g1").passed

    def test_deciding_when_clarify_needed_fails(self):
        answer = AnswerSpec(decision_status=DecisionStatus.DECIDE, action=Action.BUILD)
        result = grade_scenario(clarify_scenario(), answer)
        assert not _item(result, "g1").passed

    def test_clarifying_when_determined_fails(self):
        scenario = clarify_scenario(admissible=(1500, 5000))
        answer = AnswerSpec(
            decision_status=DecisionStatus.CLARIFY, questions=["q-volume"]
        )
        result = grade_scenario(scenario, answer)
        assert not _item(result, "g1").passed


class TestSlotGrading:
    def test_correct_slot_passes(self):
        answer = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.EXPERIMENT,
            slots={"ab_decision": "ship"},
        )
        assert _item(grade_scenario(slot_scenario(), answer), "g1").passed

    def test_wrong_slot_fails(self):
        answer = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.EXPERIMENT,
            slots={"ab_decision": "hold"},
        )
        assert not _item(grade_scenario(slot_scenario(), answer), "g1").passed


class TestClusterLabelGrading:
    def _labels(self, loud: SignalType, quiet: SignalType) -> dict:
        return {
            "cl-loud": ClusterLabels(
                signal_type=loud, confidence=ConfidenceLabel.INSUFFICIENT,
                materiality=MaterialityLabel.LOW),
            "cl-quiet": ClusterLabels(
                signal_type=quiet, confidence=ConfidenceLabel.SUFFICIENT,
                materiality=MaterialityLabel.REVENUE),
        }

    def test_perfect_labels_pass(self):
        answer = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.BUILD,
            cluster_labels=self._labels(
                SignalType.DUPLICATE_NOISE, SignalType.CONCENTRATED_ACCOUNT_RISK),
        )
        result = grade_scenario(cluster_scenario(), answer)
        assert _item(result, "g2").score == 1.0

    def test_all_noise_cannot_exploit_imbalance(self):
        # both clusters labelled noise: signal_type balanced accuracy is 0.5
        # (recall 1.0 on the noise class, 0.0 on the risk class), not 0.5+ via count
        answer = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.BUILD,
            cluster_labels=self._labels(
                SignalType.DUPLICATE_NOISE, SignalType.DUPLICATE_NOISE),
        )
        result = grade_scenario(cluster_scenario(), answer)
        item = _item(result, "g2")
        # signal_type 0.5, confidence 1.0, materiality 1.0 -> mean 0.833
        assert item.score == pytest.approx((0.5 + 1.0 + 1.0) / 3)

    def test_missing_cluster_counts_as_wrong(self):
        answer = AnswerSpec(
            decision_status=DecisionStatus.DECIDE, action=Action.BUILD,
            cluster_labels={},
        )
        result = grade_scenario(cluster_scenario(), answer)
        assert _item(result, "g2").score == 0.0


class TestFormatFailure:
    def test_none_spec_fails_every_item(self):
        result = grade_scenario(selection_scenario(), None)
        assert not result.format_compliant
        assert all(i.applicable and i.score == 0.0 for i in result.items)
