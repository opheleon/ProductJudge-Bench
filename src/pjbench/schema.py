"""Pydantic models for ProductJudge-Bench scenarios, decision specs, and answers.

Construct: policy-conditioned product decisions. Every scenario carries a
machine-executable `decision_spec` (the source of truth — `oracle.py` derives
gold from it) and renders its policy into the prompt from the same expressions,
so the stated policy and the graded policy cannot diverge.

Everything gradable is a set/enum/arithmetic check; if a criterion cannot be
expressed that way it is not a valid benchmark item.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DecisionStatus(str, Enum):
    DECIDE = "decide"
    CLARIFY = "clarify"


class Action(str, Enum):
    BUILD = "build"
    CONFIGURE = "configure"
    PARTNER = "partner"
    EXPERIMENT = "experiment"
    DEFER = "defer"
    DECLINE = "decline"


class SignalType(str, Enum):
    BROAD_TREND = "broad-trend"
    CONCENTRATED_ACCOUNT_RISK = "concentrated-account-risk"
    ISOLATED_REQUEST = "isolated-request"
    DUPLICATE_NOISE = "duplicate-noise"


class ConfidenceLabel(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class MaterialityLabel(str, Enum):
    REVENUE = "revenue"
    STRATEGIC = "strategic"
    CONTRACTUAL = "contractual"
    COMPLIANCE = "compliance"
    LOW = "low"


LABEL_DIMENSIONS: dict[str, type[Enum]] = {
    "signal_type": SignalType,
    "confidence": ConfidenceLabel,
    "materiality": MaterialityLabel,
}


class Dimension(str, Enum):
    DECISION = "decision"                # the principal decision of a decide-scenario
    CLARIFICATION = "clarification"      # the principal decision of a clarify-scenario
    SIGNAL_ANALYSIS = "signal_analysis"  # cluster labels (balanced accuracy)
    SEQUENCING = "sequencing"            # dependency-respecting order
    COVERAGE = "coverage"                # criterion->item edges valid + jointly satisfied


class SpecKind(str, Enum):
    SELECTION = "selection"
    ACTION_CHOICE = "action_choice"
    SLOT_DECISION = "slot_decision"


class Variant(str, Enum):
    BASE = "base"
    COUNTERFACTUAL = "counterfactual"
    INVARIANCE = "invariance"
    HAYSTACK = "haystack"
    PRECOMPUTED = "precomputed"
    CONTROL = "control"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Split(str, Enum):
    PUBLIC = "public"
    HELDOUT = "heldout"


# ---------------------------------------------------------------------------
# Scenario building blocks
# ---------------------------------------------------------------------------

FactValue = Union[int, float, str, bool]


class Candidate(BaseModel):
    """A backlog item / option the model may select."""
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    facts: dict[str, FactValue] = {}
    addresses: list[str] = []    # success-criterion ids this item addresses
    depends_on: list[str] = []   # candidate ids that must ship before this one


class Cluster(BaseModel):
    """A pre-clustered feedback theme the model must label."""
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    facts: dict[str, FactValue] = {}


class SuccessCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class Question(BaseModel):
    """An entry in the clarify menu. `fact` is the global fact it would reveal."""
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str
    fact: str


class Unknown(BaseModel):
    """A global fact whose value is not stated. The oracle solves the scenario
    once per admissible value; if the optimal outcome differs, the fact is
    decision-changing and its question becomes required."""
    model_config = ConfigDict(extra="forbid")
    fact: str
    admissible_values: list[FactValue] = Field(min_length=2)


class Trap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    why_attractive: str


# ---------------------------------------------------------------------------
# Decision spec — the machine-executable policy (source of truth for gold)
# ---------------------------------------------------------------------------


class Rule(BaseModel):
    """Ordered first-match rule. `when` is an exprs.py expression over facts;
    it is rendered verbatim into the prompt as the stated policy."""
    model_config = ConfigDict(extra="forbid")
    when: str
    # exactly one of the following is populated depending on context
    actions: list[Action] = []   # action_choice rules (ties: all acceptable)
    label: str = ""              # classification / slot rules


class Gate(BaseModel):
    """Hard feasibility rule on selection, rendered verbatim into the prompt.

    Gates are predicates over item facts, never item-id literals — an item-id
    gate would hand the model the answer, while a predicate keeps the policy
    public and the *application* (spotting which item the buried fact makes
    mandatory) the model's job. `basis` names the constraint id that justifies
    the gate; lint verifies the basis is rendered.
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["mandatory_when", "excluded_when", "requires_dependencies", "must_cover"]
    when: str = ""            # mandatory_when/excluded_when: expr over item facts
    criteria: list[str] = []  # must_cover: success-criterion ids
    basis: str = ""           # constraint id justifying the gate


class Capacity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cost_fact: str      # per-candidate fact holding the item's cost
    budget: float
    unit: str = ""      # rendered ("engineer-weeks")


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sense: Literal["maximize", "minimize"]
    value_fact: str     # per-candidate fact summed over the selected set
    unit: str = ""


class SelectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["selection"]
    action: Action                 # the action a correct decide-answer carries
    capacity: Capacity | None = None
    objective: Objective
    gates: list[Gate] = []


class ActionChoiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["action_choice"]
    rules: list[Rule]              # ordered, first-match; each needs actions


class SlotDecisionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["slot_decision"]
    action: Action                 # e.g. EXPERIMENT for an A/B readout
    slot: str                      # e.g. "ab_decision"
    labels: list[str]              # admissible slot values, rendered as the menu
    rules: list[Rule]              # ordered, first-match; each needs label


DecisionSpec = Annotated[
    Union[SelectionSpec, ActionChoiceSpec, SlotDecisionSpec],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Gold checks — stored for auditability; validate re-derives via the oracle
# and fails on any disagreement.
# ---------------------------------------------------------------------------


class _GoldBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    dimension: Dimension
    weight: float = Field(default=1.0, gt=0)


class PrincipalSelection(_GoldBase):
    check: Literal["principal_selection"]
    action: Action
    optimal_sets: list[list[str]]   # every optimal selection (order-insensitive)


class PrincipalAction(_GoldBase):
    check: Literal["principal_action"]
    acceptable_actions: list[Action]


class PrincipalSlot(_GoldBase):
    check: Literal["principal_slot"]
    action: Action
    slot: str
    acceptable: list[str]


class PrincipalClarify(_GoldBase):
    check: Literal["principal_clarify"]
    required_questions: list[str]   # decision-changing unknowns' questions
    allowed_questions: list[str]    # required + genuinely-unknown-but-inert


class ClusterLabelsCheck(_GoldBase):
    check: Literal["cluster_labels"]
    expected: dict[str, dict[str, str]]  # cluster id -> {label dimension -> label}


class SequenceCheck(_GoldBase):
    check: Literal["sequence"]
    # graded from candidate depends_on edges over the model's own selected set


class CoverageCheck(_GoldBase):
    check: Literal["coverage"]
    criteria: dict[str, list[str]]  # criterion id -> acceptable candidate ids


GoldCheck = Annotated[
    Union[
        PrincipalSelection,
        PrincipalAction,
        PrincipalSlot,
        PrincipalClarify,
        ClusterLabelsCheck,
        SequenceCheck,
        CoverageCheck,
    ],
    Field(discriminator="check"),
]

PRINCIPAL_CHECKS = frozenset(
    {"principal_selection", "principal_action", "principal_slot", "principal_clarify"}
)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    family: str                   # scenario family (equal-weighted in scoring)
    pair_id: str                  # matched-pair group
    variant: Variant
    domain: str
    difficulty: Difficulty
    version: str
    split: Split

    title: str
    company_context: str          # strategy, OKRs, situation — rendered prose
    evidence: str = ""            # documents/tickets/notes — rendered prose

    facts: dict[str, FactValue] = {}          # global fact table
    fact_definitions: dict[str, str] = {}     # fact name -> rendered glossary text
    facts_in_prose: bool = False  # haystack variants: item/cluster facts live in
                                  # evidence, not tables (lint enforces presence)

    candidates: list[Candidate] = []
    clusters: list[Cluster] = []
    success_criteria: list[SuccessCriterion] = []
    constraints: list[Constraint] = []
    questions: list[Question] = []            # the clarify menu
    unknowns: list[Unknown] = []

    decision_spec: DecisionSpec
    label_rules: dict[str, list[Rule]] = {}   # label dimension -> ordered rules
                                              # (evaluated per cluster)

    gold: list[GoldCheck]
    traps: list[Trap] = []
    gold_rationale: str

    def content_hash(self) -> str:
        """Stable hash of everything that affects the prompt or gold. Part of
        the prediction cache identity so edited scenarios never reuse stale
        model outputs."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def candidate_ids(self) -> set[str]:
        return {c.id for c in self.candidates}

    @property
    def cluster_ids(self) -> set[str]:
        return {c.id for c in self.clusters}

    @property
    def question_ids(self) -> set[str]:
        return {q.id for q in self.questions}

    @property
    def principal_check(self) -> GoldCheck:
        principals = [g for g in self.gold if g.check in PRINCIPAL_CHECKS]
        if len(principals) != 1:
            raise ValueError(
                f"{self.instance_id}: expected exactly one principal check, "
                f"found {len(principals)}"
            )
        return principals[0]


def load_scenario(path: Path) -> Scenario:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario.model_validate(data)


def load_scenarios(scenario_dir: Path) -> list[Scenario]:
    return [load_scenario(p) for p in sorted(scenario_dir.rglob("*.yaml"))]


# ---------------------------------------------------------------------------
# Answer spec (the model's graded response)
# ---------------------------------------------------------------------------


class ClusterLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal_type: SignalType
    confidence: ConfidenceLabel
    materiality: MaterialityLabel


class AnswerSpec(BaseModel):
    """The single JSON artifact a model submits per scenario.

    `notes` is never graded. Everything else is matched against oracle output.
    """
    model_config = ConfigDict(extra="forbid")

    decision_status: DecisionStatus

    # decide branch
    action: Action | None = None
    selected: list[str] = []                 # candidate ids
    sequence: list[str] = []                 # selected ids in execution order
    slots: dict[str, str] = {}               # e.g. {"ab_decision": "ship"}
    cluster_labels: dict[str, ClusterLabels] = {}
    coverage: dict[str, list[str]] = {}      # criterion id -> selected item ids

    # clarify branch
    questions: list[str] = []                # question ids from the menu

    notes: str = ""
