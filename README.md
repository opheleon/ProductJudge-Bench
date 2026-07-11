# ProductJudge-Bench

**A fully deterministic benchmark for policy-conditioned product decisions.**

LLMs are marketed as product copilots, but nothing measures whether they make
the calls a product manager is paid to make: prioritizing the right items under
a real capacity constraint, telling a genuine trend from a loud account,
knowing when to configure instead of build, picking the request that serves the
strategy over the one that serves the loudest logo, and going back to the
customer when a load-bearing fact is missing. ProductJudge-Bench measures those
decisions, and it does so **without an LLM judge anywhere in the grading path**.

## The construct, stated precisely

ProductJudge-Bench evaluates **policy-conditioned product decisions**:
selecting, rejecting, sequencing, or investigating options under stated
objectives, constraints, and evidence. Each scenario states its decision policy
explicitly; the model's job is to apply that policy to adversarial evidence
where the load-bearing fact is buried, the loud signal is hollow, or the
canonical framework answer is wrong under the stated rules.

It deliberately does **not** measure inventing the strategy, framing new
opportunities, persuasion, stakeholder management, or product taste. Those may
matter more; they are not deterministically gradable, so they are out of scope
by design.

## How it works

Each scenario gives the model a company context, evidence documents, fact
definitions, and a decision policy. The model responds with one structured
JSON decision, not a prose essay:

- `decision_status`: `decide` or `clarify`
- deciding: an `action` (`build` / `configure` / `partner` / `experiment` /
  `defer` / `decline`) plus the payload the scenario calls for — selected
  items, execution sequence, decision slots, feedback-cluster labels, a
  coverage map
- clarifying: question ids from the scenario's menu

An ungraded `notes` field lets models explain themselves; it earns nothing.

### The oracle is the source of truth

Every scenario carries a machine-executable `decision_spec`: capacity and
objective for selections, ordered first-match rules for action and slot
decisions, threshold rules for cluster labels, hard gates written as
predicates over item facts. A pure-Python oracle (`oracle.py`) enumerates
every feasible answer and derives the gold answer — all optima, not one
author-blessed pick. The policy text the model reads is rendered **from the
same expressions the oracle executes**, so the stated policy and the graded
policy cannot diverge. `pjbench validate` re-derives gold on every run and
fails on any disagreement with the YAML; a scenario whose hand-written answer
contradicts its own policy is broken by definition and cannot ship.

Clarification is derived decision-theoretically, not from a checklist: a
scenario's unknowns carry admissible values, the oracle solves the scenario
once per assignment, and a question is **required** iff two assignments
differing only in that fact change the optimal outcome. Facts that are
missing but inert do not justify asking; facts stated in the prompt make
asking a failure.

## Scored dimensions

| Dimension | What passes |
|---|---|
| Decision | The principal call: the exact-optimal selection, the policy-entailed action, or the pre-registered readout |
| Clarification | Underdetermined scenarios: asks the decision-changing questions; determined scenarios: decides without stalling |
| Signal analysis | Feedback clusters labeled per the stated rules on three separated axes: signal type, confidence, materiality (balanced accuracy — labeling everything noise cannot exploit class imbalance) |
| Sequencing | Execution order respects dependency edges |
| Coverage | Claimed criterion-to-item edges are valid and the selected set actually satisfies the criteria |

Alongside the binary results, selections report **feasibility** and
**normalized regret** — a 1% miss and a catastrophic miss stop looking
identical. Scoring is family-weighted: scenario families count equally, so a
30-label corpus cannot outweigh an A/B readout. Wrong-branch answers fail the
principal check once; downstream items become conditional diagnostics rather
than repeated zeros.

## Anti-gaming: matched pairs, not vibes

Every scenario belongs to a matched pair:

- **Counterfactual pairs** change one load-bearing fact and the correct
  decision must flip (enforced by the validator). A model that learned
  "distrust the big number" fails the variant where the big number is right.
- **Invariance pairs** reorder evidence and add irrelevant documents; the
  decision must not move.
- **Extraction pairs** present identical facts as tidy tables vs. realistic
  document piles with decoys, separating reading from judgment.
- **Arithmetic pairs** present raw figures vs. precomputed metrics,
  separating calculation from judgment.

Symmetry is enforced across the set: fully-determined scenarios punish
reflexive clarifying, healthy asks punish reflexive declining, and control
variants make the canonical answer correct so second-guessing has a price.

### Deterministic baselines

`pjbench baseline` runs seven no-LLM strategies through the identical grading
path: always-build, always-clarify, always-decline, select-all, greedy
value-density, all-noise, and **the oracle itself, which must score 100% — a
permanent end-to-end self-test of the harness**. A model result means nothing
unless it beats every trivial baseline; v0.1 floors: best trivial strategy 57%
overall, 50% on Decision.

## Quickstart

    pip install -e ".[run]"          # plain `pip install -e .` suffices for grading
    pjbench validate                                    # oracle-equality + lint
    pjbench run --model anthropic/claude-fable-5 --reasoning high   # -> predictions.jsonl
    pjbench baseline                                    # -> baselines.jsonl (no API keys)
    pjbench grade predictions.jsonl                     # -> scores.json (no LLM calls)
    pjbench report scores.json                          # -> markdown leaderboard

Grading makes zero LLM calls; only `run` needs API keys (copy `.env.example`
to `.env`). Predictions are cached under
`(model, reasoning effort, instance, scenario-content hash)` — editing a
scenario can never silently reuse a stale answer — and grading refuses
missing, duplicate, or stale predictions outright.

## Dataset status

v0.1 contains **12 public scenarios**: 6 families x 2 matched variants
(capacity prioritization, feedback signal analysis, build/configure/decline,
conflicting customer requests, A/B readouts, clarify-vs-decide). This is a
harness pilot, not a leaderboard: discrimination claims wait for at least
three model families, repeat runs, and a held-out split, per the roadmap.

## Related work

We found no benchmark specifically targeting deterministic product-management
decisions. The adjacent efforts differ on exactly the axis that matters here:

- **GDPval** (OpenAI) includes PM-adjacent occupations, grading realistic
  work products via blinded expert comparison — deliverable quality, not
  decision correctness.
- **TheAgentCompany** (CMU) tests project-manager *workflow execution* in a
  simulated company, with LLM-judge fallback for non-mechanical checkpoints.
- **ManagerBench** deterministically grades managerial safety-vs-pragmatism
  choices; **RetailBench** uses an oracle policy over a retail simulation for
  long-horizon operational decisions. Neither covers product judgment.
- **SysDesign-Bench** (our sibling project) applies the same
  deterministic-grading philosophy to system design; ProductJudge-Bench adds
  the executable-oracle layer on top of it.

## Philosophy

1. **Deterministic or it doesn't ship.** Grading is set, enum, and arithmetic
   matching in pure Python. A criterion that cannot be expressed that way is
   not a valid item.
2. **The oracle outranks the author.** Gold answers are derived from the
   executable policy, never hand-blessed; the validator enforces equality.
   Hand-written rationales exist for human audit of the policy, not as truth.
3. **Difficulty comes from evidence, not grading.** The policy is public and
   verbatim in the prompt. What is hard is noticing the changelog that makes
   the build unnecessary, the one account behind 42 tickets, the gate a
   contractual fact triggers, the threshold an unknown straddles.
4. **Every optimal answer passes; near-misses are measured.** All optima are
   accepted, feasible-but-suboptimal selections earn regret rather than a
   bare zero, and precedence checks accept any dependency-respecting order.

## Contributing

Scenario PRs must include a machine-executable `decision_spec`, a matched
pair, and a `gold_rationale` with the worked math (the human review
artifact). `pjbench validate` must pass, which enforces: gold equals oracle
output, counterfactual pairs flip, invariance/haystack/precomputed variants
do not, rule lists terminate, gates carry a rendered constraint basis, prose
variants literally contain every numeric fact, and no gold or trap material
leaks into the rendered prompt. `scripts/derive_gold.py` prints the oracle's
answer for a scenario file during authoring.

What gets rejected: items requiring taste rather than entailment, policies
that give away the application (item-id gates), unmatched single scenarios,
and free-text judgments.

## License

Code: MIT. Dataset (`scenarios/`): CC-BY-4.0.
