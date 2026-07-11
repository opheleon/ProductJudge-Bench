#!/usr/bin/env python
"""Authoring helper: print the oracle-derived gold for a scenario file.

Usage: python scripts/derive_gold.py scenarios/public/foo.yaml

The printed block is what the YAML gold section must contain for
`pjbench validate` to pass. Deriving first and pasting is fine — the point of
the stored gold is human review, and the rationale must still prove WHY the
oracle output is what it is.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pjbench.oracle import derive_gold, solve_selection_for  # noqa: E402
from pjbench.schema import SelectionSpec, load_scenario  # noqa: E402


def main() -> int:
    scenario = load_scenario(Path(sys.argv[1]))
    derived = derive_gold(scenario)
    print(f"# {scenario.instance_id}")
    if derived.must_clarify:
        print(f"must_clarify: true")
        print(f"required_questions: {sorted(derived.required_questions)}")
        print(f"allowed_questions: {sorted(derived.allowed_questions)}")
    else:
        print(f"outcome: {derived.outcome}")
        if isinstance(scenario.decision_spec, SelectionSpec):
            solution = solve_selection_for(scenario)
            print(f"optimal objective: {solution.optimal_objective}")
            ranked = sorted(
                solution.feasible_objectives.items(),
                key=lambda kv: kv[1],
                reverse=scenario.decision_spec.objective.sense == "maximize",
            )
            print("top feasible sets:")
            for s, v in ranked[:5]:
                print(f"  {sorted(s)}: {v}")
    if derived.cluster_labels:
        print("cluster_labels:")
        for cid, labels in sorted(derived.cluster_labels.items()):
            print(f"  {cid}: {labels}")
    if derived.coverage:
        print(f"coverage: { {k: sorted(v) for k, v in derived.coverage.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
