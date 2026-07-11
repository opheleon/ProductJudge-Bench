"""Score aggregation and reporting.

Aggregation is family-weighted: items aggregate within a scenario family
first, then families average with equal weight — a 30-cluster feedback corpus
cannot outweigh an A/B scenario by sheer label count.

Headline **Decision** = principal-check pass rate (family-weighted): did the
model make the right call. **Overall** macro-averages all dimensions.
Feasibility rate and mean normalized regret are reported alongside as
diagnostics, never blended into the headline.
"""
from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from .grade import ScenarioResult
from .schema import PRINCIPAL_CHECKS


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fraction: float      # family-weighted
    n_families: int
    n_items: int


class RunScores(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model: str
    benchmark_version: str
    n_scenarios: int
    format_compliance: float
    branch_accuracy: float           # decide-vs-clarify
    decision: float                  # principal-check pass rate, family-weighted
    dimensions: dict[str, DimensionScore]
    overall: float                   # macro-average over dimensions
    feasibility_rate: float | None   # selection answers only
    mean_regret: float | None        # over feasible selection answers
    scenarios: list[ScenarioResult]


def _family_weighted(
    per_family: dict[str, tuple[float, float]],
) -> float:
    """per_family: family -> (weighted score sum, weight sum)."""
    fractions = [s / w for s, w in per_family.values() if w > 0]
    return sum(fractions) / len(fractions) if fractions else 0.0


def score(
    results: list[ScenarioResult], model: str, benchmark_version: str
) -> RunScores:
    # dimension -> family -> (score sum, weight sum)
    dim_family: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    dim_items: dict[str, int] = defaultdict(int)
    principal_family: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    feas_total = feas_ok = 0
    regrets: list[float] = []

    for result in results:
        for item in result.items:
            if not item.applicable:
                continue
            bucket = dim_family[item.dimension][result.family]
            bucket[0] += item.weight * item.score
            bucket[1] += item.weight
            dim_items[item.dimension] += 1
            if item.check in PRINCIPAL_CHECKS:
                pb = principal_family[result.family]
                pb[0] += item.weight * item.score
                pb[1] += item.weight
        if result.feasible is not None:
            feas_total += 1
            feas_ok += result.feasible
            if result.regret is not None:
                regrets.append(result.regret)

    dimensions = {
        dim: DimensionScore(
            fraction=_family_weighted(
                {f: (s, w) for f, (s, w) in families.items()}
            ),
            n_families=len(families),
            n_items=dim_items[dim],
        )
        for dim, families in sorted(dim_family.items())
    }
    overall = (
        sum(d.fraction for d in dimensions.values()) / len(dimensions)
        if dimensions
        else 0.0
    )
    n = len(results)
    return RunScores(
        model=model,
        benchmark_version=benchmark_version,
        n_scenarios=n,
        format_compliance=sum(r.format_compliant for r in results) / n if n else 0.0,
        branch_accuracy=sum(r.branch_correct for r in results) / n if n else 0.0,
        decision=_family_weighted(
            {f: (s, w) for f, (s, w) in principal_family.items()}
        ),
        dimensions=dimensions,
        overall=overall,
        feasibility_rate=feas_ok / feas_total if feas_total else None,
        mean_regret=sum(regrets) / len(regrets) if regrets else None,
        scenarios=results,
    )


def render_report(all_scores: list[RunScores]) -> str:
    """Markdown leaderboard plus per-model failed/skipped item detail."""
    dims = sorted({d for s in all_scores for d in s.dimensions})
    header = (
        ["Model", "Overall", "Decision"]
        + dims
        + ["Branch", "Format", "Feasible", "Regret"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def pct(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:.0f}%"

    for s in sorted(all_scores, key=lambda s: s.overall, reverse=True):
        row = [s.model, pct(s.overall), pct(s.decision)]
        for d in dims:
            row.append(pct(s.dimensions[d].fraction) if d in s.dimensions else "—")
        row += [
            pct(s.branch_accuracy),
            pct(s.format_compliance),
            pct(s.feasibility_rate),
            "—" if s.mean_regret is None else f"{s.mean_regret * 100:.1f}%",
        ]
        lines.append("| " + " | ".join(row) + " |")

    for s in all_scores:
        failed = [
            (r.instance_id, i)
            for r in s.scenarios
            for i in r.items
            if i.applicable and not i.passed
        ]
        if failed:
            lines += ["", f"### Failed items — {s.model}", ""]
            for instance_id, item in failed:
                lines.append(
                    f"- `{instance_id}` / `{item.gold_id}` ({item.dimension}, "
                    f"score {item.score:.2f}): {item.detail}"
                )
        skipped = [
            (r.instance_id, i)
            for r in s.scenarios
            for i in r.items
            if not i.applicable
        ]
        if skipped:
            lines += ["", f"### Skipped conditional diagnostics — {s.model}", ""]
            for instance_id, item in skipped:
                lines.append(f"- `{instance_id}` / `{item.gold_id}`: {item.detail}")
    return "\n".join(lines)
