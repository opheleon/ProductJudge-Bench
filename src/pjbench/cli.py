"""ProductJudge-Bench command-line interface.

Decoupled generate/grade flow (SWE-bench convention):
  pjbench run      -> predictions.jsonl   (needs API keys; the only LLM step)
  pjbench baseline -> predictions.jsonl   (deterministic strategies, no LLM)
  pjbench grade    -> scores.json         (pure Python, zero LLM calls)
  pjbench report   -> markdown table
  pjbench validate -> oracle-equality + authoring lint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .baselines import BASELINES
from .grade import grade_scenario
from .prompt import build_prompt
from .providers import build_complete_fn, estimate_cost, parse_model_spec
from .runner import (
    PredictionIntegrityError,
    PredictionRecord,
    load_cached,
    match_predictions,
    read_predictions,
    run_scenarios,
    store_cached,
    write_predictions,
)
from .schema import Scenario, load_scenario, load_scenarios
from .scoring import RunScores, render_report, score
from .validate import validate_scenario, validate_set


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from .env into the environment (existing vars win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


def _load_scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios = load_scenarios(Path(args.scenarios))
    if not scenarios:
        raise SystemExit(f"no scenario yaml files found under {args.scenarios}")
    return scenarios


def _cmd_validate(args: argparse.Namespace) -> int:
    scenario_dir = Path(args.scenarios)
    paths = sorted(scenario_dir.rglob("*.yaml"))
    if not paths:
        print(f"FAIL: no scenario yaml files found under {scenario_dir}")
        return 1

    errors: list[str] = []
    scenarios: list[Scenario] = []
    n_ok = 0
    for path in paths:
        try:
            scenario = load_scenario(path)
        except (OSError, ValueError, ValidationError) as exc:
            errors.append(f"{path}: failed to parse: {exc}")
            continue
        if scenario.instance_id != path.stem:
            errors.append(
                f"{path}: filename stem != instance_id {scenario.instance_id!r}"
            )
        scenarios.append(scenario)
        scenario_errors = validate_scenario(scenario)
        if scenario_errors:
            errors.extend(scenario_errors)
        else:
            n_ok += 1

    errors.extend(validate_set(scenarios))

    for e in errors:
        print(f"ERROR: {e}")
    print(f"{n_ok}/{len(paths)} scenarios valid, {len(errors)} error(s).")
    return 1 if errors else 0


# Assumed completion tokens per scenario for pre-run cost estimates; with
# high reasoning effort, thinking tokens bill as output.
_EST_COMPLETION_TOKENS = {"none": 3000, "low": 5000, "medium": 8000, "high": 12000}


def _cmd_run(args: argparse.Namespace) -> int:
    _load_dotenv()
    scenarios = _load_scenarios(args)
    reasoning = None if args.reasoning == "none" else args.reasoning
    out = Path(args.out)
    cache_dir = Path(args.cache_dir)

    if args.only:
        wanted = {i.strip() for i in args.only.split(",") if i.strip()}
        unknown = wanted - {s.instance_id for s in scenarios}
        if unknown:
            raise SystemExit(f"--only names unknown scenario(s): {sorted(unknown)}")
        scenarios = [s for s in scenarios if s.instance_id in wanted]

    skip: set[str] = set()
    if args.resume and out.exists():
        skip = {r.instance_id for r in read_predictions(out)}
        print(f"Resuming: {len(skip)} prediction(s) already in {out}.", flush=True)
    elif out.exists() and not args.only:
        out.unlink()

    # Partition into cache hits and scenarios needing API calls. The cache key
    # includes reasoning effort AND the scenario content hash — an edited
    # scenario is always a miss.
    cached: list[PredictionRecord] = []
    todo: list[Scenario] = []
    for s in scenarios:
        if s.instance_id in skip:
            continue
        hit = (
            None
            if args.no_cache
            else load_cached(cache_dir, args.model, args.reasoning, s)
        )
        if hit is not None:
            cached.append(hit)
        else:
            todo.append(s)

    _, model_id = parse_model_spec(args.model)
    est_prompt_tokens = sum(len(build_prompt(s)) // 4 for s in todo)
    est_completion = _EST_COMPLETION_TOKENS[args.reasoning] * len(todo)
    est = estimate_cost(model_id, est_prompt_tokens, est_completion)
    print(
        f"Running {args.model} | {len(cached)} cached, {len(todo)} to run "
        f"(reasoning={args.reasoning}, parallel={args.parallel}) | "
        f"estimated cost for uncached: ~${est:.2f} "
        f"(~{est_prompt_tokens:,} prompt + ~{est_completion:,} completion tokens)",
        flush=True,
    )

    complete_fn = build_complete_fn(
        args.model, reasoning=reasoning, api_base=args.api_base,
        api_key_env=args.api_key_env,
    )

    def on_progress(r: PredictionRecord) -> None:
        store_cached(cache_dir, r)
        status = "ok" if r.format_compliant else f"FAIL ({r.error[:60]})"
        print(
            f"  {r.instance_id}: {status}"
            f"{' (after repair)' if r.repair_attempted and r.format_compliant else ''}"
            f"  [${r.cost_usd:.4f}]",
            flush=True,
        )

    fresh = run_scenarios(
        scenarios, args.model, complete_fn,
        reasoning=args.reasoning,
        skip_instance_ids=skip | {c.instance_id for c in cached},
        on_progress=on_progress,
        parallel=args.parallel,
    )

    by_id = {r.instance_id: r for r in cached}
    by_id.update({r.instance_id: r for r in fresh})
    if (args.only or args.resume) and out.exists():
        for r in read_predictions(out):
            by_id.setdefault(r.instance_id, r)
    ordered_ids = [s.instance_id for s in load_scenarios(Path(args.scenarios))]
    records = [by_id[i] for i in ordered_ids if i in by_id]
    write_predictions(records, out)

    compliant = sum(r.format_compliant for r in records)
    total_cost = sum(r.cost_usd for r in fresh)
    print(
        f"Wrote {len(records)} prediction(s) to {out} "
        f"({compliant}/{len(records)} format-compliant; {len(cached)} from cache). "
        f"Actual cost this run: ${total_cost:.2f} "
        f"({sum(r.prompt_tokens for r in fresh):,} prompt + "
        f"{sum(r.completion_tokens for r in fresh):,} completion tokens).",
        flush=True,
    )
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    scenarios = _load_scenarios(args)
    strategies = (
        sorted(BASELINES) if args.strategy == "all" else [args.strategy]
    )
    records: list[PredictionRecord] = []
    for strategy in strategies:
        answer_fn = BASELINES[strategy]
        for scenario in scenarios:
            answer = answer_fn(scenario)
            records.append(
                PredictionRecord(
                    instance_id=scenario.instance_id,
                    model=f"baseline/{strategy}",
                    reasoning="none",
                    scenario_hash=scenario.content_hash(),
                    raw_output=answer.model_dump_json(),
                    spec=answer,
                    format_compliant=True,
                    repair_attempted=False,
                )
            )
    write_predictions(records, Path(args.out))
    print(
        f"Wrote {len(records)} baseline prediction(s) "
        f"({len(strategies)} strateg{'ies' if len(strategies) != 1 else 'y'} x "
        f"{len(scenarios)} scenarios) to {args.out}."
    )
    return 0


def _cmd_grade(args: argparse.Namespace) -> int:
    scenarios = _load_scenarios(args)
    records = read_predictions(Path(args.predictions))
    if not records:
        raise SystemExit(f"no predictions found in {args.predictions}")

    all_scores: list[RunScores] = []
    for model in sorted({r.model for r in records}):
        model_records = [r for r in records if r.model == model]
        try:
            by_id = match_predictions(scenarios, model_records)
        except PredictionIntegrityError as exc:
            raise SystemExit(f"{model}: {exc}") from exc
        results = [
            grade_scenario(s, by_id[s.instance_id].spec) for s in scenarios
        ]
        all_scores.append(score(results, model=model, benchmark_version=__version__))

    out = Path(args.out)
    out.write_text(
        json.dumps([s.model_dump(mode="json") for s in all_scores], indent=2),
        encoding="utf-8",
    )
    for s in all_scores:
        regret = "n/a" if s.mean_regret is None else f"{s.mean_regret:.3f}"
        print(
            f"{s.model}: overall {s.overall:.2f}, decision {s.decision:.2f} "
            f"(branch {s.branch_accuracy:.2f}, format {s.format_compliance:.2f}, "
            f"mean regret {regret})"
        )
    print(f"Wrote {out}.")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    all_scores = []
    for path in args.scores:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        all_scores.extend(RunScores.model_validate(entry) for entry in data)
    print(render_report(all_scores))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pjbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_scenarios_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--scenarios", default="scenarios/public",
                       help="Scenario directory (default: the public split)")

    p_validate = sub.add_parser("validate", help="Oracle-equality + authoring lint")
    add_scenarios_arg(p_validate)
    p_validate.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="Run a model over scenarios -> predictions.jsonl")
    add_scenarios_arg(p_run)
    p_run.add_argument("--model", required=True,
                       help="provider/model-id: anthropic/claude-fable-5, openai/gpt-5.6-sol, "
                            "xai/grok-4.5, baseten/zai-org/GLM-5.2")
    p_run.add_argument("--out", default="predictions.jsonl")
    p_run.add_argument("--reasoning", choices=["none", "low", "medium", "high"], default="none")
    p_run.add_argument("--resume", action="store_true",
                       help="Skip instance_ids already present in --out (crash recovery)")
    p_run.add_argument("--only", default=None,
                       help="Comma-separated instance_ids to (re)run; other records kept")
    p_run.add_argument("--parallel", type=int, default=4)
    p_run.add_argument("--no-cache", action="store_true")
    p_run.add_argument("--cache-dir", default="results/cache",
                       help="Cache keyed by (model, reasoning, instance, scenario hash)")
    p_run.add_argument("--api-base", default=None)
    p_run.add_argument("--api-key-env", default=None)
    p_run.set_defaults(func=_cmd_run)

    p_baseline = sub.add_parser(
        "baseline", help="Emit deterministic baseline predictions (no LLM)"
    )
    add_scenarios_arg(p_baseline)
    p_baseline.add_argument("--strategy", default="all",
                            choices=["all", *sorted(BASELINES)])
    p_baseline.add_argument("--out", default="baselines.jsonl")
    p_baseline.set_defaults(func=_cmd_baseline)

    p_grade = sub.add_parser("grade", help="Grade predictions.jsonl -> scores.json (no LLM calls)")
    add_scenarios_arg(p_grade)
    p_grade.add_argument("predictions")
    p_grade.add_argument("--out", default="scores.json")
    p_grade.set_defaults(func=_cmd_grade)

    p_report = sub.add_parser("report", help="Render markdown report from scores.json file(s)")
    p_report.add_argument("scores", nargs="+")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
