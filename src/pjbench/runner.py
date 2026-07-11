"""Model execution: scenarios -> predictions.jsonl.

Generation is decoupled from grading (SWE-bench convention): this module's only
job is to produce a predictions file; grading never makes an LLM call.

Reliability contract (inherited from SysDesign-Bench):
- every request carries a hard timeout (providers.py);
- a provider exception fails THAT scenario's prediction, not the run;
- predictions append incrementally, so a killed run is resumable.

Integrity contract (new here):
- every record carries the scenario content hash and the sha256 of the exact
  rendered prompt; grading refuses records whose scenario hash does not match
  the current YAML;
- the prediction cache is keyed by (model, reasoning, instance_id) AND the
  scenario hash — editing a scenario can never silently reuse a stale answer.

Format handling follows the BFCL convention: one repair attempt with the
validation error fed back; after that the prediction is non-compliant.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from .prompt import build_prompt
from .providers import CompleteFn, Usage
from .schema import AnswerSpec, Scenario

REPAIR_TEMPLATE = (
    "Your previous response was not a valid decision spec: {error}\n"
    "Respond again with ONLY the corrected JSON object, nothing else."
)


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    instance_id: str
    model: str
    reasoning: str = "none"
    scenario_hash: str = ""
    prompt_sha256: str = ""
    raw_output: str
    spec: AnswerSpec | None
    format_compliant: bool
    repair_attempted: bool
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0

    @property
    def is_provider_failure(self) -> bool:
        """Infrastructure failure (timeout, 4xx/5xx) — retryable, never cached."""
        return self.error.startswith("provider error")


def extract_json(text: str) -> str | None:
    """Return the first balanced top-level JSON object in `text`, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_spec(text: str) -> tuple[AnswerSpec | None, str]:
    blob = extract_json(text)
    if blob is None:
        return None, "no JSON object found in response"
    try:
        return AnswerSpec.model_validate(json.loads(blob)), ""
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    except ValidationError as exc:
        return None, f"schema violation: {exc}"


def run_scenario(
    scenario: Scenario,
    model_spec: str,
    complete_fn: CompleteFn,
    reasoning: str = "none",
) -> PredictionRecord:
    import time  # noqa: PLC0415

    prompt = build_prompt(scenario)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    messages: list[dict] = [{"role": "user", "content": prompt}]
    usage = Usage()
    started = time.monotonic()

    def _record(raw: str, spec, error: str, repaired: bool) -> PredictionRecord:
        return PredictionRecord(
            instance_id=scenario.instance_id,
            model=model_spec,
            reasoning=reasoning,
            scenario_hash=scenario.content_hash(),
            prompt_sha256=prompt_sha,
            raw_output=raw,
            spec=spec,
            format_compliant=spec is not None,
            repair_attempted=repaired,
            error=error,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=round(usage.cost_usd, 6),
            duration_s=round(time.monotonic() - started, 2),
        )

    try:
        completion = complete_fn(messages)
    except Exception as exc:  # provider failure fails the scenario, not the run
        return _record("", None, f"provider error: {exc}", False)
    usage = usage + completion.usage
    spec, error = parse_spec(completion.text)
    if spec is not None:
        return _record(completion.text, spec, "", False)

    messages = messages + [
        {"role": "assistant", "content": completion.text},
        {"role": "user", "content": REPAIR_TEMPLATE.format(error=error)},
    ]
    try:
        retry = complete_fn(messages)
    except Exception as exc:
        return _record(completion.text, None, f"provider error on repair: {exc}", True)
    usage = usage + retry.usage
    spec, error = parse_spec(retry.text)
    return _record(retry.text, spec, error, True)


def run_scenarios(
    scenarios: list[Scenario],
    model_spec: str,
    complete_fn: CompleteFn,
    reasoning: str = "none",
    skip_instance_ids: set[str] | None = None,
    on_progress: Callable[[PredictionRecord], None] | None = None,
    parallel: int = 1,
) -> list[PredictionRecord]:
    """Run scenarios, optionally with a thread pool. on_progress is always
    invoked from the calling thread so file appends stay single-threaded."""
    skip = skip_instance_ids or set()
    todo = [s for s in scenarios if s.instance_id not in skip]
    records: list[PredictionRecord] = []

    if parallel <= 1:
        for scenario in todo:
            record = run_scenario(scenario, model_spec, complete_fn, reasoning)
            records.append(record)
            if on_progress:
                on_progress(record)
        return records

    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(run_scenario, s, model_spec, complete_fn, reasoning): s
            for s in todo
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if on_progress:
                on_progress(record)
    return records


# ---------------------------------------------------------------------------
# Prediction cache: (model, reasoning, instance_id, scenario_hash) -> record.
# Provider failures are never cached. Genuine outputs, including
# format-noncompliant ones, are valid benchmark results and are cached.
# ---------------------------------------------------------------------------


def cache_path(cache_dir: Path, model_spec: str, reasoning: str, instance_id: str) -> Path:
    safe_model = model_spec.replace("/", "__")
    return cache_dir / safe_model / reasoning / f"{instance_id}.json"


def load_cached(
    cache_dir: Path,
    model_spec: str,
    reasoning: str,
    scenario: Scenario,
) -> PredictionRecord | None:
    path = cache_path(cache_dir, model_spec, reasoning, scenario.instance_id)
    if not path.is_file():
        return None
    try:
        record = PredictionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None  # unreadable cache entry counts as a miss
    if record.scenario_hash != scenario.content_hash():
        return None  # scenario was edited since this prediction — stale
    return record


def store_cached(cache_dir: Path, record: PredictionRecord) -> None:
    if record.is_provider_failure:
        return
    path = cache_path(cache_dir, record.model, record.reasoning, record.instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(record.model_dump_json(), encoding="utf-8")
    tmp.replace(path)


def append_prediction(record: PredictionRecord, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
        f.flush()


def write_predictions(records: list[PredictionRecord], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def read_predictions(path: Path) -> list[PredictionRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(PredictionRecord.model_validate_json(line))
    return records


class PredictionIntegrityError(ValueError):
    pass


def match_predictions(
    scenarios: list[Scenario], records: list[PredictionRecord]
) -> dict[str, PredictionRecord]:
    """Pair every expected scenario with exactly one prediction, verifying the
    scenario hash. Missing, duplicate, or stale predictions are hard errors —
    a leaderboard number must never silently rest on a partial or mixed run."""
    by_id: dict[str, PredictionRecord] = {}
    problems: list[str] = []
    for record in records:
        if record.instance_id in by_id:
            problems.append(f"duplicate prediction for {record.instance_id}")
        by_id[record.instance_id] = record

    expected = {s.instance_id: s for s in scenarios}
    for instance_id, scenario in sorted(expected.items()):
        record = by_id.get(instance_id)
        if record is None:
            problems.append(f"missing prediction for {instance_id}")
        elif record.scenario_hash and record.scenario_hash != scenario.content_hash():
            problems.append(
                f"stale prediction for {instance_id}: scenario edited since the "
                f"run (hash {record.scenario_hash} != {scenario.content_hash()})"
            )
    extras = set(by_id) - set(expected)
    for extra in sorted(extras):
        problems.append(f"prediction for unknown scenario {extra}")
    if problems:
        raise PredictionIntegrityError(
            "predictions do not match the scenario set:\n- " + "\n- ".join(problems)
        )
    return by_id
