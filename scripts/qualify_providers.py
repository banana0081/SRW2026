"""Transport-only qualification of the OpenRouter providers behind one model.

The two evaluation models are each served by several providers. Across the
canonical runs the same documentation, the same seed and the same model id
produced materially different CP depending on who answered, so a provider has
to be pinned before any documentation variant is compared. This script decides
that pin **without ever scoring a benchmark outcome**.

What it measures, at the concurrency the real runs use:

- whether a long tool-catalog prompt comes back at all (transport failures),
- whether the completion is well-formed JSON of the declared shape,
- whether it stopped because it ran out of tokens (`finish_reason=length`),
- how long it took.

What it never touches: RestBench queries, gold paths, traces or CP. The
prompts are assembled from the released `Initial` tool catalog plus generic
instructions written here, so nothing in the choice can be a function of the
result it will later be used to measure.

  python scripts/qualify_providers.py --model flash \
      --providers wafer deepinfra digitalocean
  python scripts/qualify_providers.py --model ling --providers novita --report
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import threading
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # `python scripts/qualify_providers.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tooldoc_nir.draft_agent_reproduction import (  # noqa: E402
    CostBudget,
    CostCapReached,
    CallContext,
    FailureSink,
    HarnessFailure,
    LoggedOpenRouter,
    ProviderDrift,
    RELEASED_DECODING,
    _dump_json,
    _extract_json,
    read_jsonl,
)
from tooldoc_nir.openrouter import (  # noqa: E402
    OpenRouterClient,
    load_env_file,
    normalize_provider,
)
from tooldoc_nir.provenance import payload_digest  # noqa: E402
from tooldoc_nir.restbench_data import (  # noqa: E402
    DEFAULT_DRAFT_ROOT,
    load_instructions,
)

MODELS: dict[str, str] = {
    "ling": "inclusionai/ling-3.0-flash",
    "flash": "deepseek/deepseek-v4-flash-0731",
}

# Ruled out before any measurement, on the transport evidence already in the
# usage logs of the canonical runs.
EXCLUDED: dict[str, str] = {
    "sailresearch": "29.1% of its completions finished on the token limit",
}

QUALIFICATION_ROOT = Path("artifacts/results/provider_qualification")

# Generic instructions in the shape the released `task_decompose` stage sees.
# None of them is a RestBench query; they exist only to make the prompt as long
# and as JSON-demanding as a real one.
PROBE_INSTRUCTIONS: tuple[str, ...] = (
    "Describe the steps needed to look up one film and then read its crew.",
    "Describe the steps needed to find a studio and then read its logos.",
    "Describe the steps needed to find a series and then read its network.",
    "Describe the steps needed to find an actor and then read their profile.",
    "Describe the steps needed to list current releases and then read one of them.",
    "Describe the steps needed to find a boxed set and then read the films in it.",
)

SHAPE = '{"Tasks": ["<step>", "<step>"]}'


def catalog_prompt(
    instructions: Mapping[str, Mapping[str, Any]], instruction: str
) -> list[dict[str, str]]:
    """One long, JSON-constrained prompt built from the released catalog."""
    catalog = [
        {
            "tool_name": document.get("tool_name"),
            "description": str(document.get("description") or "")[:400],
            "required_parameters": [
                item.get("name") for item in document.get("required_parameters") or []
            ],
        }
        for document in instructions.values()
    ]
    return [
        {
            "role": "system",
            "content": (
                "You decompose an instruction into ordered steps over a REST "
                f"API catalog. Answer with JSON only, in the shape {SHAPE}."
            ),
        },
        {
            "role": "user",
            "content": (
                "API catalog:\n"
                + json.dumps(catalog, ensure_ascii=False)
                + f"\n\nInstruction: {instruction}\n\nAnswer with {SHAPE}."
            ),
        },
    ]


def _valid_shape(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("Tasks"), list)
        and bool(value["Tasks"])
    )


def qualify(
    *,
    model_key: str,
    provider: str,
    prompts: Sequence[list[dict[str, str]]],
    repeats: int,
    workers: int,
    max_cost_usd: float,
    root: Path,
) -> dict[str, Any]:
    """Send the same prompts to one provider and report transport health."""
    usage_path = root / f"{model_key}_{normalize_provider(provider)}.jsonl"
    usage_path.unlink(missing_ok=True)
    client = OpenRouterClient.from_env()
    budget = CostBudget(max_cost_usd)
    lock = threading.Lock()
    outcomes: list[dict[str, Any]] = []

    def one(index: int, messages: list[dict[str, str]]) -> dict[str, Any]:
        transport = LoggedOpenRouter(
            client=client,
            model=MODELS[model_key],
            role="qualification",
            usage_path=usage_path,
            budget=budget,
            context=CallContext(condition=provider, query_index=index),
            failures=FailureSink(),
            provider=provider,
        )
        try:
            text = transport.complete(
                messages=messages,
                temperature=RELEASED_DECODING["temperature"],
                top_p=RELEASED_DECODING["top_p"],
                max_tokens=RELEASED_DECODING["max_tokens"],
                stage="task_decompose",
                structured=False,
                seed=None,
            )
        except ProviderDrift as exc:
            return {"index": index, "outcome": "provider_drift", "detail": str(exc)}
        except HarnessFailure as exc:
            return {"index": index, "outcome": "transport_failure", "detail": str(exc)}
        try:
            parsed = _extract_json(text)
        except ValueError:
            return {"index": index, "outcome": "unparsed", "chars": len(text)}
        if not _valid_shape(parsed):
            return {"index": index, "outcome": "wrong_shape", "chars": len(text)}
        return {"index": index, "outcome": "ok", "chars": len(text)}

    jobs = [
        (position, prompts[position % len(prompts)])
        for position in range(repeats * len(prompts))
    ]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, index, messages) for index, messages in jobs]
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    outcomes.append(result)
                print(
                    f"[{model_key} {provider}] {len(outcomes)}/{len(jobs)} "
                    f"{result['outcome']}",
                    flush=True,
                )
    except CostCapReached as exc:
        print(f"STOP cost-cap: {exc}", flush=True)

    records = read_jsonl(usage_path)
    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in records)
    counts: dict[str, int] = {}
    for result in outcomes:
        counts[result["outcome"]] = counts.get(result["outcome"], 0) + 1
    attempted = len(outcomes)
    return {
        "model": MODELS[model_key],
        "provider": provider,
        "requested": len(jobs),
        "attempted": attempted,
        "outcomes": counts,
        "served_providers": sorted(
            {str(row.get("provider") or "") for row in records} - {""}
        ),
        "json_success_rate": (
            round(counts.get("ok", 0) / attempted, 4) if attempted else None
        ),
        "length_truncated_rate": (
            round(
                sum(1 for row in records if row.get("finish_reason") == "length")
                / len(records),
                4,
            )
            if records
            else None
        ),
        "transport_failure_rate": (
            round(
                (
                    counts.get("transport_failure", 0)
                    + counts.get("provider_drift", 0)
                )
                / attempted,
                4,
            )
            if attempted
        else None
        ),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": (
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            if latencies
            else None
        ),
        "cost_usd": round(budget.spent_usd, 6),
        "usage_log": str(usage_path),
    }


def rank(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick the pin on transport evidence, in a fixed order of criteria.

    JSON success first, then the token-limit rate, then median latency. The
    order is stated before the numbers exist so it cannot be reordered to
    favour whichever provider happened to look best on the benchmark.
    """
    eligible = []
    rejected = []
    for result in results:
        key = normalize_provider(str(result["provider"]))
        if key in EXCLUDED:
            rejected.append({**dict(result), "why": EXCLUDED[key]})
            continue
        if (result.get("transport_failure_rate") or 0.0) > 0.05:
            rejected.append(
                {**dict(result), "why": "over 5% transport failures at this concurrency"}
            )
            continue
        if result.get("json_success_rate") is None:
            rejected.append({**dict(result), "why": "no completed calls"})
            continue
        eligible.append(dict(result))
    eligible.sort(
        key=lambda result: (
            -float(result["json_success_rate"]),
            float(result.get("length_truncated_rate") or 0.0),
            float(result.get("median_latency_ms") or 0.0),
            str(result["provider"]),
        )
    )
    return {
        "criteria": [
            "json_success_rate desc",
            "length_truncated_rate asc",
            "median_latency_ms asc",
        ],
        "excluded_before_measurement": EXCLUDED,
        "eligible": [result["provider"] for result in eligible],
        "rejected": [
            {"provider": result["provider"], "why": result["why"]}
            for result in rejected
        ],
        "pin": eligible[0]["provider"] if eligible else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--providers", nargs="+", required=True)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-cost-usd", type=float, default=0.60)
    parser.add_argument("--output-root", type=Path, default=QUALIFICATION_ROOT)
    parser.add_argument(
        "--report",
        action="store_true",
        help="Re-rank an existing qualification without sending any request.",
    )
    args = parser.parse_args()

    root: Path = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / f"{args.model}_qualification.json"

    if args.report:
        if not report_path.exists():
            raise SystemExit(f"{report_path} does not exist yet")
        recorded = json.loads(report_path.read_text(encoding="utf-8"))
        recorded["ranking"] = rank(recorded["results"])
        _dump_json(report_path, recorded)
        print(json.dumps(recorded["ranking"], ensure_ascii=False, indent=2))
        return 0

    load_env_file()
    instructions = load_instructions(args.draft_root, "TMDB", "Initial")
    prompts = [
        catalog_prompt(instructions, instruction)
        for instruction in PROBE_INSTRUCTIONS
    ]
    results = [
        qualify(
            model_key=args.model,
            provider=provider,
            prompts=prompts,
            repeats=args.repeats,
            workers=args.workers,
            max_cost_usd=args.max_cost_usd,
            root=root,
        )
        for provider in args.providers
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model,
        "model": MODELS[args.model],
        "policy": (
            "transport-only: JSON validity, token-limit rate and latency at "
            "the production concurrency; no RestBench query, gold path or CP "
            "is read"
        ),
        "decoding": RELEASED_DECODING,
        "workers": args.workers,
        "repeats": args.repeats,
        "prompts": {
            "count": len(prompts),
            "digest": payload_digest(prompts),
            "instructions": list(PROBE_INSTRUCTIONS),
        },
        "results": results,
    }
    payload["ranking"] = rank(results)
    _dump_json(report_path, payload)
    print(json.dumps(payload["ranking"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
