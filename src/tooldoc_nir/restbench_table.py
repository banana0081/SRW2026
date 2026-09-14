"""Reproduce DRAFT Table 1 RestBench-TMDB CP%: ReAct, DFSDT, DRAFT.

How this maps to the paper:
- Benchmark and gold paths come from RestGPT (same 100 TMDB queries).
- The controller is the released Inference_DFSDT.py (EasyTool-style DFS),
  because that is the only agent DRAFT shipped, and RestBench is stored in
  that agent's schema (Tool_dic + relevant APIs).
- DFSDT row: that agent + TMDB_Initial.json
- DRAFT row: that agent + TMDB_DRAFT.json
- Ours row: that agent + TMDB_Initial plus hop fill-contracts (where the
  path id comes from). choose_tool keeps the Initial purpose; the fill
  line is only in the guideline. Schema unchanged.
- ReAct row: the same retrieval/answer/check loop on the original question,
  without task_decompose / topology / summarize
- EasyTool is skipped: DRAFT did not release RestBench EasyTool docs
- Only allowed substitution: evaluation model (default Ling flash)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import threading
import traceback
from types import SimpleNamespace
from typing import Any

from tooldoc_nir.draft_agent_reproduction import (
    FORMAT_ATTEMPTS,
    FORMAT_CONTRACTS,
    CallContext,
    CostBudget,
    CostCapReached,
    FailureSink,
    HarnessFailure,
    LoggedOpenRouter,
    PROVIDER_DRIFT,
    RELEASED_DECODING,
    ProviderDrift,
    ReleasedDriverFailure,
    TRANSPORT,
    _dump_json,
    _extract_json,
    _load_json,
    load_released_draft_module,
    model_alias,
)
from tooldoc_nir.openrouter import OpenRouterClient, OpenRouterError, load_env_file
from tooldoc_nir.provenance import file_digest, payload_digest
from tooldoc_nir.restbench_adapt import url_index, wrap_instructions, wrap_query
from tooldoc_nir.restbench_data import (
    DEFAULT_DRAFT_ROOT,
    DEFAULT_MODEL,
    correct_path as gold_subsequence,
    gold_apis,
    instruction_path,
    load_instructions,
    load_queries,
    test_path,
)
from tooldoc_nir.restbench_ours import build_ours_instructions
from tooldoc_nir.restbench_http import SpotifyClient, TmdbClient
from tooldoc_nir.restbench_tmdb_backend import TmdbDfsdtBackend

ROWS = (
    ("ReAct", "react", "Initial"),
    ("DFSDT", "dfsdt", "Initial"),
    ("DRAFT", "draft", "DRAFT"),
    ("Ours", "dfsdt", "Ours"),
)

# The error policy, split so that the runner can retry infrastructure without
# inventing a path:
#
# - `transport`      network, 429, 5xx, a dead provider. Identical retry.
# - `provider_drift` the pinned endpoint was not the one that answered. Same
#                    treatment as transport, reported separately.
# - `driver`         a malformed or truncated completion after the released
#                    three-sample resample. Inference_DFSDT then continues
#                    (`{}` / None), it does not score the episode as a miss.
#                    A leftover driver error is retried or dropped from n;
#                    `cp_over_n` is not the published comparison.
TRANSPORT_ERROR_KINDS = (TRANSPORT, PROVIDER_DRIFT)
DRIVER_ERROR = "driver"


def error_kind(row: dict[str, Any]) -> str:
    """Read the error class off a trace row, defaulting to a model result."""
    if not row.get("error"):
        return ""
    return str(row.get("error_kind") or DRIVER_ERROR)


def is_transport_error(row: dict[str, Any]) -> bool:
    return error_kind(row) in TRANSPORT_ERROR_KINDS


def is_credits_exhausted(error: str | None) -> bool:
    text = error or ""
    return "HTTP 402" in text or "Insufficient credits" in text


def is_endpoint_dead(error: str | None) -> bool:
    """The chat endpoint never accepted a TCP connection.

    Distinct from a slow completion or a 5xx: those still retry. A connect
    timeout on a local box means the seed should stop, not spend hours
    writing transport misses.
    """
    text = error or ""
    return any(
        marker in text
        for marker in (
            "ConnectTimeout",
            "Connection refused",
            "NewConnectionError",
            "NameResolutionError",
            "Failed to establish a new connection",
        )
    )


def refuse_openrouter_for_local_model(model: str, client: Any) -> None:
    token = (model or "").lower()
    if "qwen" in token and not bool(getattr(client, "local", False)):
        raise SystemExit(
            f"{model} must use OPENAI_BASE_URL; refusing OpenRouter"
        )


_tls = threading.local()
_print_lock = threading.Lock()
_log_lock = threading.Lock()


def _executed_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    log = record.get("execute_log") or {}
    for group in log.get("api_result_ls") or []:
        for api in group:
            names.append(str(api.get("api_name") or api.get("tool_name") or ""))
    return names


def _score(record: dict[str, Any], gold: list[str]) -> bool:
    return gold_subsequence(_executed_names(record), gold)


def _redirect_open(draft: Any) -> None:
    real_open = open

    def redirected(file: Any, *args: Any, **kwargs: Any) -> Any:
        target = getattr(_tls, "jsonl_path", None)
        if target and isinstance(file, str) and file.endswith(".jsonl"):
            file = str(target)
        return real_open(file, *args, **kwargs)

    def quiet_print(*args: Any, **kwargs: Any) -> None:
        path = getattr(_tls, "print_log", None)
        if not path:
            return
        with real_open(path, "a", encoding="utf-8") as handle:
            print(*args, file=handle)

    draft.open = redirected
    draft.print = quiet_print


def _install_driver(draft: Any) -> None:
    def openai_response(
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_name: str,
        is_string: bool,
    ) -> Any:
        del model_name
        transport: LoggedOpenRouter = _tls.transport
        counters = _tls.counters
        stage = inspect.stack()[1].function
        shape, is_valid = FORMAT_CONTRACTS.get(stage, ("", None))
        attempts = FORMAT_ATTEMPTS if is_valid is not None else 1
        for attempt in range(1, attempts + 1):
            text = transport.complete(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stage=stage,
                structured=False,
                seed=getattr(_tls, "seed", None),
            )
            if is_string:
                return text
            try:
                parsed = _extract_json(text)
            except ValueError:
                counters.unparsed_completions += 1
                if is_valid is None:
                    return None
                counters.format_resamples += 1
                continue
            if is_valid is None or is_valid(parsed):
                return parsed
            counters.format_resamples += 1
        raise ReleasedDriverFailure(
            f"{stage} did not return {shape} in {attempts} samples; the "
            "released driver indexes this result without a guard."
        )

    def backend(input_dict: dict[str, Any]) -> dict[str, Any]:
        return _tls.backend(input_dict)

    draft.openai_response = openai_response
    draft.get_rapidapi_response = backend
    _redirect_open(draft)


def _mark_used(
    api_result: Any,
    tool_instruction: Any,
    tool_id: Any,
    api_used: list[str],
    tool_used: list[str],
    api_name_list: list[str],
) -> None:
    try:
        for ele in api_result:
            api_used.append(str(ele["api_name"]))
        remaining = []
        for ele in tool_instruction["tool_guidelines"].keys():
            if ele in api_name_list and ele not in api_used:
                remaining.append(ele)
        if len(remaining) == 0:
            tool_used.append(str(tool_id["ID"]))
    except Exception:
        return


def _run_react(
    draft: Any,
    query: dict[str, Any],
    dataset: dict[str, Any],
    retrieval_num: int,
    model_name: str,
    query_index: int,
) -> dict[str, Any]:
    question = query["query"]
    tool_dic = query["Tool_dic"]
    api_name_list = [api["api_name"] for api in query["api_list"]]
    tool_used: list[str] = []
    api_used: list[str] = []
    api_result_ls: list[Any] = []
    call_result_ls: list[Any] = []
    answer = ""
    for _ in range(retrieval_num):
        remaining = [
            str(ele) for ele in tool_dic if str(ele.get("ID")) not in tool_used
        ]
        if not remaining:
            break
        for ele in tool_dic:
            try:
                ele["Description"] = dataset[str(ele["ID"])]["tool_description"]
            except Exception:
                continue
        tool_id, api_result, call_result, tool_instruction, _api = draft.retrieval(
            question,
            tool_dic,
            dataset,
            query,
            api_name_list,
            api_used,
            tool_used,
            query_index,
            model_name,
            {},
        )
        call_result = str(call_result)
        answer = draft.answer_generation(question, call_result, model_name)
        if api_result:
            api_result_ls.append(api_result)
            call_result_ls.append(call_result)
        check = draft.answer_check(question, answer, model_name)
        if check == 1:
            break
        _mark_used(
            api_result, tool_instruction, tool_id, api_used, tool_used, api_name_list
        )
    return {
        "ID": query_index + 1,
        "question": question,
        "final_answer": answer,
        "execute_log": {
            "api_result_ls": api_result_ls,
            "parameter_ls": [],
            "call_result_ls": call_result_ls,
        },
    }


def _run_dfsdt(
    draft: Any,
    query: dict[str, Any],
    dataset: dict[str, Any],
    retrieval_num: int,
    model_name: str,
    method: str,
    query_index: int,
    jsonl_path: Path,
    progress_path: Path,
) -> dict[str, Any]:
    jsonl_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    _tls.jsonl_path = str(jsonl_path)
    draft.task_execution(
        "TMDB",
        ".",
        {},
        dataset,
        [query],
        str(progress_path),
        0,
        1,
        retrieval_num,
        query_index,
        model_name,
        method,
    )
    if not jsonl_path.exists():
        raise ReleasedDriverFailure("Inference_DFSDT wrote no record.")
    text = jsonl_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    value, _consumed = decoder.raw_decode(text.lstrip())
    if not isinstance(value, dict):
        raise ReleasedDriverFailure("Inference_DFSDT wrote a non-object record.")
    return value


def _job(payload: dict[str, Any]) -> dict[str, Any]:
    row_name = str(payload["row"])
    agent = str(payload["agent"])
    query_index = int(payload["query_index"])
    gold = list(payload["gold"])
    query = copy.deepcopy(payload["query"])
    dataset = copy.deepcopy(payload["dataset"])
    model = str(payload["model"])
    retrieval_num = int(payload["retrieval_num"])
    work = Path(payload["work"])
    work.mkdir(parents=True, exist_ok=True)

    client: OpenRouterClient = payload["client"]
    budget: CostBudget = payload["budget"]
    budget_lock: threading.Lock = payload["budget_lock"]
    tmdb = payload["http"]
    urls: dict[str, str] = payload["urls"]
    draft = payload["draft"]

    with budget_lock:
        budget.reserve_check()

    context = CallContext(condition=row_name, query_index=query_index)
    failures = FailureSink()
    usage_path = work / "usage.jsonl"
    transport = LoggedOpenRouter(
        client=client,
        model=model,
        role="agent",
        usage_path=usage_path,
        budget=budget,
        context=context,
        failures=failures,
        provider=payload.get("provider") or None,
    )
    backend = TmdbDfsdtBackend(tmdb, urls)
    counters = SimpleNamespace(unparsed_completions=0, format_resamples=0)
    _tls.transport = transport
    _tls.context = context
    _tls.backend = backend
    _tls.counters = counters
    _tls.seed = payload.get("seed")
    _tls.jsonl_path = None
    _tls.print_log = str(work / "console.log")

    record: dict[str, Any] | None = None
    error = ""
    kind = ""
    try:
        if agent == "react":
            record = _run_react(
                draft,
                query,
                dataset,
                retrieval_num,
                model_alias(model),
                query_index,
            )
        else:
            record = _run_dfsdt(
                draft,
                query,
                dataset,
                retrieval_num,
                model_alias(model),
                row_name,
                query_index,
                work / "record.jsonl",
                work / "progress.txt",
            )
        if failures.kind == "cost_cap":
            raise CostCapReached(failures.message)
        if failures.kind == PROVIDER_DRIFT:
            raise ProviderDrift(failures.message)
        if failures.pending:
            raise HarnessFailure(failures.message)
    except CostCapReached:
        raise
    except ProviderDrift as exc:
        error = f"{type(exc).__name__}: {exc}"
        kind = PROVIDER_DRIFT
    except (HarnessFailure, OpenRouterError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        kind = TRANSPORT
    except (ReleasedDriverFailure, TypeError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        kind = DRIVER_ERROR
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        kind = DRIVER_ERROR

    # The released driver swallows exceptions in bare `except:` blocks, so a
    # transport fault can surface as a driver-shaped error. The sink saw the
    # original cause first and decides the class.
    if error and failures.kind in TRANSPORT_ERROR_KINDS:
        kind = failures.kind

    if error:
        with (work / "console.log").open("a", encoding="utf-8") as handle:
            handle.write("\n" + error + "\n" + traceback.format_exc())
    executed = _executed_names(record) if record else []
    hit = bool(record) and not error and _score(record, gold)
    cost = 0.0
    if usage_path.exists():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cost += float(json.loads(line).get("cost_usd") or 0.0)
    row = {
        "row": row_name,
        "agent": agent,
        "query_index": query_index,
        "question": query.get("query"),
        "gold": gold,
        "executed": executed,
        "correct_path": hit,
        "cost_usd": round(cost, 6),
        "http_calls": backend.calls,
        "http_live": backend.live,
        "http_errors": backend.errors,
        "error": error,
        "error_kind": kind,
        "final_answer": (record or {}).get("final_answer") or "",
        "seed": payload.get("seed"),
        "provider": payload.get("provider") or "",
    }
    (work / "result.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return row


def select_queries(
    queries: list[dict[str, Any]],
    *,
    start: int = 0,
    limit: int = 100,
    indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Resolve the query slice and its true 0-based indices.

    The pilot panel is a fixed, non-contiguous subset, so the indices have to
    travel with the rows instead of being recomputed as `start + offset`;
    otherwise a panel run would label q033 as q002 in its traces.
    """
    if indices:
        chosen = list(dict.fromkeys(indices))
        out_of_range = [index for index in chosen if not 0 <= index < len(queries)]
        if out_of_range:
            raise SystemExit(f"query indices out of range: {out_of_range}")
        return [queries[index] for index in chosen], chosen
    selected = queries[start : start + limit]
    return selected, list(range(start, start + len(selected)))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, _agent, _docs in ROWS:
        subset = [row for row in rows if row.get("row") == name]
        if not subset:
            continue
        scored = [row for row in subset if not row.get("error")]
        hits = sum(1 for row in scored if row.get("correct_path"))
        kinds: dict[str, int] = {}
        for row in subset:
            kind = error_kind(row)
            if kind:
                kinds[kind] = kinds.get(kind, 0) + 1
        report[name] = {
            "n": len(subset),
            "ok": len(scored),
            "errors": len(subset) - len(scored),
            "errors_by_kind": kinds,
            "cp": round(hits / len(scored), 4) if scored else None,
            "cp_over_n": round(hits / len(subset), 4) if subset else None,
            "hits": hits,
            "http_live": sum(int(row.get("http_live") or 0) for row in subset),
            "cost_usd": round(sum(float(row.get("cost_usd") or 0.0) for row in subset), 6),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DRAFT Table 1 RestBench-TMDB CP reconstruction."
    )
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument(
        "--dataset",
        default="TMDB",
        choices=["TMDB", "Spotify"],
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--query-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit 0-based query indices, overriding --start/--limit. The "
        "pilot panel is a fixed subset, so the indices must appear in the "
        "manifest instead of being implied by an offset.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider",
        default=None,
        help="Pin one OpenRouter provider slug with fallbacks disabled. Two "
        "runs of the same model id served by different providers are not the "
        "same measurement, so a served provider that differs from this one "
        "fails the cell instead of being scored.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retrieval-num", type=int, default=5)
    parser.add_argument("--max-cost-usd", type=float, default=2.00)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        default=["DFSDT", "DRAFT", "Ours"],
        choices=["ReAct", "DFSDT", "DRAFT", "Ours"],
    )
    parser.add_argument(
        "--ours-docs",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep traces.jsonl and skip finished (row, query) pairs. "
        "A recorded error is rerun.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Rerun every (row, query) whose traces recorded an error. "
        "The released driver resamples a bad completion and continues; "
        "a leftover error is retried, not scored as a miss.",
    )
    parser.add_argument(
        "--retry-driver-errors",
        action="store_true",
        help="Deprecated alias: --retry-errors already re-rolls driver "
        "failures, matching Inference_DFSDT.",
    )
    args = parser.parse_args()
    dataset = args.dataset
    if args.output_root is None:
        args.output_root = Path(f"artifacts/results/restbench_{dataset.lower()}_table")
    if args.ours_docs is None:
        args.ours_docs = Path(f"artifacts/documentation/{dataset}_Ours.json")

    load_env_file()
    cache: dict[str, dict[str, Any]] = {}
    if dataset == "TMDB":
        tmdb_key = os.environ.get("TMDB_API_KEY", "")
        if not tmdb_key:
            raise SystemExit("TMDB_API_KEY is missing.")
        http = TmdbClient(tmdb_key, cache=cache)
    else:
        client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
        client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
        refresh = os.environ.get("SPOTIFY_REFRESH_TOKEN", "")
        if not client_id or not client_secret:
            raise SystemExit("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are missing.")
        if not refresh:
            raise SystemExit(
                "RestBench-Spotify gold uses /me on 55/57 queries. "
                "Client credentials cannot score this split. Set "
                "SPOTIFY_REFRESH_TOKEN from a user OAuth refresh grant."
            )
        http = SpotifyClient(
            client_id, client_secret, refresh_token=refresh, cache=cache
        )

    raw_queries, indices = select_queries(
        load_queries(args.draft_root, dataset),
        start=args.start,
        limit=args.limit,
        indices=args.query_indices,
    )
    selected = [row for row in ROWS if row[0] in args.rows]
    raw_docs = {
        "Initial": load_instructions(args.draft_root, dataset, "Initial"),
        "DRAFT": load_instructions(args.draft_root, dataset, "DRAFT"),
    }
    if any(name == "Ours" for name, _agent, _docs in selected):
        if args.ours_docs.exists():
            raw_docs["Ours"] = _load_json(args.ours_docs)
        else:
            ours, ours_report = build_ours_instructions(
                raw_docs["Initial"],
                client=http if dataset == "TMDB" else None,
                base_name="Initial",
            )
            args.ours_docs.parent.mkdir(parents=True, exist_ok=True)
            _dump_json(args.ours_docs, ours)
            _dump_json(
                args.ours_docs.with_name(f"{dataset}_Ours_report.json"), ours_report
            )
            raw_docs["Ours"] = ours
    docs = {
        name: wrap_instructions(value, category=dataset)
        for name, value in raw_docs.items()
    }
    urls = {name: url_index(value) for name, value in raw_docs.items()}
    wrapped_queries = [wrap_query(query) for query in raw_queries]

    run_root: Path = args.output_root
    run_root.mkdir(parents=True, exist_ok=True)
    traces_path = run_root / "traces.jsonl"
    existing: list[dict[str, Any]] = []
    if traces_path.exists():
        existing = [
            json.loads(line)
            for line in traces_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    retry_keys: set[tuple[str, int]] = set()
    skip_keys: set[tuple[str, int]] = set()
    if args.retry_errors:
        retry_keys = {
            (str(row.get("row")), int(row.get("query_index") or 0))
            for row in existing
            if row.get("error")
        }
        if not retry_keys:
            print("No retryable rows in traces.jsonl.", flush=True)
            return 0
    elif args.resume:
        skip_keys = {
            (str(row.get("row")), int(row.get("query_index") or 0))
            for row in existing
            if not row.get("error")
        }
    elif traces_path.exists():
        traces_path.unlink()
        existing = []

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": f"RestBench-{dataset}",
        "n": len(raw_queries),
        "start": args.start,
        "query_indices": indices,
        "model": args.model,
        "provider": args.provider or "",
        "workers": args.workers,
        "seed": args.seed,
        "retrieval_num": args.retrieval_num,
        "decoding": RELEASED_DECODING,
        "driver": f"external/DRAFT/Inference_DFSDT.py verbatim + {dataset} HTTP",
        "rows": {
            name: {"agent": agent, "docs": condition}
            for name, agent, condition in selected
        },
        "easytool": "skipped, RestBench EasyTool docs were not released",
        "docs": {
            name: {
                "path": str(
                    args.ours_docs
                    if name == "Ours"
                    else instruction_path(args.draft_root, dataset, name)
                ),
                "digest": file_digest(
                    args.ours_docs
                    if name == "Ours"
                    else instruction_path(args.draft_root, dataset, name)
                ),
                # file_digest changes with the line endings of the machine that
                # wrote the JSON; the payload digest is what identifies the
                # documentation across runs.
                "payload_digest": payload_digest(raw_docs[name]),
            }
            for name in raw_docs
        },
        "queries_digest": file_digest(test_path(args.draft_root, dataset)),
    }
    _dump_json(run_root / "manifest.json", manifest)

    client = OpenRouterClient.from_env()
    refuse_openrouter_for_local_model(args.model, client)
    print(
        f"transport={'local' if client.local else 'openrouter'}",
        flush=True,
    )
    budget = CostBudget(args.max_cost_usd)
    budget_lock = threading.Lock()
    draft = load_released_draft_module(args.draft_root)
    _install_driver(draft)

    jobs = []
    for offset, (raw, wrapped) in enumerate(zip(raw_queries, wrapped_queries)):
        index = indices[offset]
        gold = gold_apis(raw)
        for name, agent, condition in selected:
            if retry_keys and (name, index) not in retry_keys:
                continue
            if skip_keys and (name, index) in skip_keys:
                continue
            jobs.append(
                {
                    "row": name,
                    "agent": agent,
                    "query_index": index,
                    "gold": gold,
                    "query": wrapped,
                    "dataset": docs[condition],
                    "model": args.model,
                    "provider": args.provider,
                    "retrieval_num": args.retrieval_num,
                    "seed": args.seed,
                    "work": str(run_root / name.lower() / f"q{index:03d}"),
                    "client": client,
                    "budget": budget,
                    "budget_lock": budget_lock,
                    "http": http,
                    "urls": urls[condition],
                    "draft": draft,
                }
            )

    print(
        f"RestBench-{dataset} table {len(raw_queries)} queries x {len(selected)} rows, "
        f"{args.workers} workers, {args.model}, seed={args.seed}"
        + (f", retry {len(jobs)} failed jobs" if retry_keys else "")
        + (f", resume skip {len(skip_keys)}" if skip_keys else ""),
        flush=True,
    )
    live_trace = (
        run_root / "retry_traces.jsonl" if retry_keys else traces_path
    )
    if retry_keys and live_trace.exists():
        live_trace.unlink()
    rows: list[dict[str, Any]] = []
    consecutive_dead = 0
    try:
        if jobs:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_job, job) for job in jobs]
                done = 0
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    with _log_lock:
                        with live_trace.open("a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(row, ensure_ascii=False) + "\n"
                            )
                    done += 1
                    flag = "Y" if row.get("correct_path") else "n"
                    with _print_lock:
                        print(
                            f"{done}/{len(jobs)} {row.get('row')} "
                            f"q{row.get('query_index')} "
                            f"CP={flag} live={row.get('http_live')} "
                            f"${row.get('cost_usd')} "
                            f"{(row.get('error') or '')[:80]}",
                            flush=True,
                        )
                    if is_credits_exhausted(row.get("error")):
                        print(
                            "STOP credits-exhausted: OpenRouter HTTP 402",
                            flush=True,
                        )
                        for pending in futures:
                            pending.cancel()
                        return 2
                    if is_endpoint_dead(row.get("error")):
                        consecutive_dead += 1
                        if consecutive_dead >= max(int(args.workers), 3):
                            print(
                                "STOP endpoint unreachable: "
                                f"{consecutive_dead} connect failures in a row",
                                flush=True,
                            )
                            for pending in futures:
                                pending.cancel()
                            return 3
                    else:
                        consecutive_dead = 0
    except CostCapReached as exc:
        print(f"STOP cost-cap: {exc}", flush=True)

    retry_cost = round(budget.spent_usd, 6)
    if retry_keys or skip_keys:
        merged = {
            (str(row.get("row")), int(row.get("query_index") or 0)): row
            for row in existing
        }
        for row in rows:
            merged[(str(row.get("row")), int(row.get("query_index") or 0))] = row
        rows = list(merged.values())
        traces_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    rows.sort(key=lambda row: (str(row.get("row")), int(row.get("query_index") or 0)))
    previous_cost = 0.0
    old_summary = run_root / "summary.json"
    if (retry_keys or skip_keys) and old_summary.exists():
        previous_cost = float(
            json.loads(old_summary.read_text(encoding="utf-8")).get("cost_usd") or 0.0
        )
    summary = {
        "manifest": manifest,
        "conditions": summarize(rows),
        "cost_usd": round(
            previous_cost + retry_cost, 6
        ) if retry_keys or skip_keys else retry_cost,
        "retry_cost_usd": retry_cost if retry_keys else 0.0,
        "cap_usd": args.max_cost_usd,
        "cache_entries": len(cache),
        "jobs": len(jobs),
        "records": len(rows),
        "retried": sorted(list(retry_keys)) if retry_keys else [],
    }
    _dump_json(run_root / "summary.json", summary)
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=2))
    print(f"spent ${summary['cost_usd']} cache={summary['cache_entries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
