"""Run DRAFT's released DFSDT agent on ToolBench G3 with a pinned environment.

Design constraints that the first version of this harness violated:

- The released `Inference_DFSDT.py` is executed verbatim. An earlier version
  rewrote a block it described as an empty-candidate-list bug, but the released
  `retrieval` uses the parameter `api_list` and the local `API_list` as two
  different names, so that edit was a semantically empty rename.
- A failed model call never turns into a fabricated decision. Transport
  failures stop the run as infrastructure errors. Unparseable completions on
  stages without a format contract reproduce the released `openai_response`
  behaviour of returning `None`. Stages whose own prompt requires a JSON
  shape are resampled the same way a shape violation is; after those attempts
  the query is excluded rather than passed as `None` into an unguarded index.
  No stub record is ever written, so nothing enters the denominator that the
  agent did not actually produce.
- The evaluation agent and the virtual API simulator are separate transports.
  The simulator stays pinned to one model with fixed decoding, so swapping the
  agent does not also swap the environment.
- Raw and DRAFT are interleaved per query in a seeded random order, so neither
  condition systematically populates the shared simulator cache first.

The released driver wraps large blocks in bare `except:`, which also catches
our own control-flow exceptions. Infrastructure failures are therefore recorded
in a sink and re-checked after every query; a record written while a failure was
pending is rolled back rather than counted.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import threading
import time
import traceback
import types
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import requests

from tooldoc_nir.documentation_store import (
    parse_extra_documentation,
    resolve_documentation_path,
)
from tooldoc_nir.openrouter import OpenRouterClient, OpenRouterError, load_env_file
from tooldoc_nir.provenance import (
    ManifestMismatch,
    build_manifest as build_run_manifest,
    file_digest,
    payload_digest,
    reconcile_manifest,
)

DEFAULT_DRAFT_ROOT = Path("external/DRAFT")
DEFAULT_STABLE_ROOT = Path("data/raw/stabletoolbench/cache/extracted")
DEFAULT_AGENT_MODEL = "openai/gpt-4o-mini-2024-07-18"
DEFAULT_SIMULATOR_MODEL = "openai/gpt-4o-mini-2024-07-18"
CONDITIONS = ("Initial", "DRAFT")

# The released driver hard-codes these at module level. They are asserted
# rather than assumed so a silent upstream change cannot alter a measurement.
RELEASED_DECODING = {"temperature": 0.2, "top_p": 1, "max_tokens": 2000}

# The virtual API server must not drift when the agent model changes.
SIMULATOR_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 768,
    "seed": 42,
}

SIMULATOR_SYSTEM_PROMPT = (
    "You are the deterministic virtual API server used for a tool-learning "
    "benchmark. Return a JSON object with exactly two fields: error (empty "
    "string on a valid call) and response. The response must be concrete, "
    "useful, consistent with the API documentation and input, and contain "
    "plausible identifiers or values needed by later API calls. Do not explain "
    "the simulation."
)

COST_CAP = "cost_cap"
TRANSPORT = "transport"
CACHE_MISS = "cache_miss"
SCHEMA_GATE = "schema_gate"
LIVE = "live"
LIVE_CACHE = "live_cache"
LIVE_HTTP_TIMEOUT = 30.0
LIVE_HTTP_ATTEMPTS = 3

RESERVED_PARAMETER_NAMES = frozenset(
    {"from", "class", "return", "false", "true", "id", "and", "", "ID"}
)

# The released driver indexes these two completions without any guard, so a
# shape violation crashes the whole query. Both prompts state the required
# shape verbatim, so resampling enforces the released contract rather than
# substituting a decision of ours.
FORMAT_ATTEMPTS = 3

# Two exclusions in the first dozen queries is noise, not a harness fault.
# Enforce the rate only after this many queries have actually been touched.
MIN_DRIVER_FAILURE_SAMPLE = 20


def _is_task_decomposition(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("Tasks"), list)
        and bool(value["Tasks"])
    )


def _is_task_topology(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, dict) and "dep" in item and "task" in item
            for item in value
        )
    )


FORMAT_CONTRACTS = {
    "task_decompose": ('{"Tasks": [...]}', _is_task_decomposition),
    "task_topology": (
        '[{"task": ..., "id": ..., "dep": [...]}]',
        _is_task_topology,
    ),
}


class HarnessFailure(RuntimeError):
    """Infrastructure failure. The run stops instead of degrading results."""


class CostCapReached(HarnessFailure):
    """The configured OpenRouter spend limit was reached."""


class SimulatorCacheMiss(HarnessFailure):
    """A replay run needed a response that is not in the frozen cache."""


class ReleasedDriverFailure(RuntimeError):
    """The released agent could not complete this query.

    Not an infrastructure fault and not a result: the query is dropped from
    both conditions so the comparison stays paired, and the reason is reported.
    """


class FailureSink:
    """Survives the released driver's bare `except:` blocks."""

    def __init__(self) -> None:
        self.kind: str | None = None
        self.message: str = ""

    def record(self, kind: str, message: str) -> None:
        if self.kind is None:
            self.kind = kind
            self.message = message

    def clear(self) -> None:
        self.kind = None
        self.message = ""

    @property
    def pending(self) -> bool:
        return self.kind is not None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


_JSONL_LOCK = threading.Lock()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _JSONL_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            # A killed process can leave a partial trailing line.
            break
    return records


def model_alias(model: str) -> str:
    """Filename-safe short name, matching the released output convention."""
    return model.split("/")[-1]


def _extract_json(text: str) -> Any:
    candidate = text.strip()
    candidate = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        candidate,
        count=1,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\s*```\s*$", "", candidate, count=1)
    decoder = json.JSONDecoder()
    for start in range(len(candidate)):
        if candidate[start] not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Completion does not contain JSON: {text[:500]!r}")


def load_released_draft_module(root: Path) -> types.ModuleType:
    """Execute the released driver verbatim, with no behavioural patch."""
    source_path = root / "Inference_DFSDT.py"
    source = source_path.read_text(encoding="utf-8")
    module = types.ModuleType("draft_released_inference")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    module.__dict__["__released_digest__"] = file_digest(source_path)
    assert_released_decoding(module)
    return module


def assert_released_decoding(module: types.ModuleType) -> None:
    observed = {key: module.__dict__.get(key) for key in RELEASED_DECODING}
    if observed != RELEASED_DECODING:
        raise HarnessFailure(
            "The released driver no longer declares the published decoding "
            f"parameters. Expected {RELEASED_DECODING}, found {observed}."
        )


class CostBudget:
    def __init__(self, maximum_usd: float, *, spent_usd: float = 0.0) -> None:
        self.maximum_usd = maximum_usd
        self.spent_usd = spent_usd
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent_usd >= self.maximum_usd

    def reserve_check(self) -> None:
        with self._lock:
            if self.spent_usd >= self.maximum_usd:
                raise CostCapReached(
                    f"OpenRouter cost cap reached: ${self.spent_usd:.6f} "
                    f"of ${self.maximum_usd:.2f}."
                )

    def record(self, cost_usd: float) -> None:
        with self._lock:
            self.spent_usd += cost_usd


@dataclass
class CallContext:
    condition: str = "-"
    query_index: int = -1

    def as_dict(self) -> dict[str, Any]:
        return {"condition": self.condition, "query_index": self.query_index}


class LoggedOpenRouter:
    """One transport role with its own usage log and per-call metadata."""

    def __init__(
        self,
        *,
        client: OpenRouterClient,
        model: str,
        role: str,
        usage_path: Path,
        budget: CostBudget,
        context: CallContext,
        failures: FailureSink,
        provider: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.role = role
        self.usage_path = usage_path
        self.budget = budget
        self.context = context
        self.failures = failures
        self.provider = provider
        self.request_count = 0

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stage: str,
        structured: bool,
        seed: int | None,
    ) -> str:
        try:
            self.budget.reserve_check()
        except CostCapReached as exc:
            self.failures.record(COST_CAP, str(exc))
            raise
        started = time.perf_counter()
        try:
            result = self.client.chat_completion(
                model=self.model,
                messages=messages,
                provider=self.provider,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                reasoning_effort=None,
                response_format={"type": "json_object"} if structured else None,
            )
        except OpenRouterError as exc:
            message = f"{self.role} transport failed during {stage}: {exc}"
            self.failures.record(TRANSPORT, message)
            raise HarnessFailure(message) from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        choice = result["choices"][0] or {}
        content = str((choice.get("message") or {}).get("content") or "")
        usage = result.get("usage") or {}
        cost_usd = float(usage.get("cost") or 0.0)
        self.budget.record(cost_usd)
        self.request_count += 1
        transport = result.get("_transport") or {}
        _append_jsonl(
            self.usage_path,
            {
                "role": self.role,
                **self.context.as_dict(),
                "stage": stage,
                "requested_model": self.model,
                "served_model": result.get("model", ""),
                "provider": result.get("provider", ""),
                "request_id": transport.get("request_id", ""),
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": (
                    usage.get("completion_tokens_details") or {}
                ).get("reasoning_tokens", 0),
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "finish_reason": choice.get("finish_reason", ""),
            },
        )
        return content


def usage_totals(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in records)
    return {
        "requests": len(records),
        "prompt_tokens": sum(
            int(row.get("prompt_tokens") or 0) for row in records
        ),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in records
        ),
        "reasoning_tokens": sum(
            int(row.get("reasoning_tokens") or 0) for row in records
        ),
        "cost_usd": round(
            sum(float(row.get("cost_usd") or 0.0) for row in records), 6
        ),
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "served_models": sorted(
            {str(row.get("served_model") or "") for row in records} - {""}
        ),
        "providers": sorted(
            {str(row.get("provider") or "") for row in records} - {""}
        ),
    }


def run_cost(run_root: Path) -> float:
    total = 0.0
    for name in ("usage_agent.jsonl", "usage_simulator.jsonl"):
        total += sum(
            float(row.get("cost_usd") or 0.0)
            for row in read_jsonl(run_root / name)
        )
    return round(total, 6)


class StableExecutionAdapter:
    """Virtual API server: frozen cache first, pinned simulator on a miss.

    Live RapidAPI mode skips the 2024 cache and simulator and calls the
    subscribed hosts. Identical requests are cached inside the run so paired
    conditions share one HTTP environment.
    """

    def __init__(
        self,
        *,
        root: Path,
        transport: LoggedOpenRouter | None,
        generated_cache_path: Path,
        call_log_path: Path,
        context: CallContext,
        failures: FailureSink,
        replay_only: bool = False,
        schema_gate: bool = False,
        live_api_key: str = "",
        live_session: requests.Session | None = None,
    ) -> None:
        self.root = root
        self.tools_root = root / "tools"
        self.response_cache_root = root / "tool_response_cache"
        self.transport = transport
        self.generated_cache_path = generated_cache_path
        self.call_log_path = call_log_path
        self.context = context
        self.failures = failures
        self.replay_only = replay_only
        self.schema_gate = schema_gate
        self.live_api_key = live_api_key
        self.live_session = live_session
        self.generated_cache: dict[str, dict[str, Any]] = (
            _load_json(generated_cache_path)
            if generated_cache_path.exists()
            else {}
        )
        self.stats = {
            "exact_cache": 0,
            "generated_cache": 0,
            "simulated": 0,
            "missing_document": 0,
            "unparseable_simulation": 0,
            "schema_rejected": 0,
            "live": 0,
            "live_cache": 0,
            "live_http_error": 0,
        }

    @staticmethod
    def _standardize(value: str) -> str:
        normalized = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value)
        normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
        if normalized and normalized[0].isdigit():
            normalized = "get_" + normalized
        if normalized in {
            "from",
            "class",
            "return",
            "false",
            "true",
            "id",
            "and",
            "",
        }:
            normalized = "is_" + normalized
        return normalized

    @staticmethod
    def _category(value: str) -> str:
        normalized = value.replace(" ", "_").replace(",", "_").replace("/", "_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized

    def _document(
        self,
        *,
        category: str,
        tool_name: str,
        api_name: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        path = self.tools_root / category / f"{tool_name}.json"
        if not path.exists():
            return None, None
        tool_document = _load_json(path)
        for api in tool_document.get("api_list") or []:
            if self._standardize(str(api.get("name") or "")) == api_name:
                return tool_document, api
        return tool_document, None

    @staticmethod
    def _change_parameter_name(name: str) -> str:
        if name in RESERVED_PARAMETER_NAMES:
            return "is_" + name.lower()
        return name

    def _schema_violation(
        self, api: dict[str, Any] | None, parameters: Mapping[str, Any]
    ) -> str | None:
        if not self.schema_gate or api is None:
            return None
        provided = {
            self._change_parameter_name(str(key)) for key in parameters
        }
        required = {
            self._change_parameter_name(str(item.get("name") or ""))
            for item in api.get("required_parameters") or []
            if isinstance(item, dict)
        }
        optional = {
            self._change_parameter_name(str(item.get("name") or ""))
            for item in api.get("optional_parameters") or []
            if isinstance(item, dict)
        }
        required.discard("")
        optional.discard("")
        provided.discard("")
        missing = sorted(required - provided)
        unknown = sorted(provided - required - optional)
        parts: list[str] = []
        if missing:
            parts.append("missing required parameters: " + ", ".join(missing))
        if unknown:
            parts.append("unknown parameters: " + ", ".join(unknown))
        return "; ".join(parts) if parts else None

    def _outgoing_parameters(
        self, api: dict[str, Any], parameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        catalog: dict[str, str] = {}
        fields = list(api.get("required_parameters") or []) + list(
            api.get("optional_parameters") or []
        )
        for item in fields:
            if not isinstance(item, dict):
                continue
            original = str(item.get("name") or "")
            if not original:
                continue
            catalog[original] = original
            catalog[self._change_parameter_name(original)] = original
            catalog[self._standardize(original)] = original
        outgoing: dict[str, Any] = {}
        for key, value in parameters.items():
            outgoing[catalog.get(str(key), str(key))] = value
        return outgoing

    def _live_http(
        self,
        *,
        tool_document: dict[str, Any],
        api_document: dict[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[dict[str, str], int]:
        url = str(api_document.get("url") or "")
        host = str(tool_document.get("host") or "")
        if not host and url:
            host = urlparse(url).netloc
        method = str(api_document.get("method") or "GET").upper()
        if not url:
            return {"error": "ToolEnv document has no url", "response": ""}, 0
        headers = {
            "X-RapidAPI-Key": self.live_api_key,
            "X-RapidAPI-Host": host,
            "Accept": "application/json",
        }
        outgoing = self._outgoing_parameters(api_document, parameters)
        session = self.live_session or requests.Session()
        last_error = ""
        status = 0
        for attempt in range(LIVE_HTTP_ATTEMPTS):
            try:
                kwargs: dict[str, Any] = {
                    "headers": headers,
                    "timeout": LIVE_HTTP_TIMEOUT,
                }
                if method in {"POST", "PUT", "PATCH"}:
                    kwargs["json"] = outgoing
                else:
                    kwargs["params"] = outgoing
                response = session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_error = f"transport: {exc}"
                time.sleep(2 * (attempt + 1))
                continue
            status = int(response.status_code)
            body = (response.text or "")[:2048]
            if status == 429 and attempt + 1 < LIVE_HTTP_ATTEMPTS:
                time.sleep(2 * (attempt + 1))
                continue
            if 200 <= status < 300:
                return {"error": "", "response": body}, status
            return {
                "error": f"HTTP {status}: {body[:500]}",
                "response": "",
            }, status
        return {
            "error": last_error or "live RapidAPI request failed",
            "response": "",
        }, status

    def _cached_response(
        self,
        *,
        category: str,
        tool_name: str,
        api_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any] | None:
        path = (
            self.response_cache_root
            / category
            / f"{tool_name}_for_{category}"
            / f"{api_name}.json"
        )
        if not path.exists():
            return None
        cache = _load_json(path)
        if not isinstance(cache, dict):
            return None
        value = cache.get(str(parameters))
        return value if isinstance(value, dict) else None

    def _log_call(
        self,
        *,
        source: str,
        category: str,
        tool_name: str,
        api_name: str,
        parameters: Mapping[str, Any],
        response: Mapping[str, Any],
        http_status: int | None = None,
    ) -> None:
        row: dict[str, Any] = {
            **self.context.as_dict(),
            "source": source,
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "parameters": dict(parameters),
            "error": str(response.get("error") or ""),
            "response_chars": len(str(response.get("response") or "")),
        }
        if http_status is not None:
            row["http_status"] = http_status
        _append_jsonl(self.call_log_path, row)

    def _request_cache_key(
        self,
        *,
        category: str,
        tool_name: str,
        api_name: str,
        parameters: Mapping[str, Any],
    ) -> str:
        request = {
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "parameters": dict(parameters),
        }
        return hashlib.sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()

    def _serve_live(
        self,
        *,
        category: str,
        tool_name: str,
        api_name: str,
        parameters: dict[str, Any],
        tool_document: dict[str, Any] | None,
        api_document: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cache_key = self._request_cache_key(
            category=category,
            tool_name=tool_name,
            api_name=api_name,
            parameters=parameters,
        )
        if cache_key in self.generated_cache:
            self.stats["live_cache"] += 1
            response = self.generated_cache[cache_key]
            self._log_call(
                source=LIVE_CACHE,
                category=category,
                tool_name=tool_name,
                api_name=api_name,
                parameters=parameters,
                response=response,
            )
            return response
        if tool_document is None or api_document is None:
            self.stats["missing_document"] += 1
            response = {
                "error": f"no ToolEnv document for {tool_name}.{api_name}",
                "response": "",
            }
            self._log_call(
                source=LIVE,
                category=category,
                tool_name=tool_name,
                api_name=api_name,
                parameters=parameters,
                response=response,
                http_status=0,
            )
            return response
        response, status = self._live_http(
            tool_document=tool_document,
            api_document=api_document,
            parameters=parameters,
        )
        if response["error"]:
            self.stats["live_http_error"] += 1
        self.generated_cache[cache_key] = response
        _dump_json(self.generated_cache_path, self.generated_cache)
        self.stats["live"] += 1
        self._log_call(
            source=LIVE,
            category=category,
            tool_name=tool_name,
            api_name=api_name,
            parameters=parameters,
            response=response,
            http_status=status,
        )
        return response

    def __call__(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        category = self._category(str(input_dict["category"]))
        tool_name = self._standardize(str(input_dict["tool_name"]))
        api_name = self._standardize(str(input_dict["api_name"]))
        raw_parameters = input_dict.get("tool_input") or {}
        if isinstance(raw_parameters, str):
            parameters = json.loads(raw_parameters) if raw_parameters else {}
        else:
            parameters = dict(raw_parameters)

        tool_document, api_document = self._document(
            category=category,
            tool_name=tool_name,
            api_name=api_name,
        )
        violation = self._schema_violation(api_document, parameters)
        if violation:
            self.stats["schema_rejected"] += 1
            response = {"error": violation, "response": ""}
            self._log_call(
                source=SCHEMA_GATE,
                category=category,
                tool_name=tool_name,
                api_name=api_name,
                parameters=parameters,
                response=response,
            )
            return response

        if self.live_api_key:
            return self._serve_live(
                category=category,
                tool_name=tool_name,
                api_name=api_name,
                parameters=parameters,
                tool_document=tool_document,
                api_document=api_document,
            )

        exact = self._cached_response(
            category=category,
            tool_name=tool_name,
            api_name=api_name,
            parameters=parameters,
        )
        if exact is not None:
            self.stats["exact_cache"] += 1
            response = {
                "error": str(exact.get("error") or ""),
                "response": str(exact.get("response") or "")[:2048],
            }
            self._log_call(
                source="exact_cache",
                category=category,
                tool_name=tool_name,
                api_name=api_name,
                parameters=parameters,
                response=response,
            )
            return response

        request = {
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "parameters": parameters,
        }
        cache_key = hashlib.sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        if cache_key in self.generated_cache:
            self.stats["generated_cache"] += 1
            response = self.generated_cache[cache_key]
            self._log_call(
                source="generated_cache",
                category=category,
                tool_name=tool_name,
                api_name=api_name,
                parameters=parameters,
                response=response,
            )
            return response

        if self.replay_only or self.transport is None:
            message = (
                "Replay mode requires every response to come from the frozen "
                f"cache, but {tool_name}.{api_name} with {parameters} is absent."
            )
            self.failures.record(CACHE_MISS, message)
            raise SimulatorCacheMiss(message)

        if tool_document is None or api_document is None:
            self.stats["missing_document"] += 1
        api_context = {
            "tool_name": tool_name,
            "tool_description": (tool_document or {}).get(
                "tool_description", ""
            ),
            "api": api_document or {"name": api_name},
        }
        messages = [
            {"role": "system", "content": SIMULATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "API documentation:\n"
                    f"{json.dumps(api_context, ensure_ascii=False)}"
                    f"\nRequest:\n{json.dumps(request, ensure_ascii=False)}"
                ),
            },
        ]
        text = self.transport.complete(
            messages=messages,
            temperature=float(SIMULATOR_DECODING["temperature"]),
            top_p=float(SIMULATOR_DECODING["top_p"]),
            max_tokens=int(SIMULATOR_DECODING["max_tokens"]),
            stage="stable_api_simulation",
            structured=True,
            seed=int(SIMULATOR_DECODING["seed"]),
        )
        try:
            parsed = _extract_json(text)
        except ValueError:
            parsed = {"error": "", "response": text}
            self.stats["unparseable_simulation"] += 1
        if not isinstance(parsed, dict):
            parsed = {"error": "", "response": str(parsed)}
            self.stats["unparseable_simulation"] += 1
        response = {
            "error": str(parsed.get("error") or ""),
            "response": str(parsed.get("response") or "")[:2048],
        }
        self.generated_cache[cache_key] = response
        _dump_json(self.generated_cache_path, self.generated_cache)
        self.stats["simulated"] += 1
        self._log_call(
            source="simulated",
            category=category,
            tool_name=tool_name,
            api_name=api_name,
            parameters=parameters,
            response=response,
        )
        return response


def record_offsets(path: Path) -> list[int]:
    """Byte-free character offsets where each released record starts."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    offsets: list[int] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            _, consumed = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            break
        offsets.append(offset)
        offset += consumed
    return offsets


def truncate_records(path: Path, keep: int) -> None:
    """Roll back records written while an infrastructure failure was pending."""
    offsets = record_offsets(path)
    if keep >= len(offsets):
        return
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: offsets[keep]].rstrip() + "\n", encoding="utf-8")


def _read_concatenated_json(path: Path) -> list[dict[str, Any]]:
    """Read the released pretty-printed, newline-separated JSON output."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            value, consumed = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            raise ValueError(f"Unexpected non-object in {path}.")
        records.append(value)
        offset += consumed
    return records


def record_positions(path: Path) -> dict[int, dict[str, Any]]:
    """Index records by the query position the released driver stamped on them.

    Positional alignment breaks as soon as a query is dropped, so records are
    keyed by `ID - 1` instead of by line order.
    """
    positions: dict[int, dict[str, Any]] = {}
    for order, record in enumerate(_read_concatenated_json(path)):
        try:
            position = int(record["ID"]) - 1
        except (KeyError, TypeError, ValueError) as exc:
            raise HarnessFailure(
                f"Record {order} in {path} has no usable ID field."
            ) from exc
        if position in positions:
            raise HarnessFailure(
                f"{path} holds two records for query position {position}."
            )
        positions[position] = record
    return positions


def _process_name(value: str) -> str:
    return StableExecutionAdapter._standardize(value)


def _executed_path(record: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (
            _process_name(str(api["tool_name"])),
            _process_name(str(api["api_name"])),
        )
        for group in record["execute_log"]["api_result_ls"]
        for api in group
    ]


def gold_path(query: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (_process_name(str(tool)), _process_name(str(api)))
        for tool, api in query["relevant APIs"]
    ]


def correct_path_released(
    record: dict[str, Any],
    query: dict[str, Any],
) -> bool:
    """Released evaluator: unordered set containment."""
    return set(gold_path(query)).issubset(set(_executed_path(record)))


def correct_path(record: dict[str, Any], query: dict[str, Any]) -> bool:
    """Paper definition: the gold path occurs as an ordered subsequence."""
    executed = iter(_executed_path(record))
    return all(
        any(candidate == target for candidate in executed)
        for target in gold_path(query)
    )


@dataclass
class ConditionState:
    name: str
    dataset: dict[str, Any]
    root: Path
    output_path: Path
    progress_path: Path
    log_path: Path
    failures_path: Path
    unparsed_completions: int = 0
    format_resamples: int = 0
    query_attempts: dict[int, int] = field(default_factory=dict)
    driver_failures: dict[int, str] = field(default_factory=dict)
    rolled_back: int = 0

    def completed(self) -> int:
        return len(_read_concatenated_json(self.output_path))

    def positions(self) -> dict[int, dict[str, Any]]:
        return record_positions(self.output_path)

    def done(self, position: int) -> bool:
        return position in self.positions() or position in self.driver_failures

    def load_driver_failures(self) -> None:
        if not self.failures_path.exists():
            return
        recorded = _load_json(self.failures_path)
        self.driver_failures = {
            int(position): str(reason) for position, reason in recorded.items()
        }

    def record_driver_failure(self, position: int, reason: str) -> None:
        """Persist the exclusion so a resume cannot re-roll the same query."""
        self.driver_failures[position] = reason
        _dump_json(
            self.failures_path,
            {str(key): value for key, value in sorted(self.driver_failures.items())},
        )


def build_query_plan(
    *,
    queries: list[dict[str, Any]],
    query_indices: list[int],
    conditions: Iterable[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Pin the query set and a seeded per-query condition order."""
    order_rng = random.Random(f"condition-order:{seed}")
    plan: list[dict[str, Any]] = []
    for position, (index, query) in enumerate(
        zip(query_indices, queries, strict=True)
    ):
        order = list(conditions)
        order_rng.shuffle(order)
        plan.append(
            {
                "position": position,
                "query_index": index,
                "query_id": query.get("query_id"),
                "question_digest": payload_digest(query.get("query", "")),
                "gold_path": gold_path(query),
                "condition_order": order,
            }
        )
    return plan


def _make_agent_response(
    *,
    transport: LoggedOpenRouter,
    seed: int | None,
    states: Mapping[str, ConditionState],
    context: CallContext,
):
    """Replace the released `openai_response` without inventing decisions."""

    def openrouter_response(
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        model_name: str,
        is_string: bool,
    ) -> Any:
        del model_name
        stage = inspect.stack()[1].function
        state = states.get(context.condition)
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
                seed=None if attempt > 1 else seed,
            )
            if is_string:
                return text
            try:
                parsed = _extract_json(text)
            except ValueError:
                # Stages without a format contract match the released
                # `openai_response`: unparseable text becomes None. Stages
                # that index `['Tasks']` without a guard must not receive
                # None after a single bad sample; resample instead of
                # inventing a stub.
                if state is not None:
                    state.unparsed_completions += 1
                if is_valid is None:
                    print(
                        f"{stage} returned an unparseable completion; "
                        "passing None"
                    )
                    return None
                if state is not None:
                    state.format_resamples += 1
                print(
                    f"{stage} returned an unparseable completion; "
                    f"resampling {attempt}/{attempts}"
                )
                continue
            if is_valid is None or is_valid(parsed):
                return parsed
            if state is not None:
                state.format_resamples += 1
            print(
                f"{stage} violated the shape its own prompt requires ({shape}); "
                f"resampling {attempt}/{attempts}"
            )
        raise ReleasedDriverFailure(
            f"{stage} did not return {shape} in {attempts} samples; the "
            "released driver indexes this result without a guard."
        )

    return openrouter_response


def run_paired_g3(
    *,
    draft_root: Path,
    stable_root: Path,
    run_root: Path,
    conditions: list[str],
    agent_model: str,
    simulator_model: str,
    agent_provider: str | None,
    query_indices: list[int],
    retrieval_num: int,
    seed: int,
    budget: CostBudget,
    resume: bool,
    replay_only: bool,
    query_attempts: int,
    max_driver_failure_rate: float = 0.15,
    extra_documentation: dict[str, Path] | None = None,
    schema_gate: bool = False,
    live_rapidapi: bool = False,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    alias = model_alias(agent_model)
    all_queries = _load_json(
        draft_root / "dataset" / "ToolBench" / "test_data" / "G3.json"
    )
    queries = [all_queries[index] for index in query_indices]
    plan = build_query_plan(
        queries=queries,
        query_indices=query_indices,
        conditions=conditions,
        seed=seed,
    )

    documentation_index: dict[str, dict[str, str]] = {}
    for condition in conditions:
        path = resolve_documentation_path(
            draft_root=draft_root,
            condition=condition,
            extra=extra_documentation,
        )
        if not path.exists():
            raise HarnessFailure(
                f"No tool documentation for {condition} at {path}."
            )
        documentation_index[condition] = {
            "path": str(path),
            "digest": file_digest(path),
        }

    manifest = build_run_manifest(
        draft_root=draft_root,
        stable_root=stable_root,
        agent={
            "model": agent_model,
            "provider": agent_provider or "openrouter-default",
            "alias": alias,
            "seed": seed,
        },
        simulator={
            "model": "none" if live_rapidapi else simulator_model,
            "decoding": {} if live_rapidapi else SIMULATOR_DECODING,
            "system_prompt_digest": (
                ""
                if live_rapidapi
                else payload_digest(SIMULATOR_SYSTEM_PROMPT)
            ),
            "replay_only": True if live_rapidapi else replay_only,
            "live_rapidapi": live_rapidapi,
        },
        decoding=RELEASED_DECODING,
        protocol={
            "dataset": "ToolBench G3",
            "conditions": list(conditions),
            "retrieval_num": retrieval_num,
            "released_driver": "Inference_DFSDT.py executed verbatim",
            "condition_order": "seeded per-query randomization",
            "fallback_policy": "no fabricated decisions, no stub records",
            "schema_gate": schema_gate,
            "backend": "rapidapi_live" if live_rapidapi else "stabletoolbench",
            "documentation": documentation_index,
        },
        queries={"count": len(plan), "plan_digest": payload_digest(plan)},
    )
    manifest = reconcile_manifest(
        run_root=run_root, manifest=manifest, resume=resume
    )
    _dump_json(run_root / "queries.json", plan)

    agent_usage = run_root / "usage_agent.jsonl"
    simulator_usage = run_root / "usage_simulator.jsonl"
    backend_calls = run_root / "backend_calls.jsonl"
    if not resume:
        for path in (agent_usage, simulator_usage, backend_calls):
            path.unlink(missing_ok=True)

    context = CallContext()
    failures = FailureSink()
    client = OpenRouterClient.from_env()
    live_api_key = ""
    live_session: requests.Session | None = None
    if live_rapidapi:
        load_env_file()
        live_api_key = os.environ.get("RAPIDAPI_KEY", "")
        if not live_api_key:
            raise HarnessFailure(
                "RAPIDAPI_KEY is missing; live RapidAPI mode cannot start."
            )
        live_session = requests.Session()
    agent_transport = LoggedOpenRouter(
        client=client,
        model=agent_model,
        role="agent",
        usage_path=agent_usage,
        budget=budget,
        context=context,
        failures=failures,
        provider=agent_provider,
    )
    simulator_transport: LoggedOpenRouter | None = None
    if not live_rapidapi and not replay_only:
        simulator_transport = LoggedOpenRouter(
            client=client,
            model=simulator_model,
            role="simulator",
            usage_path=simulator_usage,
            budget=budget,
            context=context,
            failures=failures,
        )
    backend = StableExecutionAdapter(
        root=stable_root,
        transport=simulator_transport,
        generated_cache_path=run_root
        / (
            "live_response_cache.json"
            if live_rapidapi
            else "stable_simulator_cache.json"
        ),
        call_log_path=backend_calls,
        context=context,
        failures=failures,
        replay_only=False if live_rapidapi else replay_only,
        schema_gate=schema_gate,
        live_api_key=live_api_key,
        live_session=live_session,
    )

    states: dict[str, ConditionState] = {}
    for condition in conditions:
        condition_root = run_root / condition.lower()
        condition_root.mkdir(parents=True, exist_ok=True)
        output_path = (
            condition_root / f"ToolBench_G3_DFS_{alias}_{condition}.jsonl"
        )
        if not resume:
            output_path.unlink(missing_ok=True)
            (condition_root / "progress.txt").unlink(missing_ok=True)
            (condition_root / "console.log").unlink(missing_ok=True)
        failures_path = condition_root / "driver_failures.json"
        if not resume:
            failures_path.unlink(missing_ok=True)
        document_path = Path(documentation_index[condition]["path"])
        state = ConditionState(
            name=condition,
            dataset=_load_json(document_path),
            root=condition_root,
            output_path=output_path,
            progress_path=condition_root / "progress.txt",
            log_path=condition_root / "console.log",
            failures_path=failures_path,
        )
        state.load_driver_failures()
        if state.completed() > len(plan):
            raise HarnessFailure(
                f"{output_path} holds {state.completed()} records for "
                f"{len(plan)} planned queries. Use a fresh output root."
            )
        states[condition] = state

    draft = load_released_draft_module(draft_root)
    draft.openai_response = _make_agent_response(
        transport=agent_transport,
        seed=seed,
        states=states,
        context=context,
    )
    draft.get_rapidapi_response = backend

    previous_cwd = Path.cwd()
    stop_reason = "completed"
    try:
        for entry in plan:
            position = int(entry["position"])
            query = queries[position]
            for condition in entry["condition_order"]:
                state = states[condition]
                if state.done(position):
                    continue
                if budget.exhausted:
                    raise CostCapReached(
                        f"Cost cap ${budget.maximum_usd:.2f} reached before "
                        f"{condition} query {position}; spent "
                        f"${budget.spent_usd:.6f}."
                    )
                context.condition = condition
                context.query_index = int(entry["query_index"])
                _execute_one(
                    draft=draft,
                    state=state,
                    query=query,
                    position=position,
                    total=len(plan),
                    stable_root=stable_root,
                    retrieval_num=retrieval_num,
                    alias=alias,
                    attempts=query_attempts,
                    previous_cwd=previous_cwd,
                    failures=failures,
                )
            _assert_driver_failures_within_tolerance(
                states=states,
                planned=len(plan),
                attempted=position + 1,
                tolerance=max_driver_failure_rate,
            )
    except CostCapReached as exc:
        stop_reason = f"cost_cap: {exc}"
        raise
    except HarnessFailure as exc:
        stop_reason = f"infrastructure: {exc}"
        raise
    finally:
        os.chdir(previous_cwd)
        _summarize(
            run_root=run_root,
            manifest=manifest,
            states=states,
            plan=plan,
            queries=queries,
            backend=backend,
            agent_usage=agent_usage,
            simulator_usage=simulator_usage,
            budget=budget,
            stop_reason=stop_reason,
        )

    return _load_json(run_root / "summary.json")


def _execute_one(
    *,
    draft: types.ModuleType,
    state: ConditionState,
    query: dict[str, Any],
    position: int,
    total: int,
    stable_root: Path,
    retrieval_num: int,
    alias: str,
    attempts: int,
    previous_cwd: Path,
    failures: FailureSink,
) -> None:
    """Run one (query, condition) pair transactionally.

    The released driver only writes its record after the query finishes, so a
    crash leaves no partial row. If a record does land while an infrastructure
    failure is pending, it is rolled back instead of being scored.
    """
    last_error = ""
    driver_failure = ""
    for attempt in range(1, attempts + 1):
        before = state.completed()
        failures.clear()
        driver_failure = ""
        os.chdir(state.root)
        try:
            with state.log_path.open("a", encoding="utf-8") as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    print(
                        f"\n===== {state.name} query {position + 1}/{total} "
                        f"attempt {attempt} =====",
                        flush=True,
                    )
                    draft.task_execution(
                        "G3",
                        str(stable_root / "tools"),
                        {},
                        state.dataset,
                        [query],
                        str(state.progress_path),
                        0,
                        1,
                        retrieval_num,
                        position,
                        alias,
                        state.name,
                    )
        except (KeyboardInterrupt, SystemExit):
            os.chdir(previous_cwd)
            raise
        except BaseException as exc:  # noqa: BLE001 - classified below
            last_error = f"{type(exc).__name__}: {exc}"
            with state.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n----- {state.name} query {position} attempt {attempt} "
                    f"raised -----\n{traceback.format_exc()}"
                )
            if isinstance(exc, (ReleasedDriverFailure, TypeError, KeyError)):
                driver_failure = last_error
            elif not isinstance(exc, HarnessFailure):
                failures.record(TRANSPORT, last_error)
        finally:
            os.chdir(previous_cwd)

        wrote_record = state.completed() > before
        if failures.pending and wrote_record:
            # The released driver swallowed our failure and still emitted a
            # record built from degraded inputs. Do not score it.
            truncate_records(state.output_path, before)
            state.rolled_back += 1
            wrote_record = False

        if failures.kind == COST_CAP:
            raise CostCapReached(failures.message)
        if failures.kind == CACHE_MISS:
            raise SimulatorCacheMiss(failures.message)
        if wrote_record:
            state.query_attempts[position] = attempt
            failures.clear()
            return
        if driver_failure and not failures.pending:
            # The released agent crashed on its own output. Re-running the
            # query until it survives would be a re-roll, so the query is
            # dropped from both conditions instead and reported.
            state.record_driver_failure(position, driver_failure)
            return
        if attempt >= attempts:
            reason = failures.message or last_error or "no record was written"
            raise HarnessFailure(
                f"{state.name} query {position} failed {attempts} times: {reason}"
            )
        time.sleep(2 * attempt)
    raise HarnessFailure(f"{state.name} query {position} could not be executed.")


def _assert_driver_failures_within_tolerance(
    *,
    states: Mapping[str, ConditionState],
    planned: int,
    attempted: int,
    tolerance: float,
) -> None:
    """A high exclusion rate means the harness is broken, not the agent."""
    for state in states.values():
        excluded = len(state.driver_failures)
        if excluded <= 1:
            continue
        # A resume re-enters this check at position 0 while prior exclusions
        # already sit on disk. Count queries this condition has actually
        # touched, not the current loop index.
        touched = max(attempted, state.completed() + excluded)
        if touched < MIN_DRIVER_FAILURE_SAMPLE:
            continue
        if excluded / touched > tolerance:
            raise HarnessFailure(
                f"{state.name} dropped {excluded} of the first {touched} "
                f"queries (planned {planned}), above the tolerated "
                f"{tolerance:.0%}. Investigate the harness before spending "
                "more; the last reason was "
                f"{list(state.driver_failures.values())[-1]}"
            )


def _summarize(
    *,
    run_root: Path,
    manifest: Mapping[str, Any],
    states: Mapping[str, ConditionState],
    plan: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    backend: StableExecutionAdapter,
    agent_usage: Path,
    simulator_usage: Path,
    budget: CostBudget,
    stop_reason: str,
) -> dict[str, Any]:
    indexed = {name: state.positions() for name, state in states.items()}
    # Only queries both conditions completed can be compared, so the paired
    # denominator is the intersection and every drop is reported.
    paired_positions = sorted(
        set.intersection(*(set(rows) for rows in indexed.values()))
        if indexed
        else set()
    )
    conditions: dict[str, Any] = {}
    for name, state in states.items():
        records = indexed[name]
        per_query = [
            {
                "position": position,
                "query_index": plan[position]["query_index"],
                "query_id": plan[position]["query_id"],
                "correct_path": correct_path(records[position], queries[position]),
                "correct_path_released": correct_path_released(
                    records[position], queries[position]
                ),
                "self_check_passed": records[position].get("check_index") == 1,
                "executed_path": _executed_path(records[position]),
                "attempts": state.query_attempts.get(position, 1),
            }
            for position in paired_positions
        ]

        def rate(key: str, rows: list[dict[str, Any]] = per_query) -> float | None:
            if not rows:
                return None
            return round(sum(row[key] for row in rows) / len(rows), 4)

        conditions[name] = {
            "condition": name,
            "records": len(records),
            "scored_queries": len(per_query),
            "complete": len(records) + len(state.driver_failures) == len(plan),
            "correct_path_rate": rate("correct_path"),
            "correct_path_rate_released": rate("correct_path_released"),
            "self_check_rate": rate("self_check_passed"),
            "unparsed_completions": state.unparsed_completions,
            "format_resamples": state.format_resamples,
            "rolled_back_records": state.rolled_back,
            "excluded_queries": {
                str(position): reason
                for position, reason in sorted(state.driver_failures.items())
            },
            "raw_output": str(state.output_path),
            "console_log": str(state.log_path),
            "per_query": per_query,
        }

    summary = {
        "manifest": dict(manifest),
        "stop_reason": stop_reason,
        "planned_queries": len(plan),
        "scored_queries": len(paired_positions),
        "excluded_positions": sorted(
            {
                position
                for state in states.values()
                for position in state.driver_failures
            }
        ),
        "pending_positions": sorted(
            position
            for position in range(len(plan))
            if any(not state.done(position) for state in states.values())
        ),
        "conditions": conditions,
        "backend_stats": dict(backend.stats),
        "usage": {
            "agent": usage_totals(agent_usage),
            "simulator": usage_totals(simulator_usage),
            "cost_usd": round(budget.spent_usd, 6),
            "cap_usd": budget.maximum_usd,
        },
        "scope_note": (
            (
                "Live RapidAPI slice of ToolBench G3 hosts using 2026 "
                "subscriptions and ToolEnv URLs from the StableToolBench "
                "snapshot. Identical calls are cached inside the run so "
                "Raw/DRAFT/Ours share one HTTP environment. Not a 2024 "
                "replica of Qu et al."
            )
            if backend.live_api_key
            else (
                "Adapted environment: RapidAPI is replaced by the "
                "StableToolBench ToolEnv snapshot. "
                + (
                    "Invalid calls are rejected against that snapshot schema "
                    "before cache or simulation, so a stale parameter list "
                    "cannot succeed by simulator leniency. "
                    if backend.schema_gate
                    else "Valid and invalid calls may both be served by cache "
                    "or the pinned simulator. "
                )
                + "Reported as a transport-adapted measurement, not a literal "
                "RapidAPI replication."
            )
        ),
    }
    _dump_json(run_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the released DRAFT DFSDT agent on ToolBench G3 with a pinned "
            "environment, seeded paired condition order and a run manifest."
        )
    )
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--stable-root", type=Path, default=DEFAULT_STABLE_ROOT)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--agent-provider", default=None)
    parser.add_argument("--simulator-model", default=DEFAULT_SIMULATOR_MODEL)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        help="Documentation conditions. Released names are Initial and DRAFT; "
        "others need --extra-documentation Condition=path.",
    )
    parser.add_argument(
        "--extra-documentation",
        action="append",
        default=[],
        help="Condition=path pairs, e.g. Ours=artifacts/documentation/Ours.json",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--retrieval-num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--query-attempts", type=int, default=2)
    parser.add_argument(
        "--max-driver-failure-rate",
        type=float,
        default=0.15,
        help=(
            "Stop if the released driver crashes on more than this share of "
            "attempted queries; that indicates a harness fault, not an agent "
            "outcome."
        ),
    )
    parser.add_argument(
        "--simulator-mode",
        choices=["simulate", "replay"],
        default="simulate",
        help="replay refuses to call the simulator and fails on a cache miss.",
    )
    parser.add_argument(
        "--schema-gate",
        action="store_true",
        help=(
            "Reject calls that miss required ToolEnv parameters or send "
            "unknown ones, instead of letting the simulator answer them."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["stable", "live"],
        default="stable",
        help=(
            "stable uses the ToolEnv cache and simulator. live calls the "
            "subscribed RapidAPI hosts and caches identical requests inside "
            "the run."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Discard existing outputs in this root instead of resuming them.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/results/g3_calibration_v2"),
    )
    args = parser.parse_args()

    run_root = args.output_root
    run_root.mkdir(parents=True, exist_ok=True)
    budget = CostBudget(
        args.max_cost_usd,
        spent_usd=run_cost(run_root) if args.resume else 0.0,
    )
    try:
        summary = run_paired_g3(
            draft_root=args.draft_root.resolve(),
            stable_root=args.stable_root.resolve(),
            run_root=run_root.resolve(),
            conditions=list(args.conditions),
            agent_model=args.agent_model,
            simulator_model=args.simulator_model,
            agent_provider=args.agent_provider,
            query_indices=list(range(args.start, args.start + args.limit)),
            retrieval_num=args.retrieval_num,
            seed=args.seed,
            budget=budget,
            resume=args.resume,
            replay_only=args.simulator_mode == "replay"
            and args.backend != "live",
            query_attempts=args.query_attempts,
            max_driver_failure_rate=args.max_driver_failure_rate,
            extra_documentation=parse_extra_documentation(
                args.extra_documentation
            ),
            schema_gate=args.schema_gate,
            live_rapidapi=args.backend == "live",
        )
    except CostCapReached as exc:
        print(f"STOP cost-cap: {exc}")
        return 2
    except HarnessFailure as exc:
        print(f"STOP infrastructure: {exc}")
        return 3
    except ManifestMismatch as exc:
        print(f"STOP provenance: {exc}")
        return 4
    printable = {
        key: value for key, value in summary.items() if key != "manifest"
    }
    for condition in printable["conditions"].values():
        condition.pop("per_query", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
