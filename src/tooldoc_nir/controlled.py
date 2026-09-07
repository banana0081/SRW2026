from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field

from .contract import normalize_type
from .dynamic_models import (
    ExecutionObservation,
    LivenessState,
    ObservationSource,
)
from .models import ApiDocument, normalize_identifier
from .probe import HttpProber, default_arguments


class ControlledScenario(BaseModel):
    scenario_id: str
    baseline: ApiDocument
    runtime_method: str
    runtime_required: dict[str, str] = Field(default_factory=dict)
    runtime_optional: dict[str, str] = Field(default_factory=dict)
    response_body: Any = Field(default_factory=dict)
    availability_pattern: list[int] = Field(default_factory=lambda: [200])
    expected_events: list[str] = Field(default_factory=list)
    expected_liveness: LivenessState = LivenessState.HEALTHY

    @classmethod
    def unchanged(
        cls, scenario_id: str, document: ApiDocument
    ) -> "ControlledScenario":
        return cls(
            scenario_id=scenario_id,
            baseline=document,
            runtime_method=(document.method or "GET").upper(),
            runtime_required={
                normalize_identifier(item.name): normalize_type(item.type)
                for item in document.required_parameters
            },
            runtime_optional={
                normalize_identifier(item.name): normalize_type(item.type)
                for item in document.optional_parameters
            },
            response_body=materialize_response(
                document.template_response or {"result": "ok"}
            ),
        )


def materialize_response(value: Any, *, schema_context: bool = False) -> Any:
    if isinstance(value, dict):
        declared_type = normalize_type(str(value.get("type", "unknown")))
        schema_keys = {
            "$schema",
            "additionalProperties",
            "description",
            "example",
            "format",
            "items",
            "properties",
            "required",
            "title",
            "type",
        }
        is_schema = (
            schema_context
            or "$schema" in value
            or "properties" in value
            or "items" in value
            or (
                declared_type
                in {"array", "boolean", "integer", "number", "object", "string"}
                and set(value) <= schema_keys
            )
        )
        if is_schema:
            if "example" in value:
                return value["example"]
            if "properties" in value:
                properties = value.get("properties")
                if isinstance(properties, dict):
                    return {
                        str(key): materialize_response(
                            nested, schema_context=True
                        )
                        for key, nested in properties.items()
                    }
            if declared_type == "array" and isinstance(value.get("items"), dict):
                return [
                    materialize_response(
                        value["items"], schema_context=True
                    )
                ]
            return {
                "boolean": True,
                "integer": 1,
                "number": 1.0,
                "object": {},
                "string": "value",
            }.get(declared_type)
        return {
            str(key): materialize_response(nested, schema_context=False)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return (
            [materialize_response(value[0], schema_context=schema_context)]
            if value
            else []
        )
    if isinstance(value, str):
        declared = normalize_type(value)
        return {
            "boolean": True,
            "integer": 1,
            "number": 1.0,
            "array": ["value"],
            "object": {"key": "value"},
            "string": "value",
        }.get(declared, value)
    if value is None:
        return None
    return value


def mutate_add_required(
    scenario: ControlledScenario,
    name: str = "locale",
    field_type: str = "string",
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    normalized = normalize_identifier(name)
    mutated.runtime_required[normalized] = normalize_type(field_type)
    mutated.expected_events.append(f"request_field:add:{normalized}")
    return mutated


def mutate_remove_parameter(
    scenario: ControlledScenario, name: str
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    normalized = normalize_identifier(name)
    mutated.runtime_required.pop(normalized, None)
    mutated.runtime_optional.pop(normalized, None)
    mutated.expected_events.append(f"request_field:remove:{normalized}")
    return mutated


def mutate_requiredness(
    scenario: ControlledScenario, name: str, *, required: bool
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    normalized = normalize_identifier(name)
    if required:
        field_type = mutated.runtime_optional.pop(normalized)
        mutated.runtime_required[normalized] = field_type
    else:
        field_type = mutated.runtime_required.pop(normalized)
        mutated.runtime_optional[normalized] = field_type
    mutated.expected_events.append(
        f"request_field:replace:{normalized}.required"
    )
    return mutated


def mutate_rename_parameter(
    scenario: ControlledScenario,
    old_name: str,
    new_name: str,
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    old = normalize_identifier(old_name)
    new = normalize_identifier(new_name)
    if old in mutated.runtime_required:
        mutated.runtime_required[new] = mutated.runtime_required.pop(old)
    elif old in mutated.runtime_optional:
        mutated.runtime_optional[new] = mutated.runtime_optional.pop(old)
    else:
        raise KeyError(f"Unknown parameter {old_name!r}")
    mutated.expected_events.extend(
        [
            f"request_field:remove:{old}",
            f"request_field:add:{new}",
        ]
    )
    return mutated


def mutate_parameter_type(
    scenario: ControlledScenario, name: str, field_type: str
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    normalized = normalize_identifier(name)
    if normalized in mutated.runtime_required:
        mutated.runtime_required[normalized] = normalize_type(field_type)
    elif normalized in mutated.runtime_optional:
        mutated.runtime_optional[normalized] = normalize_type(field_type)
    else:
        raise KeyError(f"Unknown parameter {name!r}")
    mutated.expected_events.append(f"request_field:replace:{normalized}.type")
    return mutated


def mutate_method(
    scenario: ControlledScenario, method: str = "POST"
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    mutated.runtime_method = method.upper()
    mutated.expected_events.append("method:replace:method")
    return mutated


def mutate_response(
    scenario: ControlledScenario, response_body: Any, *events: str
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    mutated.response_body = deepcopy(response_body)
    mutated.expected_events.extend(events)
    return mutated


def mutate_availability(
    scenario: ControlledScenario,
    statuses: list[int],
    expected_state: LivenessState,
) -> ControlledScenario:
    mutated = scenario.model_copy(deep=True)
    mutated.availability_pattern = statuses
    mutated.expected_liveness = expected_state
    return mutated


def _matches_type(value: Any, expected: str) -> bool:
    expected = normalize_type(expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, str) and value.lstrip("-").isdigit()
    if expected == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
    if expected == "boolean":
        return isinstance(value, bool) or (
            isinstance(value, str)
            and value.casefold() in {"true", "false", "0", "1"}
        )
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


class _ScenarioServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        scenarios: dict[str, ControlledScenario],
    ) -> None:
        super().__init__(server_address, _ScenarioHandler)
        self.scenarios = scenarios
        self.call_counts: dict[str, int] = {
            scenario_id: 0 for scenario_id in scenarios
        }


class _ScenarioHandler(BaseHTTPRequestHandler):
    server: _ScenarioServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def _json(self, status: int, payload: Any, **headers: str) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _arguments(self) -> dict[str, Any]:
        parsed = urlparse(self.path)
        if self.command in {"GET", "HEAD", "OPTIONS"}:
            return {
                normalize_identifier(key): values[-1]
                for key, values in parse_qs(
                    parsed.query, keep_blank_values=True
                ).items()
            }
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return {
            normalize_identifier(str(key)): value
            for key, value in body.items()
        }

    def _handle(self) -> None:
        path = urlparse(self.path).path
        scenario_id = path.removeprefix("/scenario/")
        scenario = self.server.scenarios.get(scenario_id)
        if scenario is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown scenario"})
            return

        index = self.server.call_counts[scenario_id]
        self.server.call_counts[scenario_id] = index + 1
        status = scenario.availability_pattern[
            min(index, len(scenario.availability_pattern) - 1)
        ]
        if status != 200:
            self._json(status, {"error": {"code": "service_failure"}})
            return
        if self.command != scenario.runtime_method:
            self._json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {"error": {"code": "method_not_allowed"}},
                Allow=scenario.runtime_method,
            )
            return

        arguments = self._arguments()
        accepted = {
            **scenario.runtime_required,
            **scenario.runtime_optional,
        }
        for name in arguments:
            if name not in accepted:
                self._json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {
                        "error": {
                            "code": "unknown_parameter",
                            "parameter": name,
                        }
                    },
                )
                return
        for name, expected_type in scenario.runtime_required.items():
            if name not in arguments:
                self._json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {
                        "error": {
                            "code": "missing_parameter",
                            "parameter": name,
                            "expected_type": expected_type,
                        }
                    },
                )
                return
        for name, value in arguments.items():
            expected_type = accepted[name]
            if not _matches_type(value, expected_type):
                self._json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {
                        "error": {
                            "code": "invalid_type",
                            "parameter": name,
                            "expected_type": expected_type,
                        }
                    },
                )
                return
        self._json(HTTPStatus.OK, scenario.response_body)


class ControlledApiServer(AbstractContextManager["ControlledApiServer"]):
    def __init__(self, scenarios: list[ControlledScenario]) -> None:
        self.scenarios = {item.scenario_id: item for item in scenarios}
        self._server: _ScenarioServer | None = None
        self._thread: Thread | None = None

    def __enter__(self) -> "ControlledApiServer":
        self._server = _ScenarioServer(("127.0.0.1", 0), self.scenarios)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def reset_counts(self) -> None:
        """Start the availability pattern over without restarting the server."""
        if self._server is None:
            return
        for scenario_id in self._server.call_counts:
            self._server.call_counts[scenario_id] = 0

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def document(self, scenario_id: str) -> ApiDocument:
        scenario = self.scenarios[scenario_id]
        return scenario.baseline.model_copy(
            update={"url": f"{self.base_url}/scenario/{scenario_id}"}
        )


def _error_payload(observation: ExecutionObservation) -> dict[str, Any]:
    body = observation.response_body
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    return error if isinstance(error, dict) else {}


def _example_for_type(field_type: str) -> Any:
    field_type = normalize_type(field_type)
    return {
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "array": ["test"],
        "object": {"key": "value"},
    }.get(field_type, "test")


def explore_controlled_api(
    document: ApiDocument,
    *,
    successful_repetitions: int = 3,
    max_attempts: int = 12,
) -> list[ExecutionObservation]:
    """Use validation feedback to reach successful calls and gather evidence."""
    prober = HttpProber(
        timeout_seconds=2.0,
        allow_unsafe_methods=True,
    )
    current = document
    arguments = default_arguments(document)
    observations: list[ExecutionObservation] = []
    successes = 0

    for _ in range(max_attempts):
        observation = prober.probe(
            current,
            arguments=arguments,
            source=ObservationSource.CONTROLLED,
        )
        observations.append(observation)
        if observation.status_code == 200:
            successes += 1
            if successes >= successful_repetitions:
                break
            continue

        if observation.status_code == 405:
            allowed = observation.response_headers.get("allow")
            if allowed:
                current = current.model_copy(update={"method": allowed})
            continue

        error = _error_payload(observation)
        code = error.get("code")
        name = normalize_identifier(str(error.get("parameter", "")))
        if code == "missing_parameter" and name:
            arguments[name] = _example_for_type(
                str(error.get("expected_type", "string"))
            )
        elif code == "unknown_parameter" and name:
            arguments.pop(name, None)
        elif code == "invalid_type" and name:
            arguments[name] = _example_for_type(
                str(error.get("expected_type", "string"))
            )
    return observations

