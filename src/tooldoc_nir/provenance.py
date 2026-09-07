"""Run provenance: pin every input that can change a measured number.

The first GPT-4o-mini G3 run could not be audited afterwards because nothing
recorded which documents, upstream agent code, simulator cache or decoding
parameters produced it. Every run now writes a manifest, and resuming a run
whose fingerprint no longer matches is refused instead of silently mixing
results from two different harnesses.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

UPSTREAM_PINNED_FILES = (
    "Inference_DFSDT.py",
    "dataset/ToolBench/tool_instruction/Initial.json",
    "dataset/ToolBench/tool_instruction/DRAFT.json",
    "dataset/ToolBench/test_data/G3.json",
)

HARNESS_PINNED_MODULES = (
    "draft_agent_reproduction.py",
    "openrouter.py",
    "provenance.py",
)

MANIFEST_NAME = "manifest.json"

FINGERPRINT_FIELDS = (
    "upstream_digests",
    "harness_digests",
    "agent",
    "simulator",
    "decoding",
    "protocol",
    "queries",
)


class ManifestMismatch(RuntimeError):
    """Raised when a resumed run no longer matches its recorded manifest."""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_or_missing(path: Path) -> str:
    return file_digest(path) if path.exists() else "missing"


def upstream_digests(draft_root: Path) -> dict[str, str]:
    """Hash the released DRAFT artifacts the measurement depends on."""
    return {
        name: _digest_or_missing(draft_root / name)
        for name in UPSTREAM_PINNED_FILES
    }


def harness_digests() -> dict[str, str]:
    """Hash our own modules that decide how the agent and transport behave."""
    package_root = Path(__file__).resolve().parent
    return {
        name: _digest_or_missing(package_root / name)
        for name in HARNESS_PINNED_MODULES
    }


def payload_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def git_commit(repository_root: Path | None = None) -> str:
    root = repository_root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def build_manifest(
    *,
    draft_root: Path,
    stable_root: Path,
    agent: Mapping[str, Any],
    simulator: Mapping[str, Any],
    decoding: Mapping[str, Any],
    protocol: Mapping[str, Any],
    queries: Mapping[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "git_commit": git_commit(),
        "draft_root": str(draft_root),
        "stable_root": str(stable_root),
        "upstream_digests": upstream_digests(draft_root),
        "harness_digests": harness_digests(),
        "agent": dict(agent),
        "simulator": dict(simulator),
        "decoding": dict(decoding),
        "protocol": dict(protocol),
        "queries": dict(queries),
    }
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest


def fingerprint(manifest: Mapping[str, Any]) -> str:
    """Digest only the fields that must stay constant within one run."""
    return payload_digest(
        {field: manifest.get(field) for field in FINGERPRINT_FIELDS}
    )


def mismatched_fields(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[str]:
    return [
        field
        for field in FINGERPRINT_FIELDS
        if recorded.get(field) != current.get(field)
    ]


def load_manifest(run_root: Path) -> dict[str, Any] | None:
    path = run_root / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(run_root: Path, manifest: Mapping[str, Any]) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def reconcile_manifest(
    *,
    run_root: Path,
    manifest: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Persist the manifest, refusing to resume across a harness change."""
    recorded = load_manifest(run_root)
    if recorded is None or not resume:
        stored = dict(manifest)
        write_manifest(run_root, stored)
        return stored
    if fingerprint(recorded) == manifest["fingerprint"]:
        return recorded
    fields = mismatched_fields(recorded, manifest)
    raise ManifestMismatch(
        "Refusing to resume: the recorded manifest in "
        f"{run_root / MANIFEST_NAME} was produced by a different harness. "
        f"Changed fields: {', '.join(fields) or 'fingerprint'}. "
        "Start a fresh output root instead of mixing runs."
    )


def describe_digests(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): _digest_or_missing(path) for path in paths}
