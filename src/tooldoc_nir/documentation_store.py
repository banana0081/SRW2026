"""Apply Ours documentation and extra condition files without touching upstream."""

from __future__ import annotations

from pathlib import Path


RELEASED_CONDITIONS = ("Initial", "DRAFT")


def released_documentation_path(draft_root: Path, condition: str) -> Path:
    return (
        draft_root
        / "dataset"
        / "ToolBench"
        / "tool_instruction"
        / f"{condition}.json"
    )


def parse_extra_documentation(items: list[str] | None) -> dict[str, Path]:
    extras: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"extra documentation {item!r} must look like Condition=path"
            )
        name, raw_path = item.split("=", 1)
        extras[name.strip()] = Path(raw_path.strip())
    return extras


def resolve_documentation_path(
    *,
    draft_root: Path,
    condition: str,
    extra: dict[str, Path] | None = None,
) -> Path:
    extras = extra or {}
    if condition in extras:
        return extras[condition]
    return released_documentation_path(draft_root, condition)
