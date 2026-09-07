from pathlib import Path

from tooldoc_nir.documentation_store import (
    parse_extra_documentation,
    resolve_documentation_path,
)


def test_extra_documentation_overrides_released_path(tmp_path: Path) -> None:
    extras = parse_extra_documentation(
        [f"Ours={tmp_path / 'Ours.json'}"]
    )
    path = resolve_documentation_path(
        draft_root=tmp_path / "draft",
        condition="Ours",
        extra=extras,
    )
    assert path == tmp_path / "Ours.json"


def test_released_conditions_stay_on_the_upstream_tree(tmp_path: Path) -> None:
    path = resolve_documentation_path(
        draft_root=tmp_path,
        condition="Initial",
        extra={},
    )
    assert path == tmp_path / "dataset" / "ToolBench" / "tool_instruction" / "Initial.json"
