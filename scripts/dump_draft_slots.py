"""Where DRAFT's single description field lands in DFSDT slots."""

from __future__ import annotations

import json
from pathlib import Path

from tooldoc_nir.restbench_adapt import wrap_document
from tooldoc_nir.restbench_data import load_instructions
from tooldoc_nir.restbench_ours import hop_params

OUT = Path("artifacts/documentation/TMDB_draft_slots.json")
FAKE_IDS = ("11111", "13579")


def main() -> None:
    draft = load_instructions(Path("external/DRAFT"), "TMDB", "DRAFT")
    initial = load_instructions(Path("external/DRAFT"), "TMDB", "Initial")
    both_same = 0
    hops = 0
    hops_with_example = 0
    fake_example = 0
    samples = []
    for key, document in draft.items():
        wrapped = wrap_document(document)
        guideline = wrapped["tool_guidelines"][wrapped["tool_name"]]
        u_t = wrapped["tool_description"]
        d_t = guideline["description"]
        if u_t == d_t:
            both_same += 1
        url = str(document.get("url") or "")
        is_hop = bool(hop_params(url))
        if is_hop:
            hops += 1
        example = document.get("example") or {}
        blob = json.dumps(example, ensure_ascii=False)
        if is_hop and example:
            hops_with_example += 1
        if any(token in blob for token in FAKE_IDS):
            fake_example += 1
        if document.get("tool_name") == "GET_movie_movie_id_reviews":
            samples.append(
                {
                    "tool": document.get("tool_name"),
                    "has_own_tool_description": "tool_description" in document,
                    "u_t_equals_d_t": u_t == d_t,
                    "u_t_head": u_t[:160],
                    "initial_description": (initial[key].get("description") or "")[:80],
                    "example": example,
                }
            )
    report = {
        "n": len(draft),
        "u_t_equals_d_t": both_same,
        "hop_apis": hops,
        "hops_with_example": hops_with_example,
        "examples_with_fake_ids": fake_example,
        "adapter": "wrap_document copies description into tool_description when absent",
        "samples": samples,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
