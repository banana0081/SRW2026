"""A faithful port of DRAFT's Explorer / Analyzer / Rewriter loop.

The withdrawn Experiment 3a compared against a hand-written string builder that
had none of DRAFT's stages. This module runs the released algorithm instead: the
prompt files from `external/DRAFT/prompts` are used verbatim, the control flow
follows `DRAFT.py`, and the stopping rule is the same mean of smoothed BLEU and
embedding cosine with the same 0.75 threshold.

Three deviations, all forced by what is reachable from here and all reported:

- `text-embedding-ada-002` is replaced by a local sentence-transformers model.
  Embeddings only gate query diversity and the early stop, never the text.
- The executed API is whatever backend the caller supplies, since RapidAPI is
  not reachable.
- A completion that omits a documented key is resampled a bounded number of
  times instead of silently becoming `None`, so a transport hiccup cannot be
  mistaken for a rewrite.

One upstream detail is preserved on purpose: `DRAFT.py` never substitutes the
`{tool_description}` placeholder in `Analyzer.txt`, so the Analyzer sees that
literal token. Fixing it would no longer be the released method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_PROMPT_ROOT = Path("external/DRAFT/prompts")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPISODES = 5
SIMILARITY_CEILING = 0.9
STOP_DELTA = 0.75
KEY_ATTEMPTS = 3

RESERVED_NAMES = frozenset(
    {"from", "class", "return", "false", "true", "id", "and", "", "ID"}
)


class DraftRewriteFailure(RuntimeError):
    """A released DRAFT stage never returned the key its own prompt shows."""


@dataclass(frozen=True)
class DraftPrompts:
    explorer: str
    explorer_follow: str
    analyzer: str
    analyzer_follow: str
    rewriter: str
    rewriter_follow: str
    tool_document: str

    @classmethod
    def load(cls, root: Path = DEFAULT_PROMPT_ROOT) -> "DraftPrompts":
        def split(name: str) -> tuple[str, str]:
            text = (root / name).read_text(encoding="utf-8")
            head, follow = text.split("=========")
            return head, follow

        explorer, explorer_follow = split("Explorer.txt")
        analyzer, analyzer_follow = split("Analyzer.txt")
        rewriter, rewriter_follow = split("Rewriter.txt")
        return cls(
            explorer=explorer,
            explorer_follow=explorer_follow,
            analyzer=analyzer,
            analyzer_follow=analyzer_follow,
            rewriter=rewriter,
            rewriter_follow=rewriter_follow,
            tool_document=(root / "rewrite_tool_doc.txt").read_text(
                encoding="utf-8"
            ),
        )


@dataclass
class Episode:
    index: int
    query: str
    parameters: Any
    api_response: Any
    suggestion: str
    rewritten_description: str
    exploration_suggestion: str
    delta: float | None = None
    explorer_resamples: int = 0


@dataclass
class RewriteResult:
    api_name: str
    initial_description: str
    description: str
    episodes: list[Episode] = field(default_factory=list)
    stopped_early: bool = False

    def history(self) -> list[str]:
        return [self.initial_description] + [
            episode.rewritten_description for episode in self.episodes
        ]


def change_name(name: str) -> str:
    return "is_" + name.lower() if name in RESERVED_NAMES else name


def standardize(value: str) -> str:
    # The character class is copied from DRAFT.py including its stray `^`
    # members, which make a literal caret one of the characters it keeps.
    import re

    result = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", value)
    result = re.sub(r"(_)\1+", "_", result).lower()
    result = result.strip("_")
    if result and result[0].isdigit():
        result = "get_" + result
    return result


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    import numpy as np

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def sentence_delta(
    reference: str,
    candidate: str,
    embed: Callable[[str], Sequence[float]],
) -> float:
    """The released stopping statistic: mean of smoothed BLEU and cosine."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    similarity = cosine_similarity(embed(reference), embed(candidate))
    bleu = sentence_bleu(
        [reference.lower().split()],
        candidate.lower().split(),
        smoothing_function=SmoothingFunction().method4,
    )
    return (float(bleu) + similarity) / 2


def local_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Callable[[str], Sequence[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    cache: dict[str, Sequence[float]] = {}

    def embed(text: str) -> Sequence[float]:
        if text not in cache:
            cache[text] = model.encode(text, normalize_embeddings=True).tolist()
        return cache[text]

    return embed


def _require_keys(
    respond: Callable[[list[dict[str, Any]], str], Any],
    *,
    prompt: str,
    stage: str,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    for _ in range(KEY_ATTEMPTS):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        answer = respond(messages, stage)
        if isinstance(answer, dict) and all(key in answer for key in keys):
            return answer
    raise DraftRewriteFailure(
        f"{stage} never returned {keys} in {KEY_ATTEMPTS} samples."
    )


def _analyze_and_rewrite(
    *,
    prompts: DraftPrompts,
    tool_info: Mapping[str, Any],
    example: Mapping[str, Any],
    history: Sequence[str],
    respond: Callable[[list[dict[str, Any]], str], Any],
) -> tuple[str, str, str]:
    """Run Analyzer then Rewriter on one usage example.

    `{tool_description}` in Analyzer.txt is left unsubstituted on purpose:
    `DRAFT.py` never fills that placeholder.
    """
    tool_description = str(dict(tool_info))
    analyzer_prompt = prompts.analyzer.replace(
        "{Tool Description}", tool_description
    ).replace("{usage_example}", str(dict(example)))
    if len(history) > 1:
        analyzer_prompt += prompts.analyzer_follow.replace(
            "{History}", str(list(history))
        )
    suggestion = _require_keys(
        respond,
        prompt=analyzer_prompt,
        stage="analyzer",
        keys=("Suggestions for tool description",),
    )
    rewriter_prompt = (
        prompts.rewriter.replace("{Tool Description}", tool_description)
        .replace("{usage_example}", str(dict(example)))
        .replace(
            "{Suggestions}",
            str(suggestion["Suggestions for tool description"]),
        )
        .replace("{tool_description}", str(tool_info["description"]))
    )
    if len(history) > 1:
        rewriter_prompt += prompts.rewriter_follow.replace(
            "{History}", str(list(history))
        )
    rewritten = _require_keys(
        respond,
        prompt=rewriter_prompt,
        stage="rewriter",
        keys=("Rewritten description", "Suggestions for exploring"),
    )
    return (
        str(suggestion["Suggestions for tool description"]),
        str(rewritten["Rewritten description"]),
        str(rewritten["Suggestions for exploring"]),
    )


def rewrite_api_description(
    *,
    prompts: DraftPrompts,
    category: str,
    tool_name: str,
    api_info: Mapping[str, Any],
    execute: Callable[[dict[str, Any]], Any],
    respond: Callable[[list[dict[str, Any]], str], Any],
    embed: Callable[[str], Sequence[float]],
    episodes: int = EPISODES,
) -> RewriteResult:
    api_name = str(api_info["name"])
    required = api_info.get("required_parameters") or []
    optional = api_info.get("optional_parameters") or []
    result = RewriteResult(
        api_name=api_name,
        initial_description=str(api_info.get("description") or ""),
        description=str(api_info.get("description") or ""),
    )
    explored_queries: list[str] = []
    explored_embeddings: list[Sequence[float]] = []
    explored_examples: list[dict[str, Any]] = []
    exploration_suggestion = ""
    # DRAFT.py rebinds `api_name` to its standardized form while building the
    # execution payload, so every episode after the first shows the agent the
    # mangled name instead of the documented one. Kept, because removing it
    # would change the prompts the released method actually sends.
    display_name = api_name

    for episode in range(episodes):
        tool_info = {
            "category": category,
            "name": display_name,
            "description": result.description,
            "required_parameters": required,
            "optional_parameters": optional,
        }
        tool_description = str(tool_info)

        explore_prompt = prompts.explorer.replace(
            "{Tool Description}", tool_description
        )
        resamples = 0
        if explored_queries:
            follow = prompts.explorer_follow.replace(
                "{Explored queries}", str(explored_queries)
            ).replace("{Suggestions}", exploration_suggestion)
            explore_prompt = explore_prompt + follow
            for _ in range(3):
                example = _require_keys(
                    respond,
                    prompt=explore_prompt,
                    stage="explorer",
                    keys=("User Query", "Parameters"),
                )
                current = embed(str(example["User Query"]))
                if all(
                    cosine_similarity(previous, current) < SIMILARITY_CEILING
                    for previous in explored_embeddings
                ):
                    break
                resamples += 1
                explore_prompt += (
                    f"\nYour last generate query '{example['User Query']}' is "
                    "too similar to the previous ones. Please generate a "
                    "different query."
                )
        else:
            example = _require_keys(
                respond,
                prompt=explore_prompt,
                stage="explorer",
                keys=("User Query", "Parameters"),
            )

        query = str(example["User Query"])
        explored_queries.append(query)
        explored_embeddings.append(embed(query))

        display_name = change_name(standardize(display_name))
        payload = {
            "category": category,
            "tool_name": change_name(standardize(tool_name)),
            "api_name": display_name,
            "tool_input": example["Parameters"],
            "strip": "filter",
        }
        example = dict(example)
        example["API_Response"] = execute(payload)
        explored_examples.append(example)

        previous_description = result.description
        suggestion_text, rewritten_text, exploration_suggestion = (
            _analyze_and_rewrite(
                prompts=prompts,
                tool_info=tool_info,
                example=example,
                history=result.history(),
                respond=respond,
            )
        )
        result.description = rewritten_text
        record = Episode(
            index=episode,
            query=query,
            parameters=example["Parameters"],
            api_response=example["API_Response"],
            suggestion=suggestion_text,
            rewritten_description=result.description,
            exploration_suggestion=exploration_suggestion,
            explorer_resamples=resamples,
        )
        result.episodes.append(record)

        # Upstream tests this after appending, so the first episode is already
        # eligible to stop: it compares the initial description to rewrite one.
        record.delta = sentence_delta(
            previous_description, result.description, embed
        )
        if record.delta > STOP_DELTA:
            result.stopped_early = True
            break

    return result


def rewrite_from_observations(
    *,
    prompts: DraftPrompts,
    category: str,
    tool_name: str,
    api_info: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    respond: Callable[[list[dict[str, Any]], str], Any],
    embed: Callable[[str], Sequence[float]],
) -> RewriteResult:
    """Analyzer/Rewriter only, on a frozen trial trace.

    Explorer is skipped so every documentation condition sees the same five
    observations. Schema fields are still never touched.
    """
    del tool_name
    result = RewriteResult(
        api_name=str(api_info["name"]),
        initial_description=str(api_info.get("description") or ""),
        description=str(api_info.get("description") or ""),
    )
    for index, observation in enumerate(observations):
        example = {
            "User Query": observation.get("query")
            or observation.get("User Query")
            or "",
            "Parameters": observation.get("parameters")
            or observation.get("Parameters")
            or {},
            "API_Response": observation.get("api_response")
            or observation.get("API_Response"),
        }
        tool_info = {
            "category": category,
            "name": str(api_info["name"]),
            "description": result.description,
            "required_parameters": api_info.get("required_parameters") or [],
            "optional_parameters": api_info.get("optional_parameters") or [],
        }
        previous_description = result.description
        suggestion_text, rewritten_text, exploration_suggestion = (
            _analyze_and_rewrite(
                prompts=prompts,
                tool_info=tool_info,
                example=example,
                history=result.history(),
                respond=respond,
            )
        )
        result.description = rewritten_text
        record = Episode(
            index=index,
            query=str(example["User Query"]),
            parameters=example["Parameters"],
            api_response=example["API_Response"],
            suggestion=suggestion_text,
            rewritten_description=result.description,
            exploration_suggestion=exploration_suggestion,
        )
        result.episodes.append(record)
        record.delta = sentence_delta(
            previous_description, result.description, embed
        )
        if record.delta > STOP_DELTA:
            result.stopped_early = True
            break
    return result


def rewrite_tool_description(
    *,
    prompts: DraftPrompts,
    tool: Mapping[str, Any],
    respond: Callable[[list[dict[str, Any]], str], Any],
) -> str:
    prompt = prompts.tool_document.replace("{Tool Description}", str(dict(tool)))
    answer = _require_keys(
        respond,
        prompt=prompt,
        stage="tool_rewriter",
        keys=("tool_description",),
    )
    return str(answer["tool_description"])


def rewrite_tool(
    *,
    prompts: DraftPrompts,
    tool: Mapping[str, Any],
    execute: Callable[[dict[str, Any]], Any],
    respond: Callable[[list[dict[str, Any]], str], Any],
    embed: Callable[[str], Sequence[float]],
    episodes: int = EPISODES,
) -> tuple[dict[str, Any], list[RewriteResult]]:
    """Rewrite every API description in one tool, then the tool blurb."""
    updated = json.loads(json.dumps(dict(tool), ensure_ascii=False))
    results: list[RewriteResult] = []
    for name, api_info in (updated.get("tool_guidelines") or {}).items():
        result = rewrite_api_description(
            prompts=prompts,
            category=str(updated.get("category") or ""),
            tool_name=str(updated.get("tool_name") or ""),
            api_info=api_info,
            execute=execute,
            respond=respond,
            embed=embed,
            episodes=episodes,
        )
        # The released loop only ever replaces the natural-language
        # description; no schema field is touched.
        api_info["description"] = result.description
        results.append(result)
        del name
    updated["tool_description"] = rewrite_tool_description(
        prompts=prompts, tool=updated, respond=respond
    )
    return updated, results


def main() -> int:
    """Offline smoke: load prompts and refuse to call a model."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect the released DRAFT rewrite prompts."
    )
    parser.add_argument(
        "--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT
    )
    args = parser.parse_args()
    prompts = DraftPrompts.load(args.prompt_root)
    print(
        json.dumps(
            {
                "prompt_root": str(args.prompt_root),
                "analyzer_keeps_tool_description_placeholder": (
                    "{tool_description}" in prompts.analyzer
                ),
                "episodes": EPISODES,
                "stop_delta": STOP_DELTA,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
