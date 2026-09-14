"""G3 nested-slice gates. Offline; no ToolEnv."""

from tooldoc_nir.g3_nested import classify_query


def _query(*, text: str, gold: list[tuple[str, str]], apis: list[dict]) -> dict:
    return {
        "query": text,
        "relevant APIs": [[tool, api] for tool, api in gold],
        "api_list": apis,
        "query_id": 1,
    }


def _search() -> dict:
    return {
        "tool_name": "Vimeo",
        "api_name": "SearchVideos",
        "required_parameters": [
            {"name": "query", "type": "STRING", "description": "search terms", "default": ""}
        ],
    }


def _details() -> dict:
    return {
        "tool_name": "Vimeo",
        "api_name": "GetVideo",
        "required_parameters": [
            {"name": "video_id", "type": "STRING", "description": "Video identifier", "default": ""}
        ],
    }


def _stream() -> dict:
    return {
        "tool_name": "YTStream",
        "api_name": "Download/Stream",
        "required_parameters": [
            {
                "name": "id",
                "type": "STRING",
                "description": "Youtube Video Id.",
                "default": "UxxajLWwzqY",
            }
        ],
    }


def test_parallel_query_with_literal_id_is_independent() -> None:
    row = classify_query(
        _query(
            text=(
                "Search Vimeo for 'documentary' and stream YouTube "
                "id 'UxxajLWwzqY'."
            ),
            gold=[("Vimeo", "SearchVideos"), ("YTStream", "Download/Stream")],
            apis=[_search(), _stream()],
        )
    )
    assert row["label"] == "independent"
    assert row["edges"] == []


def test_later_id_not_in_query_is_nested() -> None:
    row = classify_query(
        _query(
            text="Find a documentary on Vimeo and then fetch that video's details.",
            gold=[("Vimeo", "SearchVideos"), ("Vimeo", "GetVideo")],
            apis=[_search(), _details()],
        )
    )
    assert row["label"] == "nested"
    assert row["edges"][0]["param"] == "video_id"
    assert row["edges"][0]["from_api"] == "SearchVideos"
