"""One-shot live lookup of RestBench-Spotify identifier values.

Account ids and catalog search hits do not depend on the evaluation seed.
This script talks to the Spotify Web API once and writes a frozen table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from tooldoc_nir.openrouter import load_env_file
from tooldoc_nir.restbench_http import SpotifyClient

OUT = Path("artifacts/documentation/Spotify_gold_ids.json")
API = "https://api.spotify.com/v1"


def _get(client: SpotifyClient, path: str, params: dict[str, Any] | None = None) -> Any:
    token = client._access_token()
    response = client._get_session().get(
        API + path,
        headers={"Authorization": f"Bearer {token}"},
        params=params or {},
        timeout=30,
    )
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


def _search(client: SpotifyClient, q: str, type_name: str) -> dict[str, Any]:
    payload = _get(client, "/search", {"q": q, "type": type_name, "limit": 5})
    items = ((payload.get(type_name + "s") or {}).get("items")) or []
    items = [item for item in items if isinstance(item, dict)]
    return {
        "query": q,
        "type": type_name,
        "ids": [str(item.get("id") or "") for item in items if item.get("id")],
        "names": [str(item.get("name") or "") for item in items],
        "uris": [str(item.get("uri") or "") for item in items if item.get("uri")],
    }


def main() -> None:
    load_env_file()
    client = SpotifyClient(
        os.environ.get("SPOTIFY_CLIENT_ID", ""),
        os.environ.get("SPOTIFY_CLIENT_SECRET", ""),
        refresh_token=os.environ.get("SPOTIFY_REFRESH_TOKEN", ""),
    )
    me = _get(client, "/me")
    playlists = _get(client, "/me/playlists", {"limit": 50})
    tracks = _get(client, "/me/tracks", {"limit": 50})
    albums = _get(client, "/me/albums", {"limit": 50})
    following = _get(client, "/me/following", {"type": "artist", "limit": 50})
    top_artists = _get(client, "/me/top/artists", {"limit": 5})
    playlist_items = [
        item
        for item in (playlists.get("items") or [])
        if isinstance(item, dict)
    ]
    saved_tracks = [
        (item.get("track") or {})
        for item in (tracks.get("items") or [])
        if isinstance(item, dict)
    ]
    saved_albums = [
        (item.get("album") or {})
        for item in (albums.get("items") or [])
        if isinstance(item, dict)
    ]
    followed = [
        item
        for item in ((following.get("artists") or {}).get("items") or [])
        if isinstance(item, dict)
    ]
    tops = [item for item in (top_artists.get("items") or []) if isinstance(item, dict)]
    named = {
        str(item.get("name") or ""): str(item.get("id") or "")
        for item in playlist_items
        if item.get("id")
    }
    payload = {
        "user_id": str(me.get("id") or ""),
        "display_name": str(me.get("display_name") or ""),
        "playlists": {
            "ordered_ids": [str(item.get("id") or "") for item in playlist_items],
            "ordered_names": [str(item.get("name") or "") for item in playlist_items],
            "first": str(playlist_items[0].get("id") or "") if playlist_items else "",
            "by_name": named,
        },
        "library_tracks": {
            "ids": [str(item.get("id") or "") for item in saved_tracks if item.get("id")],
            "uris": [str(item.get("uri") or "") for item in saved_tracks if item.get("uri")],
            "names": [str(item.get("name") or "") for item in saved_tracks],
        },
        "library_albums": {
            "ids": [str(item.get("id") or "") for item in saved_albums if item.get("id")],
            "names": [str(item.get("name") or "") for item in saved_albums],
        },
        "following_artists": {
            "ids": [str(item.get("id") or "") for item in followed if item.get("id")],
            "names": [str(item.get("name") or "") for item in followed],
        },
        "top_artists": {
            "ids": [str(item.get("id") or "") for item in tops if item.get("id")],
            "names": [str(item.get("name") or "") for item in tops],
        },
        "catalog": {
            "mariah_carey": _search(client, "Mariah Carey", "track"),
            "dark_side": _search(client, "The Dark Side of the Moon Pink Floyd", "album"),
            "summertime_sadness": _search(
                client, "Summertime Sadness Lana Del Rey", "track"
            ),
            "taylor_swift": _search(client, "Taylor Swift", "artist"),
            "jay_chou_mojito": _search(client, "Mojito Jay Chou", "album"),
            "beatles": _search(client, "The Beatles", "artist"),
            "bigbang": _search(client, "BIGBANG", "artist"),
            "when_you_believe": _search(client, "When You Believe", "track"),
            "quiet_songs": _search(client, "quiet songs", "track"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)
    print("user_id", payload["user_id"], "playlists", payload["playlists"]["ordered_names"])
    print("first_playlist", payload["playlists"]["first"])
    print("library_tracks", len(payload["library_tracks"]["ids"]))
    print("following", payload["following_artists"]["names"][:5])
    print("top", payload["top_artists"]["names"][:3])
    for key, row in payload["catalog"].items():
        print(key, row["names"][:2], row["ids"][:2])


if __name__ == "__main__":
    main()
