"""One-shot Spotify authorization-code exchange. Writes refresh token to .env."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import sys
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tooldoc_nir.openrouter import load_env_file  # noqa: E402

REDIRECT = "http://127.0.0.1:8888/callback"
SCOPES = " ".join(
    [
        "user-read-private",
        "user-read-email",
        "user-read-currently-playing",
        "user-read-playback-state",
        "user-modify-playback-state",
        "user-follow-read",
        "user-follow-modify",
        "user-library-read",
        "user-library-modify",
        "user-top-read",
        "playlist-read-private",
        "playlist-read-collaborative",
        "playlist-modify-public",
        "playlist-modify-private",
    ]
)


def _env_value(name: str) -> str:
    import os

    return (os.environ.get(name) or "").strip()


def upsert_env(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    found = False
    updated: list[str] = []
    for line in lines:
        if line.startswith(key + "="):
            updated.append(f"{key}={value}")
            found = True
        else:
            updated.append(line)
    if not found:
        if updated and updated[-1] != "":
            updated.append("")
        updated.append(f"{key}={value}")
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> int:
    load_env_file(ROOT / ".env")
    client_id = _env_value("SPOTIFY_CLIENT_ID")
    client_secret = _env_value("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET missing in .env")
        return 1
    state = secrets.token_urlsafe(16)
    box: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_error(404)
                return
            query = parse_qs(parsed.query)
            if query.get("state", [""])[0] != state:
                self.send_error(400, "state mismatch")
                return
            if query.get("error"):
                box["error"] = query["error"][0]
                body = b"Spotify denied access. You can close this tab."
            else:
                box["code"] = query.get("code", [""])[0]
                body = b"OK. You can close this tab."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 8888), Handler)
    authorize = "https://accounts.spotify.com/authorize?" + urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": REDIRECT,
            "scope": SCOPES,
            "state": state,
            "show_dialog": "true",
        }
    )
    print("Waiting on", REDIRECT)
    print("AUTHORIZE_URL", authorize, flush=True)
    webbrowser.open(authorize)
    while "code" not in box and "error" not in box:
        server.handle_request()
    server.server_close()
    if box.get("error"):
        print("Spotify returned error:", box["error"])
        return 1
    code = str(box.get("code") or "")
    if not code:
        print("No authorization code.")
        return 1
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
        },
        auth=(client_id, client_secret),
        timeout=30,
    )
    if not response.ok:
        print("Token exchange failed:", response.status_code)
        return 1
    payload = response.json()
    refresh = str(payload.get("refresh_token") or "")
    if not refresh:
        print("Spotify did not return a refresh_token. Re-consent with show_dialog=true.")
        return 1
    upsert_env(ROOT / ".env", "SPOTIFY_REFRESH_TOKEN", refresh)
    print("Wrote SPOTIFY_REFRESH_TOKEN to .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
