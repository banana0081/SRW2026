"""Live RestBench HTTP: TMDB (and later Spotify) from documentation URLs."""

from __future__ import annotations

from hashlib import sha256
import json
import re
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

import requests

PATH_PARAM_RE = re.compile(r"\{([^{}]+)\}")
RESPONSE_CHARS = 2000
TMDB_TIMEOUT = 30.0
SPOTIFY_TIMEOUT = 30.0
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"


def http_method_from_tool(name: str) -> str:
    token = (name or "").split("_", 1)[0].upper()
    if token in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"}:
        return token
    return "GET"


def path_param_names(url: str) -> list[str]:
    return PATH_PARAM_RE.findall(url or "")


def https_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "http":
        return urlunparse(parsed._replace(scheme="https"))
    return url


def fill_url(template: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    leftover = dict(arguments)
    filled = template
    for name in path_param_names(template):
        if name not in leftover:
            raise ValueError(f"missing path parameter {name}")
        filled = filled.replace("{" + name + "}", str(leftover.pop(name)))
    return https_url(filled), leftover


def cache_key(method: str, url: str, params: Mapping[str, Any]) -> str:
    payload = {"method": method, "url": url, "params": dict(params)}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


class TmdbClient:
    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | None = None,
        cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("TMDB_API_KEY is empty.")
        self.api_key = api_key
        self._explicit_session = session
        self._local = threading.local()
        self.cache = cache if cache is not None else {}
        self._lock = threading.Lock()

    def _get_session(self) -> requests.Session:
        if self._explicit_session is not None:
            return self._explicit_session
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def call(
        self,
        *,
        url_template: str,
        arguments: Mapping[str, Any],
        method: str = "GET",
    ) -> dict[str, Any]:
        url, query = fill_url(url_template, arguments)
        params = {key: value for key, value in query.items() if value is not None}
        params["api_key"] = self.api_key
        key = cache_key(method, url, {k: v for k, v in params.items() if k != "api_key"})
        with self._lock:
            hit = self.cache.get(key)
        if hit is not None:
            return {**hit, "source": "cache"}
        response = self._get_session().request(
            method,
            url,
            params=params,
            timeout=TMDB_TIMEOUT,
        )
        body = (response.text or "")[:RESPONSE_CHARS]
        row = {
            "error": "" if response.ok else f"HTTP {response.status_code}",
            "status": int(response.status_code),
            "response": body,
            "url": url,
        }
        with self._lock:
            existing = self.cache.get(key)
            if existing is None:
                self.cache[key] = row
                existing = row
                source = "live"
            else:
                source = "cache"
        return {**existing, "source": source}


class SpotifyClient:
    """Spotify Web API. Catalog works with client credentials; /me needs a refresh token."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        refresh_token: str = "",
        session: requests.Session | None = None,
        cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not client_id or not client_secret:
            raise ValueError("Spotify client id/secret is empty.")
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self._explicit_session = session
        self._local = threading.local()
        self.cache = cache if cache is not None else {}
        self._lock = threading.Lock()
        self._access = ""
        self._expires_at = 0.0

    def _get_session(self) -> requests.Session:
        if self._explicit_session is not None:
            return self._explicit_session
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _access_token(self) -> str:
        with self._lock:
            if self._access and time.time() < self._expires_at - 60:
                return self._access
            if self.refresh_token:
                data = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                }
            else:
                data = {"grant_type": "client_credentials"}
            response = self._get_session().post(
                SPOTIFY_TOKEN_URL,
                data=data,
                auth=(self.client_id, self.client_secret),
                timeout=SPOTIFY_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            token = str(payload.get("access_token") or "")
            if not token:
                raise ValueError("Spotify token response has no access_token.")
            self._access = token
            self._expires_at = time.time() + float(payload.get("expires_in") or 3600)
            return token

    def call(
        self,
        *,
        url_template: str,
        arguments: Mapping[str, Any],
        method: str = "GET",
    ) -> dict[str, Any]:
        url, leftover = fill_url(url_template, arguments)
        params = {key: value for key, value in leftover.items() if value is not None}
        key = cache_key(method, url, params)
        with self._lock:
            hit = self.cache.get(key)
        if hit is not None:
            return {**hit, "source": "cache"}
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": SPOTIFY_TIMEOUT,
        }
        verb = (method or "GET").upper()
        if verb in {"POST", "PUT", "DELETE", "PATCH"} and params:
            kwargs["json"] = params
        else:
            kwargs["params"] = params
        response = self._get_session().request(verb, url, **kwargs)
        body = (response.text or "")[:RESPONSE_CHARS]
        row = {
            "error": "" if response.ok else f"HTTP {response.status_code}",
            "status": int(response.status_code),
            "response": body,
            "url": url,
        }
        with self._lock:
            existing = self.cache.get(key)
            if existing is None:
                self.cache[key] = row
                existing = row
                source = "live"
            else:
                source = "cache"
        return {**existing, "source": source}
