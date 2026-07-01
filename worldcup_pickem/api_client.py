"""Thin, cached client for the API-Football v3 REST API.

Design goals:
  * Read the key from the environment (or a local .env) - never hard-coded.
  * Cache every response to disk so repeated morning runs are fast and cheap
    on your API quota. Results of played matches never change, so caching is safe.
  * Fail with clear, actionable messages (missing key, blocked host, quota).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

try:  # optional: load a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

import config


class APIError(RuntimeError):
    """Raised for any unrecoverable problem talking to API-Football."""


class APIFootball:
    """Minimal wrapper around the endpoints this project needs."""

    def __init__(self, key: str | None = None, host: str | None = None):
        # Re-read from config at construction so tests can monkeypatch env.
        self.key = key if key is not None else config.API_KEY
        self.host = host or config.API_HOST
        self.base = f"https://{self.host}"
        self.session = requests.Session()
        self._last_request_ts = 0.0

    # -- internal helpers --------------------------------------------------
    def _headers(self) -> dict[str, str]:
        # API-Football accepts either its native header or the RapidAPI ones.
        if "rapidapi" in self.host:
            return {"x-rapidapi-key": self.key, "x-rapidapi-host": self.host}
        return {"x-apisports-key": self.key}

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        blob = endpoint + json.dumps(params, sort_keys=True)
        digest = hashlib.sha1(blob.encode()).hexdigest()[:16]
        safe = endpoint.strip("/").replace("/", "_")
        return config.CACHE_DIR / f"{safe}__{digest}.json"

    def _throttle(self) -> None:
        # Be polite: cap at ~5 requests/second regardless of plan.
        elapsed = time.time() - self._last_request_ts
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)
        self._last_request_ts = time.time()

    # -- public request ----------------------------------------------------
    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        ttl_hours: float | None = None,
        paginate: bool = True,
    ) -> list[dict]:
        """GET an endpoint and return the flattened ``response`` list.

        Responses are cached on disk. ``ttl_hours`` overrides the default TTL
        (use a short TTL for volatile data like upcoming fixtures/odds).
        """
        params = dict(params or {})
        ttl = config.CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
        cache_file = self._cache_path(endpoint, params)

        if cache_file.exists():
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600.0
            if age_h <= ttl:
                return json.loads(cache_file.read_text())

        if not self.key:
            raise APIError(
                "No API key found. Set the API_FOOTBALL_KEY environment secret "
                "(recommended) or add it to a local .env file. See README."
            )

        all_rows: list[dict] = []
        page = 1
        while True:
            if page > 1:
                params["page"] = page
            data = self._request(endpoint, params)
            all_rows.extend(data.get("response", []))
            paging = data.get("paging", {}) or {}
            if not paginate or paging.get("current", 1) >= paging.get("total", 1):
                break
            page += 1

        cache_file.write_text(json.dumps(all_rows))
        return all_rows

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict:
        self._throttle()
        url = f"{self.base}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, headers=self._headers(), params=params, timeout=30)
        except requests.exceptions.SSLError as exc:  # pragma: no cover
            raise APIError(
                f"TLS error contacting {self.host}. If you are on Claude Code web, "
                f"the network policy may be blocking this host. Details: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise APIError(
                f"Network error contacting {self.host}: {exc}. On Claude Code web "
                f"this host must be allowlisted in the environment's network policy."
            ) from exc

        if resp.status_code == 403:
            raise APIError(
                f"403 from {self.host}. Either your key is wrong, or (on Claude "
                f"Code web) the host is not allowlisted by the network policy."
            )
        if resp.status_code == 429:
            raise APIError("429 Too Many Requests - API-Football daily quota hit.")
        if resp.status_code != 200:
            raise APIError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")

        payload = resp.json()
        errors = payload.get("errors")
        # API-Football returns 200 with an `errors` dict for auth/param problems.
        if errors and (errors if isinstance(errors, list) else any(errors.values())):
            raise APIError(f"API-Football error for {endpoint}: {errors}")
        return payload

    # -- convenience endpoints --------------------------------------------
    def status(self) -> dict:
        rows = self.get("status", ttl_hours=0)
        return rows[0] if rows else {}

    def leagues(self, **kw) -> list[dict]:
        # Rarely changes; cache for a week. Returns every league the plan sees,
        # each with its available seasons.
        return self.get("leagues", kw, ttl_hours=24 * 7)

    def fixtures(self, league: int, season: int, **kw) -> list[dict]:
        params = {"league": league, "season": season, **kw}
        return self.get("fixtures", params)

    def fixtures_by_date(self, date: str, **kw) -> list[dict]:
        # Short TTL: upcoming-fixture lists change as line-ups/times update.
        return self.get("fixtures", {"date": date, **kw}, ttl_hours=1)

    def odds(self, fixture_id: int, bet: int = 1) -> list[dict]:
        # bet=1 is the standard Match Winner (1X2) market on API-Football.
        return self.get("odds", {"fixture": fixture_id, "bet": bet}, ttl_hours=6)


def get_client() -> APIFootball:
    return APIFootball()
