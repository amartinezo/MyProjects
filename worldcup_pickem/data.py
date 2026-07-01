"""Fetch and normalise match data from API-Football into tidy DataFrames.

Key modelling decision: goals are taken from ``score.fulltime`` = the scoreline
at the end of regulation (90'), which is exactly what the Pick 'Em scores. Extra
time and penalties are deliberately ignored for the score itself; who *advanced*
is derived separately in ``scoring.py``.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

import config
from api_client import APIFootball, get_client

# Match statuses that mean the 90' result is final and usable for training.
FINISHED = {"FT", "AET", "PEN"}
# Host nations of the 2026 World Cup - the only WC sides that can be non-neutral.
HOST_NATIONS = {"USA", "United States", "Canada", "Mexico"}

_KO_MARKERS = ("round of", "final", "quarter", "semi", "3rd", "play-off", "knockout")


def _is_knockout(round_name: str | None) -> bool:
    r = (round_name or "").lower()
    return any(m in r for m in _KO_MARKERS)


def _row_from_fixture(fx: dict, *, world_cup_league: int) -> dict | None:
    """Flatten one API-Football fixture item to a flat dict (or None to skip)."""
    fixture = fx.get("fixture", {})
    league = fx.get("league", {})
    teams = fx.get("teams", {})
    score = fx.get("score", {})

    home = teams.get("home", {})
    away = teams.get("away", {})
    ft = score.get("fulltime", {}) or {}

    league_id = league.get("id")
    round_name = league.get("round")
    home_name = home.get("name")
    away_name = away.get("name")

    # World Cup matches are treated as neutral unless a host nation is the
    # nominal home side; other competitions keep real home advantage.
    if league_id == world_cup_league:
        neutral = home_name not in HOST_NATIONS
    else:
        neutral = False

    return {
        "fixture_id": fixture.get("id"),
        "date": pd.to_datetime(fixture.get("date"), utc=True),
        "league_id": league_id,
        "season": league.get("season"),
        "round": round_name,
        "knockout": _is_knockout(round_name),
        "status": (fixture.get("status") or {}).get("short"),
        "home_id": home.get("id"),
        "home": home_name,
        "away_id": away.get("id"),
        "away": away_name,
        "neutral": neutral,
        # 90' regulation goals (may be None for not-yet-played fixtures):
        "hg": (ft.get("home")),
        "ag": (ft.get("away")),
    }


def match_specs_to_sources(leagues_response: list[dict], specs: list[dict]) -> list[tuple[int, int]]:
    """Pure resolver: map competition specs to (league_id, season) pairs.

    Only international competitions (country == "World") are considered, so club
    leagues with similar names can't sneak in. A spec's requested seasons are
    intersected with the seasons the league actually offers on this plan.
    """
    sources: list[tuple[int, int]] = []
    for entry in leagues_response:
        league = entry.get("league", {}) or {}
        country = (entry.get("country", {}) or {}).get("name")
        if country != "World":
            continue
        name = (league.get("name") or "")
        lid = league.get("id")
        if lid is None:
            continue
        available = {s.get("year") for s in (entry.get("seasons") or [])}
        for spec in specs:
            pat = spec["name"]
            hit = (name == pat) if spec.get("match") == "exact" else (pat.lower() in name.lower())
            if not hit:
                continue
            for season in spec["seasons"]:
                if season in available and (lid, season) not in sources:
                    sources.append((lid, season))
    return sources


def resolve_sources(
    client: APIFootball | None = None,
    specs: list[dict] | None = None,
) -> list[tuple[int, int]]:
    """Query /leagues and resolve configured specs to (league_id, season) pairs."""
    client = client or get_client()
    specs = specs or config.COMPETITION_SPECS
    return match_specs_to_sources(client.leagues(), specs)


def build_match_frame(
    client: APIFootball | None = None,
    sources: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Return a chronologically sorted DataFrame of *finished* matches."""
    client = client or get_client()
    if sources is None:
        if config.USE_DYNAMIC_SOURCES:
            try:
                sources = resolve_sources(client)
            except Exception:
                sources = []
        # Fall back to explicit ids if dynamic resolution is off or empty.
        if not sources:
            sources = config.TRAINING_SOURCES

    rows: list[dict] = []
    for league_id, season in sources:
        for fx in client.fixtures(league_id, season):
            row = _row_from_fixture(fx, world_cup_league=config.WORLD_CUP_LEAGUE_ID)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["status"].isin(FINISHED)].copy()
    df = df.dropna(subset=["hg", "ag", "home_id", "away_id"])
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def upcoming_fixtures(
    date: str | None = None,
    client: APIFootball | None = None,
) -> pd.DataFrame:
    """Return World Cup fixtures scheduled on ``date`` (YYYY-MM-DD, default today)."""
    client = client or get_client()
    date = date or dt.date.today().isoformat()

    raw = client.fixtures(
        config.WORLD_CUP_LEAGUE_ID, config.WORLD_CUP_SEASON, date=date
    )
    rows = [
        _row_from_fixture(fx, world_cup_league=config.WORLD_CUP_LEAGUE_ID)
        for fx in raw
    ]
    df = pd.DataFrame([r for r in rows if r])
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df
