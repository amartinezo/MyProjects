"""Central configuration for the World Cup Pick 'Em predictor.

Everything you might reasonably want to tweak lives here so the rest of the
code stays declarative. Values can be overridden with environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # raw API responses (gitignored)
MODEL_DIR = DATA_DIR / "models"         # trained artifacts (gitignored)
for _d in (DATA_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# API-Football
# --------------------------------------------------------------------------
API_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_HOST = os.environ.get("API_FOOTBALL_HOST", "v3.football.api-sports.io")
# Cached API responses older than this (hours) are re-fetched. Match results
# never change once played, so a long TTL is fine; fixtures/odds change, so
# `data.py` uses a short TTL for those explicitly.
CACHE_TTL_HOURS = float(os.environ.get("API_CACHE_TTL_HOURS", "24"))

# --------------------------------------------------------------------------
# What to train on
# --------------------------------------------------------------------------
# API-Football league id for the FIFA World Cup. The 48-team 2026 edition uses
# the same league id (1) as previous World Cups, distinguished by season.
WORLD_CUP_LEAGUE_ID = int(os.environ.get("WORLD_CUP_LEAGUE_ID", "1"))
WORLD_CUP_SEASON = int(os.environ.get("WORLD_CUP_SEASON", "2026"))

# Timezone used to bucket "today's" fixtures. API-Football filters the /fixtures
# `date` param by UTC calendar date unless a `timezone` is supplied, which would
# otherwise misfile evening kickoffs in the Americas: a game at 20:00 local
# (00:00 UTC next day) would drop off "today" and a late game the night before
# would wrongly appear. Set to your local IANA timezone.
LOCAL_TIMEZONE = os.environ.get("LOCAL_TIMEZONE", "America/New_York")

# We resolve training competitions DYNAMICALLY by name instead of hard-coding
# API-Football league ids (which vary by plan and are easy to get wrong). At
# build time `data.resolve_sources` queries /leagues, keeps only international
# ("World" country) competitions whose name matches a spec, and expands them to
# the seasons your plan actually covers. Missing competitions are skipped.
USE_DYNAMIC_SOURCES = os.environ.get("USE_DYNAMIC_SOURCES", "1") == "1"

# Each spec: name pattern, match mode ("exact" or "contains"), and seasons to try.
# These are the highest-signal international matches for estimating current
# national-team strength.
COMPETITION_SPECS: list[dict] = [
    {"name": "World Cup", "match": "exact", "seasons": [2026, 2022, 2018]},
    {"name": "World Cup - Qualification", "match": "contains",
     "seasons": [2026, 2025, 2024, 2023]},
    {"name": "UEFA Nations League", "match": "exact", "seasons": [2025, 2024, 2023, 2022]},
    {"name": "CONCACAF Nations League", "match": "exact", "seasons": [2024, 2023]},
    {"name": "Euro Championship", "match": "exact", "seasons": [2024]},
    {"name": "Copa America", "match": "exact", "seasons": [2024]},
    {"name": "Africa Cup of Nations", "match": "exact", "seasons": [2023]},
    {"name": "Friendlies", "match": "exact", "seasons": [2026, 2025, 2024, 2023]},
]

# Explicit (league_id, season) fallback, used only if dynamic resolution is off
# or returns nothing (e.g. /leagues unavailable on your plan).
TRAINING_SOURCES: list[tuple[int, int]] = [
    (WORLD_CUP_LEAGUE_ID, 2026),   # current World Cup (in progress)
    (WORLD_CUP_LEAGUE_ID, 2022),   # previous World Cups for baseline strength
    (WORLD_CUP_LEAGUE_ID, 2018),
]

# --------------------------------------------------------------------------
# Pick 'Em scoring rules (knockout stage)
# --------------------------------------------------------------------------
# Points are totals, not increments: you get the best category you hit.
PTS_WINNER = int(os.environ.get("PTS_WINNER", "3"))       # right team advances
PTS_GOAL_DIFF = int(os.environ.get("PTS_GOAL_DIFF", "5"))  # + right goal difference
PTS_EXACT = int(os.environ.get("PTS_EXACT", "8"))          # + exact 90' scoreline

# How the pool defines the "winning team" you pick for the 3 points:
#   "result_90"  -> the team ahead at the end of regulation (a 90' draw scores 0)
#   "advances"   -> the team that progresses (incl. extra time / penalties)
# The *score* you enter is always the 90' scoreline; this only affects the
# 3-point "winner" leg. Confirmed with the pool: the winner is decided at the
# end of regulation, so the default is "result_90".
WINNER_DEFINITION = os.environ.get("WINNER_DEFINITION", "result_90")

# --------------------------------------------------------------------------
# Model / Poisson settings
# --------------------------------------------------------------------------
MAX_GOALS = int(os.environ.get("MAX_GOALS", "8"))   # score-grid dimension (0..MAX_GOALS)
FORM_WINDOW = int(os.environ.get("FORM_WINDOW", "6"))  # matches in rolling form
# Dixon-Coles low-score correlation adjustment. 0.0 = plain independent Poisson.
# Small negative values slightly boost 0-0/1-1 and dampen 1-0/0-1, matching real
# football. Kept conservative; set to 0.0 to disable.
DIXON_COLES_RHO = float(os.environ.get("DIXON_COLES_RHO", "-0.05"))

# Elo parameters (World Football Elo style)
ELO_START = 1500.0
ELO_K = 40.0            # high K: international tournaments are high-importance
ELO_HOME_ADV = 65.0    # rating points added to a genuine (non-neutral) home side
# Seed each team's starting Elo from approximate known ratings (elo_seed.py)
# instead of a flat 1500. Cuts cold-start noise for teams with little history.
ELO_SEED_ENABLED = os.environ.get("ELO_SEED_ENABLED", "1") == "1"

# When blending the model with bookmaker odds at prediction time, how much
# weight to give the market (0 = ignore odds, 1 = trust odds entirely). While the
# tournament sample is thin the market is the more reliable signal, so we lean on
# it a bit more; lower this toward ~0.4 later as the model accumulates data.
MARKET_BLEND_WEIGHT = float(os.environ.get("MARKET_BLEND_WEIGHT", "0.6"))
