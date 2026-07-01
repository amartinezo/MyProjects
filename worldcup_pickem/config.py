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

# (league_id, season) pairs pulled to build the training set. More/relevant
# international matches => better team-strength estimates. World Cup Qualifiers
# and continental cups have their own league ids on API-Football; add them here
# to enrich the model (ids vary, so they are left commented as a starting point).
TRAINING_SOURCES: list[tuple[int, int]] = [
    (WORLD_CUP_LEAGUE_ID, 2026),   # current World Cup (in progress)
    (WORLD_CUP_LEAGUE_ID, 2022),   # previous World Cups for baseline strength
    (WORLD_CUP_LEAGUE_ID, 2018),
    # (32, 2025),   # e.g. World Cup Qualifiers - CONMEBOL (verify id on your plan)
    # (29, 2025),   # e.g. World Cup Qualifiers - Africa
    # (5,  2025),   # e.g. UEFA Nations League
]

# --------------------------------------------------------------------------
# Pick 'Em scoring rules (knockout stage)
# --------------------------------------------------------------------------
# Points are totals, not increments: you get the best category you hit.
PTS_WINNER = int(os.environ.get("PTS_WINNER", "3"))       # right team advances
PTS_GOAL_DIFF = int(os.environ.get("PTS_GOAL_DIFF", "5"))  # + right goal difference
PTS_EXACT = int(os.environ.get("PTS_EXACT", "8"))          # + exact 90' scoreline

# How the pool defines the "winning team" you pick for the 3 points:
#   "advances"   -> the team that progresses (incl. extra time / penalties)
#   "result_90"  -> the team ahead at the end of regulation (a 90' draw scores 0)
# You confirmed the *score* you enter is the 90' scoreline; this only affects
# the 3-point "winner" leg. Default assumes a knockout pool where you pick who
# goes through. Flip to "result_90" if your pool scores the winner at 90'.
WINNER_DEFINITION = os.environ.get("WINNER_DEFINITION", "advances")

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

# When blending the model with bookmaker odds at prediction time, how much
# weight to give the market (0 = ignore odds, 1 = trust odds entirely).
MARKET_BLEND_WEIGHT = float(os.environ.get("MARKET_BLEND_WEIGHT", "0.5"))
