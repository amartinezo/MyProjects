"""Glue: fixture -> features -> lambdas -> (optional odds blend) -> best pick."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from api_client import APIFootball
from features import TeamState, match_feature_rows
from model import GoalsModel
from poisson import outcome_probs, score_grid
from scoring import Recommendation, optimize_pick


def _implied_1x2(client: APIFootball, fixture_id: int) -> tuple[float, float, float] | None:
    """Average bookmaker 1X2 as de-vigged (home, draw, away) probabilities."""
    try:
        rows = client.odds(fixture_id)
    except Exception:
        return None
    if not rows:
        return None

    hs, ds, as_ = [], [], []
    for entry in rows:
        for bm in entry.get("bookmakers", []):
            for bet in bm.get("bets", []):
                if bet.get("name") != "Match Winner":
                    continue
                vals = {v["value"]: float(v["odd"]) for v in bet.get("values", [])}
                if {"Home", "Draw", "Away"} <= vals.keys():
                    ph, pd_, pa = 1 / vals["Home"], 1 / vals["Draw"], 1 / vals["Away"]
                    tot = ph + pd_ + pa  # remove the overround
                    hs.append(ph / tot); ds.append(pd_ / tot); as_.append(pa / tot)
    if not hs:
        return None
    return float(np.mean(hs)), float(np.mean(ds)), float(np.mean(as_))


def parse_correct_score(rows: list[dict], top_n: int = 6) -> list[tuple[int, int, float]]:
    """Average bookmaker 'Correct Score' odds -> de-vigged (home, away, prob).

    Returns the top ``top_n`` most likely scorelines the market implies.
    """
    probs: dict[tuple[int, int], list[float]] = {}
    for entry in rows:
        for bm in entry.get("bookmakers", []):
            for bet in bm.get("bets", []):
                if bet.get("name") != "Correct Score":
                    continue
                book = {}
                for v in bet.get("values", []):
                    label = str(v.get("value", "")).replace(":", "-").strip()
                    if "-" not in label:
                        continue
                    try:
                        h, a = (int(x) for x in label.split("-"))
                        odd = float(v["odd"])
                    except (ValueError, KeyError):
                        continue
                    if odd > 0:
                        book[(h, a)] = 1.0 / odd
                tot = sum(book.values())          # de-vig within this bookmaker
                for k, p in book.items():
                    probs.setdefault(k, []).append(p / tot if tot > 0 else 0.0)
    if not probs:
        return []
    avg = [(h, a, float(np.mean(ps))) for (h, a), ps in probs.items()]
    avg.sort(key=lambda t: t[2], reverse=True)
    return avg[:top_n]


def _market_correct_score(client: APIFootball, fixture_id: int) -> list[tuple[int, int, float]]:
    """Fetch and parse the bookmaker Correct Score market (best-effort)."""
    bet_id = client.correct_score_bet_id()
    if bet_id is None:
        return []
    try:
        rows = client.odds(fixture_id, bet=bet_id)
    except Exception:
        return []
    return parse_correct_score(rows)


def _blend_grid_to_market(grid: np.ndarray, market: tuple[float, float, float], w: float) -> np.ndarray:
    """Rescale the grid's win/draw/loss mass toward market probabilities.

    Scoreline *shape* within each outcome region is preserved; only the total
    mass of home-win / draw / away-win regions is shifted toward the market.
    """
    m_home, m_draw, m_away = market
    p_home, p_draw, p_away = outcome_probs(grid)
    target_home = (1 - w) * p_home + w * m_home
    target_draw = (1 - w) * p_draw + w * m_draw
    target_away = (1 - w) * p_away + w * m_away

    out = grid.copy()
    n = grid.shape[0]
    for i in range(n):
        for j in range(grid.shape[1]):
            if i > j and p_home > 0:
                out[i, j] *= target_home / p_home
            elif i < j and p_away > 0:
                out[i, j] *= target_away / p_away
            elif i == j and p_draw > 0:
                out[i, j] *= target_draw / p_draw
    s = out.sum()
    return out / s if s > 0 else grid


def predict_fixture(
    model: GoalsModel,
    state: TeamState,
    meta: dict,
    *,
    client: APIFootball | None = None,
    use_odds: bool = True,
) -> Recommendation:
    """Produce a full Pick 'Em recommendation for one fixture.

    ``meta`` needs: home_id, away_id, home, away, neutral, knockout, date and
    optionally fixture_id (for odds).
    """
    xh, xa = match_feature_rows(state, meta)
    lh = float(model.predict_lambda(xh)[0])
    la = float(model.predict_lambda(xa)[0])

    grid = score_grid(lh, la)

    market_1x2 = None
    market_scores = None
    if use_odds and client is not None and meta.get("fixture_id"):
        market_1x2 = _implied_1x2(client, meta["fixture_id"])
        if market_1x2:
            grid = _blend_grid_to_market(grid, market_1x2, config.MARKET_BLEND_WEIGHT)
        market_scores = _market_correct_score(client, meta["fixture_id"]) or None

    rec = optimize_pick(
        grid,
        home_name=meta["home"],
        away_name=meta["away"],
        lambda_home=lh,
        lambda_away=la,
    )
    rec.market_1x2 = market_1x2
    rec.market_scores = market_scores
    return rec


def predict_fixtures(
    model: GoalsModel,
    state: TeamState,
    fixtures: pd.DataFrame,
    *,
    client: APIFootball | None = None,
    use_odds: bool = True,
) -> list[Recommendation]:
    recos = []
    for fx in fixtures.itertuples(index=False):
        meta = {
            "fixture_id": getattr(fx, "fixture_id", None),
            "home_id": fx.home_id,
            "away_id": fx.away_id,
            "home": fx.home,
            "away": fx.away,
            "neutral": getattr(fx, "neutral", True),
            "knockout": getattr(fx, "knockout", True),
            "date": getattr(fx, "date", None),
        }
        recos.append(
            predict_fixture(model, state, meta, client=client, use_odds=use_odds)
        )
    return recos
