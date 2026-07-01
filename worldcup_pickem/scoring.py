"""Expected-points optimiser for the 3 / 5 / 8 knockout Pick 'Em.

The model gives a 90' score grid; this module answers the actual question you
face each morning: *which winner + scoreline should I enter to maximise my
expected points?* Because the payoff is lumpy (3 for the right team, 5 for the
right goal difference, 8 for the exact score), the best pick is NOT always the
single most likely score - this optimiser weighs every candidate against the
full distribution of outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from poisson import modal_score, outcome_probs


def advance_probabilities(grid: np.ndarray, tiebreak_home: float = 0.5) -> tuple[float, float]:
    """P(home advances), P(away advances), accounting for ET/penalties on a draw."""
    p_home_90, p_draw_90, p_away_90 = outcome_probs(grid)
    p_home_adv = p_home_90 + p_draw_90 * tiebreak_home
    p_away_adv = p_away_90 + p_draw_90 * (1.0 - tiebreak_home)
    return p_home_adv, p_away_adv


def expected_points(
    grid: np.ndarray,
    pick_winner: str,
    pick_hg: int,
    pick_ag: int,
    *,
    tiebreak_home: float = 0.5,
    winner_definition: str | None = None,
) -> float:
    """Expected Pick 'Em points for one (winner, scoreline) entry."""
    winner_definition = winner_definition or config.WINNER_DEFINITION
    pick_diff = pick_hg - pick_ag
    n = grid.shape[0]
    ev = 0.0

    for i in range(n):
        for j in range(grid.shape[1]):
            p = grid[i, j]
            if p <= 0:
                continue

            # Probability the picked winner is credited for this actual score.
            if winner_definition == "result_90":
                if i > j:
                    winner_prob = 1.0 if pick_winner == "home" else 0.0
                elif i < j:
                    winner_prob = 1.0 if pick_winner == "away" else 0.0
                else:
                    winner_prob = 0.0  # 90' draw => no winner credited
            else:  # "advances"
                if i > j:
                    winner_prob = 1.0 if pick_winner == "home" else 0.0
                elif i < j:
                    winner_prob = 1.0 if pick_winner == "away" else 0.0
                else:  # draw at 90' -> decided by ET/penalties
                    winner_prob = tiebreak_home if pick_winner == "home" else (1.0 - tiebreak_home)

            if winner_prob <= 0:
                continue

            if pick_hg == i and pick_ag == j:
                pts = config.PTS_EXACT
            elif pick_diff == (i - j):
                pts = config.PTS_GOAL_DIFF
            else:
                pts = config.PTS_WINNER

            ev += p * winner_prob * pts
    return ev


@dataclass
class Recommendation:
    winner: str                 # "home" or "away"
    winner_name: str
    hg: int
    ag: int
    expected_points: float
    p_home_adv: float
    p_away_adv: float
    modal_hg: int
    modal_ag: int
    modal_prob: float
    p_home_90: float
    p_draw_90: float
    p_away_90: float
    lambda_home: float
    lambda_away: float
    home_name: str = ""
    away_name: str = ""
    alternatives: list = field(default_factory=list)

    def pretty_pick(self) -> str:
        return f"{self.winner_name} to advance, {self.hg}-{self.ag} (90')"


def optimize_pick(
    grid: np.ndarray,
    *,
    home_name: str,
    away_name: str,
    lambda_home: float,
    lambda_away: float,
    tiebreak_home: float = 0.5,
    winner_definition: str | None = None,
    top_k: int = 3,
) -> Recommendation:
    """Search all sensible (winner, scoreline) entries for the max-EV pick."""
    n = grid.shape[0]
    candidates: list[tuple[float, str, int, int]] = []

    for winner in ("home", "away"):
        for i in range(n):
            for j in range(grid.shape[1]):
                # Only consider scorelines consistent with the picked winner,
                # so the goal-diff / exact bonuses are actually attainable.
                if winner == "home" and i < j:
                    continue
                if winner == "away" and i > j:
                    continue
                ev = expected_points(
                    grid, winner, i, j,
                    tiebreak_home=tiebreak_home,
                    winner_definition=winner_definition,
                )
                candidates.append((ev, winner, i, j))

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_ev, best_w, best_i, best_j = candidates[0]

    p_home_adv, p_away_adv = advance_probabilities(grid, tiebreak_home)
    p_home_90, p_draw_90, p_away_90 = outcome_probs(grid)
    mi, mj, mp = modal_score(grid)

    name = {"home": home_name, "away": away_name}
    alternatives = [
        {"pick": f"{name[w]} {i}-{j}", "expected_points": round(ev, 3)}
        for ev, w, i, j in candidates[1:top_k + 1]
    ]

    return Recommendation(
        winner=best_w,
        winner_name=name[best_w],
        hg=best_i,
        ag=best_j,
        expected_points=best_ev,
        p_home_adv=p_home_adv,
        p_away_adv=p_away_adv,
        modal_hg=mi,
        modal_ag=mj,
        modal_prob=mp,
        p_home_90=p_home_90,
        p_draw_90=p_draw_90,
        p_away_90=p_away_90,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        home_name=home_name,
        away_name=away_name,
        alternatives=alternatives,
    )
