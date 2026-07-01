"""Expected-points optimiser for the 3 / 5 / 8 Pick 'Em (draws included).

The scoring is one clean rule applied to your picked 90' scoreline vs the actual
90' scoreline:

    outcome (home win / draw / away win) correct  -> 3
    ...and goal difference also correct           -> 5
    ...and the exact score correct                -> 8

A correct **draw** pick always nails the goal difference (0), so it scores at
least 5 (never just 3), and 8 for the exact draw score. This module finds the
scoreline that maximises expected points, and also surfaces a higher-variance
"swing" alternative for when you need to differentiate from the field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from poisson import modal_score, most_likely_by_outcome, outcome_probs, top_scores


def outcome_of(hg: int, ag: int) -> str:
    return "home" if hg > ag else "away" if hg < ag else "draw"


def advance_probabilities(grid: np.ndarray, tiebreak_home: float = 0.5) -> tuple[float, float]:
    """P(home advances), P(away advances) - informational (ET/pens on a draw)."""
    p_home_90, p_draw_90, p_away_90 = outcome_probs(grid)
    p_home_adv = p_home_90 + p_draw_90 * tiebreak_home
    p_away_adv = p_away_90 + p_draw_90 * (1.0 - tiebreak_home)
    return p_home_adv, p_away_adv


def expected_points(grid: np.ndarray, pick_hg: int, pick_ag: int) -> float:
    """Expected Pick 'Em points for entering the scoreline ``pick_hg-pick_ag``."""
    picked = outcome_of(pick_hg, pick_ag)
    pick_diff = pick_hg - pick_ag
    ev = 0.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            p = grid[i, j]
            if p <= 0 or outcome_of(i, j) != picked:
                continue
            if i == pick_hg and j == pick_ag:
                pts = config.PTS_EXACT
            elif (i - j) == pick_diff:      # true for every draw-vs-draw
                pts = config.PTS_GOAL_DIFF
            else:
                pts = config.PTS_WINNER
            ev += p * pts
    return ev


def score_pick(pick_hg: int, pick_ag: int, actual_hg: int, actual_ag: int) -> int:
    """Actual points the scoreline ``pick`` earns against ``actual`` (0/3/5/8)."""
    if outcome_of(pick_hg, pick_ag) != outcome_of(actual_hg, actual_ag):
        return 0
    if pick_hg == actual_hg and pick_ag == actual_ag:
        return config.PTS_EXACT
    if (pick_hg - pick_ag) == (actual_hg - actual_ag):
        return config.PTS_GOAL_DIFF
    return config.PTS_WINNER


@dataclass
class Recommendation:
    outcome: str                 # "home" / "draw" / "away"
    pick_name: str               # team name, or "Draw"
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
    top_scores: list = field(default_factory=list)
    outcome_scores: dict = field(default_factory=dict)
    # Higher-variance alternative that differs in outcome from the safe pick.
    swing: dict | None = None
    # Bookmaker signals (set by predict.py when odds are available):
    market_1x2: tuple | None = None        # (P home, P draw, P away)
    market_scores: list | None = None      # [(home, away, prob), ...]

    def _label(self, outcome: str, hg: int, ag: int) -> str:
        if outcome == "draw":
            return f"Draw {hg}-{ag} (90')"
        name = self.home_name if outcome == "home" else self.away_name
        return f"{name} to win, {hg}-{ag} (90')"

    def pretty_pick(self) -> str:
        return self._label(self.outcome, self.hg, self.ag)

    def pretty_swing(self) -> str:
        if not self.swing:
            return ""
        s = self.swing
        return self._label(s["outcome"], s["hg"], s["ag"])


def optimize_pick(
    grid: np.ndarray,
    *,
    home_name: str,
    away_name: str,
    lambda_home: float,
    lambda_away: float,
    top_k: int = 3,
) -> Recommendation:
    """Find the max-EV scoreline plus a higher-variance swing alternative."""
    candidates: list[tuple[float, int, int]] = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            candidates.append((expected_points(grid, i, j), i, j))
    candidates.sort(key=lambda c: c[0], reverse=True)

    best_ev, best_i, best_j = candidates[0]
    best_outcome = outcome_of(best_i, best_j)

    def name_of(outcome: str) -> str:
        return {"home": home_name, "away": away_name, "draw": "Draw"}[outcome]

    # Swing = highest-EV candidate whose OUTCOME differs from the safe pick,
    # i.e. a genuine way to differentiate from chalk. We report how much EV it
    # costs and its chance of landing the exact score (the 8-point jackpot).
    swing = None
    for ev, i, j in candidates:
        if outcome_of(i, j) != best_outcome:
            swing = {
                "outcome": outcome_of(i, j),
                "name": name_of(outcome_of(i, j)),
                "hg": i, "ag": j,
                "ev": ev,
                "exact_prob": float(grid[i, j]),
                "ev_drop": best_ev - ev,
            }
            break

    p_home_adv, p_away_adv = advance_probabilities(
        grid, float(np.clip(lambda_home / (lambda_home + lambda_away), 0.35, 0.65))
        if (lambda_home + lambda_away) > 0 else 0.5,
    )
    p_home_90, p_draw_90, p_away_90 = outcome_probs(grid)
    mi, mj, mp = modal_score(grid)

    alternatives = [
        {"pick": self_label(name_of(outcome_of(i, j)), outcome_of(i, j), i, j),
         "expected_points": round(ev, 3)}
        for ev, i, j in candidates[1:top_k + 1]
    ]

    return Recommendation(
        outcome=best_outcome,
        pick_name=name_of(best_outcome),
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
        top_scores=top_scores(grid, n=6),
        outcome_scores=most_likely_by_outcome(grid),
        swing=swing,
    )


def self_label(name: str, outcome: str, hg: int, ag: int) -> str:
    """Compact pick label used in the alternatives list."""
    if outcome == "draw":
        return f"Draw {hg}-{ag}"
    return f"{name} {hg}-{ag}"
