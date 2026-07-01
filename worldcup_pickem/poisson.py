"""Turn two expected-goal rates into a 90' score-probability grid.

``score_grid`` returns a matrix P where P[i, j] = P(home scores i, away scores j)
at the end of regulation. An optional Dixon-Coles adjustment nudges the very
low scores to better match real football (draws are a touch more common than
independent Poisson implies).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

import config


def _dixon_coles_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles correction factor for the four lowest scorelines."""
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_grid(
    lh: float,
    la: float,
    max_goals: int | None = None,
    rho: float | None = None,
) -> np.ndarray:
    """Probability grid for scorelines 0..max_goals for each side."""
    max_goals = config.MAX_GOALS if max_goals is None else max_goals
    rho = config.DIXON_COLES_RHO if rho is None else rho

    home_p = poisson.pmf(np.arange(max_goals + 1), lh)
    away_p = poisson.pmf(np.arange(max_goals + 1), la)
    grid = np.outer(home_p, away_p)

    if rho != 0.0:
        for i in (0, 1):
            for j in (0, 1):
                grid[i, j] *= _dixon_coles_tau(i, j, lh, la, rho)

    grid /= grid.sum()  # renormalise (truncation + DC change the total slightly)
    return grid


def outcome_probs(grid: np.ndarray) -> tuple[float, float, float]:
    """Return (P home win, P draw, P away win) at 90' from a score grid."""
    p_home = float(np.tril(grid, -1).sum())  # i > j
    p_away = float(np.triu(grid, 1).sum())   # i < j
    p_draw = float(np.trace(grid))           # i == j
    return p_home, p_draw, p_away


def modal_score(grid: np.ndarray) -> tuple[int, int, float]:
    """Most likely exact scoreline and its probability."""
    i, j = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return int(i), int(j), float(grid[i, j])
