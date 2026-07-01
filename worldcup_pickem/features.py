"""Feature engineering: Elo ratings, rolling form, rest days.

State is built chronologically so every feature for a match uses only
information available *before* kickoff (no leakage). The same row-builder is
used for training and for predicting upcoming fixtures, guaranteeing the
feature columns line up.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

import config

# Column order shared by training and inference. Keep in sync deliberately.
FEATURE_COLS = [
    "elo",
    "opp_elo",
    "elo_diff",
    "form_gf",
    "form_ga",
    "opp_form_gf",
    "opp_form_ga",
    "rest_days",
    "opp_rest_days",
    "is_home",
    "is_neutral",
    "is_knockout",
]


class TeamState:
    """Mutable running state used to derive pre-match features."""

    def __init__(self):
        self.elo: dict[int, float] = defaultdict(lambda: config.ELO_START)
        self.gf: dict[int, deque] = defaultdict(lambda: deque(maxlen=config.FORM_WINDOW))
        self.ga: dict[int, deque] = defaultdict(lambda: deque(maxlen=config.FORM_WINDOW))
        self.last_date: dict[int, pd.Timestamp] = {}
        self.name: dict[int, str] = {}

    # -- feature construction (read-only) ---------------------------------
    def team_row(self, team_id, opp_id, *, is_physical_home, neutral, knockout, date):
        gf = self.gf[team_id]
        ga = self.ga[team_id]
        ogf = self.gf[opp_id]
        oga = self.ga[opp_id]

        def rest(tid):
            if tid in self.last_date and date is not None:
                return (date - self.last_date[tid]).days
            return float("nan")

        avg = lambda d: (sum(d) / len(d)) if d else float("nan")
        return {
            "elo": self.elo[team_id],
            "opp_elo": self.elo[opp_id],
            "elo_diff": self.elo[team_id] - self.elo[opp_id],
            "form_gf": avg(gf),
            "form_ga": avg(ga),
            "opp_form_gf": avg(ogf),
            "opp_form_ga": avg(oga),
            "rest_days": rest(team_id),
            "opp_rest_days": rest(opp_id),
            "is_home": 1 if (is_physical_home and not neutral) else 0,
            "is_neutral": 1 if neutral else 0,
            "is_knockout": 1 if knockout else 0,
        }

    # -- state updates (write) --------------------------------------------
    def _elo_expected(self, home_id, away_id, neutral):
        ha = 0.0 if neutral else config.ELO_HOME_ADV
        rh = self.elo[home_id] + ha
        ra = self.elo[away_id]
        e_home = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
        return e_home

    def update(self, home_id, away_id, hg, ag, neutral, date, home_name=None, away_name=None):
        if home_name:
            self.name[home_id] = home_name
        if away_name:
            self.name[away_id] = away_name

        # Elo update using the 90' result.
        e_home = self._elo_expected(home_id, away_id, neutral)
        if hg > ag:
            s_home = 1.0
        elif hg < ag:
            s_home = 0.0
        else:
            s_home = 0.5
        gd = abs(hg - ag)
        if gd <= 1:
            g_mult = 1.0
        elif gd == 2:
            g_mult = 1.5
        else:
            g_mult = (11 + gd) / 8.0
        delta = config.ELO_K * g_mult * (s_home - e_home)
        self.elo[home_id] += delta
        self.elo[away_id] -= delta

        # Rolling form + schedule.
        self.gf[home_id].append(hg); self.ga[home_id].append(ag)
        self.gf[away_id].append(ag); self.ga[away_id].append(hg)
        self.last_date[home_id] = date
        self.last_date[away_id] = date


def build_training_features(df: pd.DataFrame) -> tuple[pd.DataFrame, TeamState]:
    """Walk matches in time order, emitting two team-perspective rows each.

    Returns the feature DataFrame (with a ``goals`` target column) and the
    final :class:`TeamState` for predicting future fixtures.
    """
    state = TeamState()
    rows: list[dict] = []
    for m in df.itertuples(index=False):
        # Features are read BEFORE this match is folded into the state.
        home_row = state.team_row(
            m.home_id, m.away_id, is_physical_home=True,
            neutral=m.neutral, knockout=m.knockout, date=m.date,
        )
        home_row["goals"] = m.hg
        away_row = state.team_row(
            m.away_id, m.home_id, is_physical_home=False,
            neutral=m.neutral, knockout=m.knockout, date=m.date,
        )
        away_row["goals"] = m.ag
        rows.append(home_row)
        rows.append(away_row)

        state.update(
            m.home_id, m.away_id, m.hg, m.ag, m.neutral, m.date,
            home_name=m.home, away_name=m.away,
        )

    feat = pd.DataFrame(rows)
    return feat, state


def match_feature_rows(state: TeamState, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (home_features, away_features) for a single upcoming fixture.

    ``meta`` needs: home_id, away_id, neutral, knockout, date.
    """
    home = state.team_row(
        meta["home_id"], meta["away_id"], is_physical_home=True,
        neutral=meta["neutral"], knockout=meta["knockout"], date=meta.get("date"),
    )
    away = state.team_row(
        meta["away_id"], meta["home_id"], is_physical_home=False,
        neutral=meta["neutral"], knockout=meta["knockout"], date=meta.get("date"),
    )
    return (
        pd.DataFrame([home])[FEATURE_COLS],
        pd.DataFrame([away])[FEATURE_COLS],
    )
