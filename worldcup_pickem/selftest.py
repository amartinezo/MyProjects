"""Offline self-test - verifies the maths and full non-network pipeline.

Run:  python selftest.py
Needs NO API key and NO internet. It fabricates matches, trains the real
model, and checks the Poisson grid + expected-points optimiser behave sanely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import elo_seed
from data import match_specs_to_sources
from features import TeamState, build_training_features, match_feature_rows
from model import GoalsModel
from poisson import outcome_probs, score_grid
from predict import predict_fixture
from scoring import advance_probabilities, expected_points, optimize_pick


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_poisson():
    print("Poisson grid:")
    g = score_grid(1.6, 1.1)
    _check("grid sums to ~1", abs(g.sum() - 1.0) < 1e-9)
    ph, pd_, pa = outcome_probs(g)
    _check("outcome probs sum to ~1", abs(ph + pd_ + pa - 1.0) < 1e-9)
    _check("stronger side favoured", ph > pa)


def test_scoring():
    print("Scoring / optimiser:")
    # Lopsided game: home much stronger.
    g = score_grid(2.4, 0.6)
    p_home_adv, p_away_adv = advance_probabilities(g, tiebreak_home=0.6)
    _check("advance probs sum to ~1", abs(p_home_adv + p_away_adv - 1.0) < 1e-9)
    _check("favourite more likely to advance", p_home_adv > p_away_adv)

    rec = optimize_pick(g, home_name="Alpha", away_name="Beta",
                        lambda_home=2.4, lambda_away=0.6, tiebreak_home=0.6)
    _check("recommends the favourite", rec.winner == "home")
    _check("EV within [0, 8]", 0 <= rec.expected_points <= 8)

    # EV of best pick beats an obviously bad pick (underdog exact upset score).
    bad = expected_points(g, "away", 0, 3)
    _check("optimiser beats a bad pick", rec.expected_points > bad)

    # Score distribution fields.
    _check("top_scores sorted descending",
           all(rec.top_scores[k][2] >= rec.top_scores[k + 1][2]
               for k in range(len(rec.top_scores) - 1)))
    _check("outcome_scores has all three results",
           {"home", "draw", "away"} <= rec.outcome_scores.keys())
    _check("conditional prob <= 1",
           all(v[3] <= 1.0 + 1e-9 for v in rec.outcome_scores.values()))


def _fake_matches(n_teams=16, n_matches=600, seed=0):
    rng = np.random.default_rng(seed)
    strengths = rng.normal(0, 0.5, n_teams)  # latent attack strength per team
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for k in range(n_matches):
        h, a = rng.choice(n_teams, size=2, replace=False)
        lh = np.exp(0.2 + strengths[h] - 0.5 * strengths[a])
        la = np.exp(0.1 + strengths[a] - 0.5 * strengths[h])
        rows.append({
            "home_id": int(h), "away_id": int(a),
            "home": f"T{h}", "away": f"T{a}",
            "hg": int(rng.poisson(lh)), "ag": int(rng.poisson(la)),
            "neutral": True, "knockout": False,
            "date": start + pd.Timedelta(days=k),
        })
    return pd.DataFrame(rows)


def test_end_to_end():
    print("End-to-end (no network):")
    matches = _fake_matches()
    feat, state = build_training_features(matches)
    _check("two feature rows per match", len(feat) == 2 * len(matches))

    model = GoalsModel().fit(feat)
    xh, xa = match_feature_rows(state, {
        "home_id": 0, "away_id": 1, "neutral": True, "knockout": True,
        "date": matches["date"].max(),
    })
    lh = float(model.predict_lambda(xh)[0])
    la = float(model.predict_lambda(xa)[0])
    _check("lambdas positive & bounded", 0 < lh < 8 and 0 < la < 8)

    rec = predict_fixture(model, state, {
        "home_id": 0, "away_id": 1, "home": "T0", "away": "T1",
        "neutral": True, "knockout": True, "date": matches["date"].max(),
    }, client=None, use_odds=False)
    _check("recommendation produced", rec.winner in ("home", "away"))
    print(f"     e.g. pick: {rec.pretty_pick()}  (EV {rec.expected_points:.2f})")


def test_league_resolver():
    print("Dynamic league resolver:")
    # A realistic slice of an API-Football /leagues response.
    fake = [
        {"league": {"id": 1, "name": "World Cup"}, "country": {"name": "World"},
         "seasons": [{"year": 2026}, {"year": 2022}, {"year": 2018}]},
        {"league": {"id": 32, "name": "World Cup - Qualification Europe"},
         "country": {"name": "World"}, "seasons": [{"year": 2025}, {"year": 2024}]},
        {"league": {"id": 5, "name": "UEFA Nations League"}, "country": {"name": "World"},
         "seasons": [{"year": 2024}]},
        # Club league that must NOT match despite "Cup" in the name.
        {"league": {"id": 45, "name": "FA Cup"}, "country": {"name": "England"},
         "seasons": [{"year": 2024}]},
    ]
    specs = [
        {"name": "World Cup", "match": "exact", "seasons": [2026, 2022, 2018]},
        {"name": "World Cup - Qualification", "match": "contains", "seasons": [2025, 2024]},
        {"name": "UEFA Nations League", "match": "exact", "seasons": [2024]},
    ]
    src = match_specs_to_sources(fake, specs)
    _check("main World Cup seasons resolved", (1, 2026) in src and (1, 2018) in src)
    _check("qualifiers resolved by 'contains'", (32, 2025) in src and (32, 2024) in src)
    _check("nations league resolved", (5, 2024) in src)
    _check("club FA Cup excluded (not country=World)", all(s[0] != 45 for s in src))
    _check("unavailable season dropped", (5, 2023) not in src)


def test_elo_seed():
    print("Elo seeding:")
    st = TeamState()
    st.seed_elo({10: "Brazil", 11: "Panama", 12: "Nowhere FC"})
    _check("known team seeded above default", st.elo[10] > st.elo[12])
    _check("stronger team seeded above weaker", st.elo[10] > st.elo[11])
    _check("unknown team falls back to default", st.elo[12] == elo_seed.seed_for("Nowhere FC"))


if __name__ == "__main__":
    test_poisson()
    test_scoring()
    test_league_resolver()
    test_elo_seed()
    test_end_to_end()
    print("\nAll self-tests passed.")
