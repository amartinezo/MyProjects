"""Walk-forward backtest: does the model actually beat naive chalk?

For each test match (default: World Cup matches), the model is trained ONLY on
matches that kicked off before it, then its pick is scored against the real 90'
result under the 3/5/8 rules. Elo/form features are already leak-free (state is
built chronologically); retraining per matchday keeps the ML model leak-free too.

Strategies compared per match:
  * model   - the expected-points-optimised pick
  * modal   - the single most likely scoreline (+ its implied winner)
  * fav1-0  - naive baseline: higher-xG team to win 1-0

Also reports calibration of the model's 90' outcome probabilities (Brier, RPS).

    python backtest.py                       # backtest World Cup matches
    python backtest.py --since 2026-06-11    # only from a date
    python backtest.py --all-internationals  # test on every competition, bigger n

Runs model-only (no bookmaker odds; historical odds aren't fetched).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from api_client import APIError, get_client
from data import build_match_frame
from features import TeamState
from model import GoalsModel
from poisson import modal_score, outcome_probs, score_grid
from scoring import optimize_pick, score_pick

# Column order the GoalsModel expects, plus bookkeeping columns.
from features import FEATURE_COLS


def brier_1x2(p_home: float, p_draw: float, p_away: float, actual_idx: int) -> float:
    """Multiclass Brier score for one (home/draw/away) prediction. Lower better."""
    target = [0.0, 0.0, 0.0]
    target[actual_idx] = 1.0
    preds = [p_home, p_draw, p_away]
    return sum((p - t) ** 2 for p, t in zip(preds, target))


def rps(p_home: float, p_draw: float, p_away: float, actual_idx: int) -> float:
    """Ranked Probability Score for ordered outcomes (home>draw>away). Lower better."""
    preds = [p_home, p_draw, p_away]
    target = [0.0, 0.0, 0.0]
    target[actual_idx] = 1.0
    cum_p = cum_t = 0.0
    total = 0.0
    for k in range(2):  # sum over first (n-1) categories
        cum_p += preds[k]
        cum_t += target[k]
        total += (cum_p - cum_t) ** 2
    return total / 2.0


def _outcome_index(hg: int, ag: int) -> int:
    return 0 if hg > ag else 2 if hg < ag else 1


def run_backtest(
    matches: pd.DataFrame,
    *,
    test_league_id: int | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    """Walk-forward evaluation over a DataFrame of finished matches."""
    df = matches.sort_values("date").reset_index(drop=True)

    # Which matches are scored (the "test" set).
    test_mask = pd.Series(True, index=df.index)
    if test_league_id is not None:
        test_mask &= df["league_id"] == test_league_id
    if since is not None:
        test_mask &= df["date"] >= pd.to_datetime(since, utc=True)
    if until is not None:
        test_mask &= df["date"] <= pd.to_datetime(until, utc=True)

    # Chronological pass: build leak-free feature rows for every match.
    state = TeamState()
    id2name = {}
    for m in df.itertuples(index=False):
        id2name[m.home_id] = m.home
        id2name[m.away_id] = m.away
    state.seed_elo(id2name)

    train_rows: list[dict] = []          # every match's (features + goals + date)
    test_items: list[dict] = []          # test matches with their as-of feature rows
    for idx, m in enumerate(df.itertuples(index=False)):
        hrow = state.team_row(m.home_id, m.away_id, is_physical_home=True,
                              neutral=m.neutral, knockout=m.knockout, date=m.date)
        arow = state.team_row(m.away_id, m.home_id, is_physical_home=False,
                              neutral=m.neutral, knockout=m.knockout, date=m.date)
        if test_mask.iloc[idx]:
            test_items.append({"date": m.date, "home": m.home, "away": m.away,
                               "hg": m.hg, "ag": m.ag, "hrow": hrow, "arow": arow})
        train_rows.append({**hrow, "goals": m.hg, "date": m.date})
        train_rows.append({**arow, "goals": m.ag, "date": m.date})
        state.update(m.home_id, m.away_id, m.hg, m.ag, m.neutral, m.date,
                     home_name=m.home, away_name=m.away)

    train_df = pd.DataFrame(train_rows)

    strategies = ("model", "modal", "fav1-0")
    agg = {s: {"points": 0, "winner": 0, "gd": 0, "exact": 0} for s in strategies}
    brier_sum = rps_sum = 0.0
    n = 0

    # Group test matches by date; retrain once per matchday on strictly-earlier data.
    by_date: dict = {}
    for item in test_items:
        by_date.setdefault(item["date"], []).append(item)

    for d in sorted(by_date):
        prior = train_df[train_df["date"] < d]
        if len(prior) < 100:      # not enough history to train a meaningful model yet
            continue
        model = GoalsModel().fit(prior)

        for item in by_date[d]:
            xh = pd.DataFrame([item["hrow"]])[FEATURE_COLS]
            xa = pd.DataFrame([item["arow"]])[FEATURE_COLS]
            lh = float(model.predict_lambda(xh)[0])
            la = float(model.predict_lambda(xa)[0])
            grid = score_grid(lh, la)
            hg, ag = item["hg"], item["ag"]

            # Calibration of the 90' outcome distribution.
            ph, pd_, pa = outcome_probs(grid)
            oi = _outcome_index(hg, ag)
            brier_sum += brier_1x2(ph, pd_, pa, oi)
            rps_sum += rps(ph, pd_, pa, oi)
            n += 1

            # --- strategy picks ---
            picks = {}
            rec = optimize_pick(grid, home_name=item["home"], away_name=item["away"],
                                lambda_home=lh, lambda_away=la)
            picks["model"] = (rec.winner, rec.hg, rec.ag)

            mi, mj, _ = modal_score(grid)
            m_winner = "home" if mi > mj else "away" if mi < mj else ("home" if lh >= la else "away")
            picks["modal"] = (m_winner, mi, mj)

            if lh >= la:
                picks["fav1-0"] = ("home", 1, 0)
            else:
                picks["fav1-0"] = ("away", 0, 1)

            for s, (w, ph_, pa_) in picks.items():
                pts = score_pick(w, ph_, pa_, hg, ag)
                agg[s]["points"] += pts
                if pts >= config.PTS_WINNER:
                    agg[s]["winner"] += 1
                if pts == config.PTS_EXACT:
                    agg[s]["exact"] += 1
                elif pts == config.PTS_GOAL_DIFF:
                    agg[s]["gd"] += 1

    return {"n": n, "strategies": agg,
            "brier": (brier_sum / n) if n else float("nan"),
            "rps": (rps_sum / n) if n else float("nan")}


def print_report(res: dict) -> None:
    n = res["n"]
    print(f"\nBacktest over {n} scored matches "
          f"(rules {config.PTS_WINNER}/{config.PTS_GOAL_DIFF}/{config.PTS_EXACT}, "
          f"winner={config.WINNER_DEFINITION})\n")
    if n == 0:
        print("  Not enough matches to score. Try --all-internationals or a wider window.")
        return
    header = f"  {'strategy':<10}{'pts/game':>10}{'total':>8}{'winner%':>9}{'GD%':>7}{'exact%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s, a in res["strategies"].items():
        print(f"  {s:<10}{a['points'] / n:>10.3f}{a['points']:>8}"
              f"{100 * a['winner'] / n:>8.0f}%{100 * a['gd'] / n:>6.0f}%"
              f"{100 * a['exact'] / n:>7.0f}%")
    print(f"\n  Model calibration:  Brier {res['brier']:.3f}   RPS {res['rps']:.3f}  (lower = better)")
    print("  Baseline 'fav1-0' is naive chalk; the model earns its keep only if it beats it.\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest for the Pick 'Em model")
    ap.add_argument("--since", default=None, help="Only score matches on/after YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="Only score matches on/before YYYY-MM-DD")
    ap.add_argument("--all-internationals", action="store_true",
                    help="Score every competition, not just the World Cup (bigger sample)")
    args = ap.parse_args(argv)

    try:
        print("Fetching match data ...")
        matches = build_match_frame(get_client())
        if matches.empty:
            print("No matches returned.")
            return 1
        test_league = None if args.all_internationals else config.WORLD_CUP_LEAGUE_ID
        res = run_backtest(matches, test_league_id=test_league,
                           since=args.since, until=args.until)
        print_report(res)
    except APIError as exc:
        print(f"\nAPI error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
