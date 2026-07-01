"""Your morning command: train on all data, predict today's World Cup games.

    python daily.py                 # today's fixtures
    python daily.py --date 2026-07-02
    python daily.py --no-odds       # ignore bookmaker odds

The whole pipeline (fetch -> features -> train -> predict) runs each morning so
predictions always reflect the latest results. Training is fast (seconds).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import config
from api_client import APIError, get_client
from data import build_match_frame, upcoming_fixtures
from features import build_training_features
from model import GoalsModel
from predict import predict_fixtures


def _fmt_pct(x: float) -> str:
    return f"{100 * x:4.0f}%"


def print_recommendation(r) -> None:
    print("=" * 66)
    print(f"  {r.home_name}  vs  {r.away_name}")
    print("-" * 66)
    print(f"  Expected goals (90'):  {r.home_name} {r.lambda_home:.2f}"
          f"  -  {r.lambda_away:.2f} {r.away_name}")
    print(f"  90' result:  Home {_fmt_pct(r.p_home_90)}   "
          f"Draw {_fmt_pct(r.p_draw_90)}   Away {_fmt_pct(r.p_away_90)}")
    print(f"  Advance:     {r.home_name} {_fmt_pct(r.p_home_adv)}   "
          f"{r.away_name} {_fmt_pct(r.p_away_adv)}")

    # Score distribution: the most likely exact scorelines.
    if r.top_scores:
        line = "   ".join(f"{i}-{j} ({100 * p:.0f}%)" for i, j, p in r.top_scores)
        print(f"  Most likely scores:  {line}")

    # Most likely scoreline within each result (home win / draw / away win).
    if r.outcome_scores:
        h = r.outcome_scores.get("home")
        d = r.outcome_scores.get("draw")
        a = r.outcome_scores.get("away")
        if h and d and a:
            print(f"  By result:   {r.home_name} win {h[0]}-{h[1]} ({_fmt_pct(h[2])})   "
                  f"Draw {d[0]}-{d[1]} ({_fmt_pct(d[2])})   "
                  f"{r.away_name} win {a[0]}-{a[1]} ({_fmt_pct(a[2])})")

    print(f"\n  >> RECOMMENDED PICK:  {r.pretty_pick()}")
    print(f"     Expected points:   {r.expected_points:.2f}  "
          f"(rules {config.PTS_WINNER}/{config.PTS_GOAL_DIFF}/{config.PTS_EXACT})")
    if r.alternatives:
        alts = ",  ".join(
            f"{a['pick']} (EV {a['expected_points']})" for a in r.alternatives
        )
        print(f"     Next best:         {alts}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="World Cup Pick 'Em daily predictions")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="Match date YYYY-MM-DD (default: today)")
    ap.add_argument("--no-odds", action="store_true", help="Ignore bookmaker odds")
    args = ap.parse_args(argv)

    client = get_client()

    try:
        print(f"Fetching & training on data from {config.TRAINING_SOURCES} ...")
        matches = build_match_frame(client)
        if matches.empty:
            print("No finished matches returned. Check league/season ids in config.py.")
            return 1
        print(f"  Trained on {len(matches)} matches "
              f"({matches['date'].min().date()} -> {matches['date'].max().date()}).")

        feat, state = build_training_features(matches)
        model = GoalsModel().fit(feat)
        model.save()

        fixtures = upcoming_fixtures(args.date, client)
        if fixtures.empty:
            print(f"\nNo World Cup fixtures scheduled on {args.date}.")
            return 0

        print(f"\n{len(fixtures)} fixture(s) on {args.date}:\n")
        recos = predict_fixtures(
            model, state, fixtures, client=client, use_odds=not args.no_odds
        )
        for r in recos:
            print_recommendation(r)

    except APIError as exc:
        print(f"\nAPI error: {exc}\n", file=sys.stderr)
        print("Fixes: (1) set the API_FOOTBALL_KEY secret; (2) on Claude Code web, "
              "allowlist v3.football.api-sports.io in the environment network policy.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
