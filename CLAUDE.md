# MyProjects — context for Claude

Personal projects repo. The active project is the **World Cup 2026 Pick 'Em
predictor** in `worldcup_pickem/`. Other files (`Hurrican Analysis.py`, the REIT
notebook) are unrelated older exercises.

## World Cup 2026 Pick 'Em (`worldcup_pickem/`)

Predicts World Cup knockout matches to help the user win a Pick 'Em pool
(currently chasing 1st from 4th). For each match the user enters a **winner**
and a **90' scoreline**.

### Pool scoring (based on the 90' scoreline)
One rule, best category you hit, comparing your entered 90' scoreline to the
actual 90' scoreline:
- **3** — correct outcome (home win / draw / away win)
- **5** — correct outcome **+** goal difference
- **8** — exact score

Outcome is decided at the **end of 90'** (regulation), NOT by who advances on
ET/penalties. **Draws are pickable and score**: a correct draw always nails the
goal difference (0), so it scores **≥5** (never just 3), and **8** for the exact
draw score. `scoring.py` derives the outcome from the picked scoreline, so a
draw entry is a first-class option (often the best *swing* pick in tight games).

### How it works
ML hybrid: a Poisson-loss gradient-boosted model predicts each team's expected
goals from Elo/form/rest features (`model.py`, `features.py`) → a 90'
score-probability grid (`poisson.py`) → optional de-vigged bookmaker-odds blend
(`predict.py`) → an expected-points optimiser for the 3/5/8 rules (`scoring.py`).
Each recommendation carries a **safe** (max-EV) pick and a **swing** (highest-EV
alternative with a *different outcome*, for differentiating from the field when
chasing), plus bookmaker 1X2 + correct-score markets shown alongside.

### Running it
```bash
python worldcup_pickem/daily.py                 # today's fixtures
python worldcup_pickem/daily.py --date 2026-07-05
python worldcup_pickem/daily.py --no-odds       # ignore bookmaker odds
python worldcup_pickem/backtest.py              # walk-forward accuracy vs baselines
python worldcup_pickem/backtest.py --all-internationals   # bigger test sample
python worldcup_pickem/selftest.py              # offline checks, no key/network
```

`backtest.py` retrains per matchday on prior-only data and scores the model's pick
vs. `modal` and a naive `fav1-0` baseline (pts/game, winner/GD/exact %), plus
Brier/RPS calibration. The model earns its keep only if it beats `fav1-0`. Runs
model-only (no historical odds).

### Presenting results to the user
The user cares MOST about the **most likely scorelines** per match (and the most
likely score for each result: home win / draw / away win), and LESS about the
expected-points number. Lead with the score distribution; keep EV secondary.
Each recommendation object exposes `top_scores` and `outcome_scores` for this.

### Environment (already configured in the cloud environment)
- `API_FOOTBALL_KEY` — set as an environment variable.
- Network: **Custom** access allowlisting `v3.football.api-sports.io` (+ default
  package registries). API-Football league id **1**, season **2026**.
- Setup script installs `worldcup_pickem/requirements.txt`.

### Training data (dynamic)
Training competitions are resolved by **name** at runtime (`data.resolve_sources`
→ `/leagues`), not hard-coded ids. `COMPETITION_SPECS` in `config.py` lists them:
World Cup + qualifiers, Nations League (UEFA/CONCACAF), Euro 2024, Copa América
2024, AFCON, international friendlies. Only `country == "World"` leagues match, and
only seasons the plan covers are pulled. Falls back to `TRAINING_SOURCES` (explicit
ids) if `/leagues` is unavailable. Elo is seeded from `elo_seed.py` priors
(`ELO_SEED_ENABLED`) to cut cold-start noise. User is on the **Pro** API plan.

### Handy config knobs (`config.py`)
`WINNER_DEFINITION`, `PTS_WINNER/PTS_GOAL_DIFF/PTS_EXACT`, `MARKET_BLEND_WEIGHT`
(currently 0.6 — lean on odds while sample is thin; lower toward 0.4 as data
grows), `COMPETITION_SPECS`, `USE_DYNAMIC_SOURCES`, `ELO_SEED_ENABLED`, `MAX_GOALS`,
`DIXON_COLES_RHO`, `ELO_K`, `FORM_WINDOW`.

## Dev practices
- Run `python worldcup_pickem/selftest.py` after changing model/scoring code.
- Commit on the working branch, open a PR, and squash-merge into `main` (fresh
  sessions clone `main`). Don't push straight to `main`.
