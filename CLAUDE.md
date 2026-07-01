# MyProjects — context for Claude

Personal projects repo. The active project is the **World Cup 2026 Pick 'Em
predictor** in `worldcup_pickem/`. Other files (`Hurrican Analysis.py`, the REIT
notebook) are unrelated older exercises.

## World Cup 2026 Pick 'Em (`worldcup_pickem/`)

Predicts World Cup knockout matches to help the user win a Pick 'Em pool
(currently chasing 1st from 4th). For each match the user enters a **winner**
and a **90' scoreline**.

### Pool scoring (knockout)
Points are totals, best category you hit:
- **3** — correct winning team
- **5** — correct winner **+** goal difference (90')
- **8** — correct winner **+** exact score (90')

The **winner is decided at the end of 90'** (regulation), NOT by who advances on
ET/penalties. This is set via `WINNER_DEFINITION = "result_90"` in `config.py`.
A 90' draw scores no winner points.

### How it works
ML hybrid: a Poisson-loss gradient-boosted model predicts each team's expected
goals from Elo/form/rest features (`model.py`, `features.py`) → a 90'
score-probability grid (`poisson.py`) → optional de-vigged bookmaker-odds blend
(`predict.py`) → an expected-points optimiser for the 3/5/8 rules (`scoring.py`).

### Running it
```bash
python worldcup_pickem/daily.py                 # today's fixtures
python worldcup_pickem/daily.py --date 2026-07-05
python worldcup_pickem/daily.py --no-odds       # ignore bookmaker odds
python worldcup_pickem/selftest.py              # offline checks, no key/network
```

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
