# World Cup 2026 Pick 'Em Predictor

A machine-learning model that recommends **who to pick and what score to enter**
for each World Cup knockout match, tuned to your pool's **3 / 5 / 8** scoring:

| You get | For picking correctly (90' scoreline) |
|--------:|-----------------------|
| **3 pts** | correct outcome — home win / **draw** / away win |
| **5 pts** | correct outcome **+** goal difference |
| **8 pts** | exact score |

Draws are pickable: a correct draw always nails the goal difference (0), so it
scores **≥5** (never just 3), and **8** for the exact draw score.

## How it works

1. **Data** (`data.py`) — pulls finished international matches from API-Football.
   Scores are the **90' (regulation)** scoreline, exactly what the Pick 'Em uses.
2. **Features** (`features.py`) — Elo ratings, rolling form, rest days, neutral /
   knockout flags, built chronologically (no data leakage).
3. **ML model** (`model.py`) — a Poisson-loss gradient-boosted model predicts each
   team's **expected goals** for a match.
4. **Score distribution** (`poisson.py`) — those expected goals become a full
   90' score-probability grid (with an optional Dixon-Coles low-score tweak).
5. **Optional odds blend** (`predict.py`) — bookmaker 1X2 odds are de-vigged and
   blended into the grid, and the **correct-score** market is shown alongside the
   model (skipped automatically if your plan lacks odds).
6. **The important part** (`scoring.py`) — an **expected-points optimiser** searches
   every scoreline (including draws) and returns the **safe** max-EV pick, plus a
   **swing** pick: the highest-EV alternative with a *different outcome*, its
   exact-score chance, and how much EV it gives up — for when you need to
   differentiate from the field rather than track it.

## Setup (two things needed)

Because this runs in **Claude Code on the web**, two one-time settings are required:

### 1. Allowlist the API host
The default network policy **blocks** `v3.football.api-sports.io`. In your web
environment's **network settings**, add that host to the allowlist (a custom /
allowlist policy). See https://code.claude.com/docs/en/claude-code-on-the-web.

### 2. Add your API key as a secret
Set an environment secret named `API_FOOTBALL_KEY` in the web environment
settings (safest — it never touches git and persists across sessions).

*Running locally instead?* Copy `.env.example` to `.env` and put your key there.

```bash
pip install -r requirements.txt
python selftest.py          # verifies the maths offline (no key/network needed)
```

## Daily use

Each morning, just ask me for the day's predictions, or run:

```bash
python daily.py                 # today's fixtures
python daily.py --date 2026-07-05
python daily.py --no-odds       # ignore bookmaker odds
```

Output per match: expected goals, 90' result probabilities, advance
probabilities, the most likely exact score, and the **recommended pick with its
expected points** plus the next-best alternatives.

## Tuning (`config.py`)

- `COMPETITION_SPECS` — international competitions used for training, resolved to
  league ids **by name** at runtime (World Cup + qualifiers, Nations League, Euro,
  Copa América, AFCON, friendlies). Add/remove entries to tune the training mix;
  `USE_DYNAMIC_SOURCES=0` falls back to explicit `TRAINING_SOURCES` ids.
- `ELO_SEED_ENABLED` — seed team Elo from `elo_seed.py` priors (reduces
  cold-start noise) instead of a flat 1500.
- `WINNER_DEFINITION` — `"result_90"` (default) scores the 3-point winner by the
  90' result; switch to `"advances"` if your pool credits whoever progresses.
- `PTS_WINNER / PTS_GOAL_DIFF / PTS_EXACT` — change if your pool's points differ.
- `MARKET_BLEND_WEIGHT` — how much to trust bookmaker odds vs. the model (0.6 by
  default; lower toward 0.4 as the model accumulates more data).
- `DIXON_COLES_RHO`, `ELO_K`, `FORM_WINDOW` — modelling knobs.

## Notes & assumptions

- Outcome is judged on the **90' scoreline** (home win / draw / away win); a
  correct **draw** scores 5–8, not zero. Advancement via ET/penalties does not
  affect scoring.
- World Cup matches are treated as neutral venues except when a host nation
  (USA / Canada / Mexico) is the home side.
- Cached API responses live in `data/cache/` and the trained model in
  `data/models/` (both gitignored; regenerated automatically).
