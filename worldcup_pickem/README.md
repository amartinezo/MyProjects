# World Cup 2026 Pick 'Em Predictor

A machine-learning model that recommends **who to pick and what score to enter**
for each World Cup knockout match, tuned to your pool's **3 / 5 / 8** scoring:

| You get | For picking correctly |
|--------:|-----------------------|
| **3 pts** | the winning team (90') |
| **5 pts** | winning team **+** goal difference (90') |
| **8 pts** | winning team **+** exact score (90') |

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
   blended into the grid (skipped automatically if your plan lacks odds).
6. **The important part** (`scoring.py`) — an **expected-points optimiser** searches
   every (winner, scoreline) entry and returns the one that **maximises expected
   points** under the 3 / 5 / 8 rules. This is often *not* the single most likely
   score, which is the edge over just eyeballing it.

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

- `TRAINING_SOURCES` — add World Cup Qualifier / Nations League league ids to
  enrich team-strength estimates (more relevant matches = better model).
- `WINNER_DEFINITION` — `"result_90"` (default) scores the 3-point winner by the
  90' result; switch to `"advances"` if your pool credits whoever progresses.
- `PTS_WINNER / PTS_GOAL_DIFF / PTS_EXACT` — change if your pool's points differ.
- `MARKET_BLEND_WEIGHT` — how much to trust bookmaker odds vs. the model.
- `DIXON_COLES_RHO`, `ELO_K`, `FORM_WINDOW` — modelling knobs.

## Notes & assumptions

- "Winning team" means the **team ahead at the end of 90'** (a 90' draw scores no
  winner points); the entered **score is the 90' scoreline**. Flip
  `WINNER_DEFINITION` to `"advances"` if your pool credits whoever progresses.
- World Cup matches are treated as neutral venues except when a host nation
  (USA / Canada / Mexico) is the home side.
- Cached API responses live in `data/cache/` and the trained model in
  `data/models/` (both gitignored; regenerated automatically).
