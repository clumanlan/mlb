# v8 Results — k_predictor (league-wide rolling context)

**Date:** 2026-08-29
**Task:** Binary classification (PA grain), aggregated to batter-game grain for the decision verdict
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC (PA grain, triage filter) → reliability/resolution (game grain, decision metric)
**Data:** s3://mlbdk

## What this pass adds

Two new league-wide (whole-MLB, not per-team) rolling feature tables, testing whether
the model has any sense of "what does the league look like right now" — secular
shifts like the 2023 pitch-clock/shift-ban/bigger-bases rules or a juiced/dead-ball
year — a signal no prior version captured (every league-wide table before this
session was either a static last-season snapshot never used by k_predictor, or
scoped to one opposing team's own recent form). See `ROADMAP.md`'s 2026-08-29 entry
and `implementation_plan.md` for the full design brief.

New production code (TDD'd, `rolling_stats.py`, 8 new tests in
`tests/hit_predictor/test_rolling_stats.py`), both pooled across the ENTIRE league
(every team, every batter, unfiltered by role/starting-lineup) and keyed by
`(game_season, game_date)` only — the first feature tables in this repo without a
`gamepk`/`team_id`/`personId` key, since there's no single per-game "entity" to
roll league-wide the way one team/pitcher/batter has:

1. **`build_league_pa_outcome_rolling_feats`** — rolling K/BB/HBP/single/XBH/HR
   rate, the rolling equivalent of the existing static
   `season_stats.build_league_pa_outcome_stats`.
2. **`build_league_batter_rolling_stats`** — rolling BA/OBP/SLG/ISO/BABIP, the
   rolling equivalent of `build_batter_rolling_stats`, pooled to the whole league.

Both merge onto `pa_outcome` via a plain two-column merge on `(game_season,
game_date)` — every PA on the same date gets an identical league value by
construction, verified with an explicit `nunique() == 1` sanity assert in
`train.py` before training.

Carries forward v6's exact winning hyperparameters (LR: C=0.1/L1/no class
weighting; XGBoost: max_depth=2, learning_rate=0.03) and v7's full feature set
(the toughest-out/best-batter opposing-lineup features), rather than
re-grid-searching — same convention as every prior version.

## Sanity check on the new columns

All 11 new columns land at 99.8% non-null (the same small early-season gap every
other rolling feature in this repo has) and in a plausible range matching known
league averages:

| Feature | Non-null | Mean | Sanity |
|---|---|---|---|
| `league_roll_season_pa_strikeout_rate` | 99.8% | 0.2236 | matches ~22% MLB league K rate |
| `league_roll_season_pa_walk_rate` | 99.8% | 0.0835 | matches ~8% MLB league BB rate |
| `league_roll_season_pa_hbp_rate` | 99.8% | 0.0106 | matches ~1% MLB league HBP rate |
| `league_roll_season_pa_single_rate` | 99.8% | 0.1412 | plausible |
| `league_roll_season_pa_xbh_rate` | 99.8% | 0.0782 | plausible |
| `league_roll_season_pa_hr_rate` | 99.8% | 0.0305 | matches ~3% MLB league HR rate |
| `league_roll_season_ba` | 99.8% | 0.2456 | matches ~.245 MLB league BA |
| `league_roll_season_slg` | 99.8% | 0.4058 | matches ~.405 MLB league SLG |
| `league_roll_season_obp` | 99.8% | 0.3120 | matches ~.312 MLB league OBP |
| `league_roll_season_iso` | 99.8% | 0.1601 | = slg - ba, consistent |
| `league_roll_season_babip` | 99.8% | 0.2956 | matches the well-known ~.290-.300 league BABIP band |

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | Δ vs best naive | ROC-AUC |
|-------|--------|------------------|---------|
| Naive (most frequent) | 0.2190 | — | 0.5000 |
| Naive (per-role K rate) | 0.2207 | — | 0.5048 |
| Logistic regression (v6 config) | 0.2822 | +0.0615 | 0.5979 |
| XGBoost (v6 config) | 0.2836 | +0.0629 | 0.5995 |
| (v6 tuned XGBoost, standing best, for reference) | 0.2838 | | |

**PA-grain: flat vs v6** (0.2836 vs 0.2838, -0.0002 — well inside this project's
~0.005 "real" threshold), and essentially identical to v7's own PA-grain result
(0.2836) — adding 11 league-context columns on top of v7's feature set moved
nothing at this grain.

## Game-grain check (batter-game "1+ strikeout")

| Metric | Naive (per-role) | XGBoost (v6 config) | v6 (standing best, for reference) |
|---|---|---|---|
| reliability | 0.0005 | 0.0002 | 0.0001 |
| resolution | 0.0045 | 0.0137 | 0.0137 |
| roc_auc | 0.5626 | 0.6362 | — |

**Verdict: `real_improvement`** vs naive (same as every version since v2) — but not
a deeper one than v6's own result; resolution matches v6's exactly, reliability is
within noise.

## Feature importance — the 11 new columns

| Feature | Rank (of 56) | Importance |
|---|---|---|
| `league_roll_season_pa_walk_rate` | #12 | 0.0165 |
| `league_roll_season_pa_single_rate` | #14 | 0.0137 |
| `league_roll_season_babip` | #15 | 0.0119 |
| `league_roll_season_ba` | #24 | 0.0090 |
| `league_roll_season_pa_hbp_rate` | #28 | 0.0084 |
| `league_roll_season_iso` | #39 | 0.0070 |
| `league_roll_season_obp` | #40 | 0.0068 |
| `league_roll_season_slg` | #42 | 0.0068 |
| `league_roll_season_pa_strikeout_rate` | #47 | 0.0062 |
| `league_roll_season_pa_xbh_rate` | #50 | 0.0059 |
| `league_roll_season_pa_hr_rate` | #51 | 0.0056 |

**A genuinely counter-intuitive split, not a uniform "league context doesn't
matter" result.** `league_roll_season_pa_walk_rate` and `..._pa_single_rate` land
in the top quartile of all 56 features (#12, #14) — real, non-trivial placement,
ahead of several of v1-v7's own established features. But
`league_roll_season_pa_strikeout_rate` — the single most directly on-target
feature per the original "does the model know the league's K rate has shifted"
hypothesis — ranks #47 of 56, in the bottom fifth, alongside XBH rate (#50) and HR
rate (#51). The slash-line half (BA/OBP/SLG/ISO/BABIP) splits similarly: BABIP
(#15) and BA (#24) place respectably, while OBP/SLG/ISO all land in the bottom
third (#39-42) — components already well-represented elsewhere in the feature set
(the model already has plenty of batter- and team-level K-rate signal from v1-v3,
so a league-wide K-rate echo of that same category is the most redundant of the
11, not the least).

## Interpretation

**Flat aggregate result — same "additive feature, no PA/game-grain movement"
pattern seen throughout most of this project's feature-hunt history** (v3, v4, v5,
v7 here; v3-v7 on `batters_faced_predictor`). This does not close the "does the
model know what season it is" question the way a uniformly-low-importance result
would have — walk rate and single rate carry real, top-quartile signal
individually, they just don't move PR-AUC/resolution at the margin, the same
"well-ranked individually, redundant with what's already in the model" shape seen
repeatedly in this project (e.g. v3's whiff rate, v7's toughest-out K-rate). The
one feature built specifically to test the stated hypothesis — league strikeout
rate itself — is the weakest-performing of the PA-outcome group, which is real
evidence against (not merely "untested for") the specific "league K-rate has
drifted and the model needs to know it" mechanism, at least over this val season's
single-year window.

**v6's tuned XGBoost (PR-AUC 0.2838, `real_improvement`, max_depth=2,
learning_rate=0.03) remains the strongest k_predictor version and standing
production candidate.**

## Next steps

- Don't pursue a season-to-date league-strikeout-rate feature further without a
  new angle — it under-performed its own hypothesis here. If the "what season is
  it" question is revisited, league walk rate and single rate (both top-quartile)
  are the better-motivated leads, not strikeout rate.
- Not attempted this pass, still open: a trailing-N-game league window (this pass
  only tried `window='season'`); ROADMAP's separate gap #2 (opposing-team
  last-season shrinkage); a third league-wide pitcher-composite table
  (WHIP/FIP, needs a new innings-pitched aggregation path).
- `build_league_pa_outcome_rolling_feats`/`build_league_batter_rolling_stats` and
  the `(game_season, game_date)`-only join pattern are real, tested, reusable
  infrastructure regardless of this specific result — any future model needing a
  whole-league rolling context signal (bb_predictor, a future hits-allowed model)
  can reuse them directly.
