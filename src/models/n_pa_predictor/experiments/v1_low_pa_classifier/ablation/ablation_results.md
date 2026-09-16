# n_pa_predictor v1 low_pa classifier — feature-ablation results
Val season 2024, XGBoost @ 0.85 confidence threshold. Baseline (locked production, all 9 features): precision 0.644, 95% Wilson CI [55.4%, 72.5%], n=118.
## Methodology
1. Cluster the 9 frozen features by Spearman rank correlation (threshold 0.8) into redundant families.
2. Within each family with more than one member, keep only the highest-XGBoost-importance member.
3. Backward-eliminate remaining features one at a time, weakest importance first, retraining and re-checking precision-at-0.85 each time.
4. Stop eliminating once a drop's 95% Wilson CI no longer overlaps the baseline's — that feature is load-bearing, not a passenger.
## Correlation families
- ['batting_order']
- ['is_home']
- ['expected_start_innings']
- ['expected_start_innings_weight']
- ['opp_starter_whip_season']
- ['batter_n_pa_roll_season_games_n']
- ['batter_n_pa_roll_season_avg_n_pa_per_game']
- ['team_roll_season_win_pct']
- ['team_roll_season_runs_scored']

## XGBoost feature importances (baseline model)
```
batting_order                                0.6598
batter_n_pa_roll_season_avg_n_pa_per_game    0.1203
is_home                                      0.0483
team_roll_season_runs_scored                 0.0355
team_roll_season_win_pct                     0.0296
expected_start_innings_weight                0.0281
opp_starter_whip_season                      0.0280
expected_start_innings                       0.0253
batter_n_pa_roll_season_games_n              0.0250
```

## Sweep results
```
                                       variant                           dropped_feature  precision   n   ci_low  ci_high  within_baseline_ci                                                                                                                                                                                                                                features_used
                                      baseline                                      None   0.644068 118 0.554387 0.724664                True [batting_order, is_home, expected_start_innings, expected_start_innings_weight, opp_starter_whip_season, batter_n_pa_roll_season_games_n, batter_n_pa_roll_season_avg_n_pa_per_game, team_roll_season_win_pct, team_roll_season_runs_scored]
          drop_batter_n_pa_roll_season_games_n           batter_n_pa_roll_season_games_n   0.565217  92 0.463321 0.661886                True                                  [batting_order, is_home, expected_start_innings, expected_start_innings_weight, opp_starter_whip_season, batter_n_pa_roll_season_avg_n_pa_per_game, team_roll_season_win_pct, team_roll_season_runs_scored]
                   drop_expected_start_innings                    expected_start_innings   0.613636  88 0.509186 0.708581                True                                                          [batting_order, is_home, expected_start_innings_weight, opp_starter_whip_season, batter_n_pa_roll_season_avg_n_pa_per_game, team_roll_season_win_pct, team_roll_season_runs_scored]
                  drop_opp_starter_whip_season                   opp_starter_whip_season   0.600000  80 0.490453 0.700383                True                                                                                   [batting_order, is_home, expected_start_innings_weight, batter_n_pa_roll_season_avg_n_pa_per_game, team_roll_season_win_pct, team_roll_season_runs_scored]
            drop_expected_start_innings_weight             expected_start_innings_weight   0.542857  70 0.426981 0.654274                True                                                                                                                  [batting_order, is_home, batter_n_pa_roll_season_avg_n_pa_per_game, team_roll_season_win_pct, team_roll_season_runs_scored]
                 drop_team_roll_season_win_pct                  team_roll_season_win_pct   0.613333  75 0.500173 0.715450                True                                                                                                                                            [batting_order, is_home, batter_n_pa_roll_season_avg_n_pa_per_game, team_roll_season_runs_scored]
             drop_team_roll_season_runs_scored              team_roll_season_runs_scored   0.597222  72 0.481805 0.702791                True                                                                                                                                                                          [batting_order, is_home, batter_n_pa_roll_season_avg_n_pa_per_game]
                                  drop_is_home                                   is_home   0.625000  64 0.502502 0.733342                True                                                                                                                                                                                   [batting_order, batter_n_pa_roll_season_avg_n_pa_per_game]
drop_batter_n_pa_roll_season_avg_n_pa_per_game batter_n_pa_roll_season_avg_n_pa_per_game        NaN   0      NaN      NaN               False                                                                                                                                                                                                                              [batting_order]
```

## Recommendation
Minimal feature set (2 of 9): `['batting_order', 'batter_n_pa_roll_season_avg_n_pa_per_game']`.

## Caveats — read before trusting the minimal set

1. **No correlation-redundancy pruning actually fired.** All 9 features formed
   their own singleton cluster at the 0.8 Spearman threshold — none of this
   model's features are near-duplicates of each other. The entire reduction
   below came from backward elimination on weak-but-independent features, not
   from collapsing correlated families as originally hypothesized.

2. **CI overlap is a low bar at these sample sizes, not strong evidence of
   zero value.** Once the val season is filtered down to only the rows above
   a 0.85 confidence threshold, n ranges from 64 to 118 across steps — wide
   enough Wilson intervals (roughly +/-10pp) that "the CI still overlaps
   baseline" mostly means "we lack the power to prove this feature matters,"
   not "this feature provably contributes nothing." Same caution this repo
   already applies to k_predictor's small-sample backtests
   (`significance_check.py`).

3. **The stopping point was a coverage collapse, not a precision failure.**
   The final rejected drop (`batter_n_pa_roll_season_avg_n_pa_per_game`) has
   `n=0` — with `batting_order` alone, no val-season prediction ever reaches
   0.85 confidence (consistent with the per-slot naive floor's known ceiling,
   ~49% at the 9-hole). That's a real, valid reason to keep this feature —
   without it there's no usable high-confidence signal at all, not that this
   feature was caught improving precision. `opp_starter_whip_season` and
   `expected_start_innings` — previously reported as carrying real signal in
   the pre-classifier regression work — were both dropped here with CI
   overlap intact; this is genuine tension with that earlier finding, not
   confirmation of it. Read as: this val-season sample can't distinguish
   "these 7 features add nothing" from "these 7 features add something real
   too small for ~70-120 examples to detect."

4. **Preprocessing here is not byte-identical to `train.py`'s** (median-fill
   instead of `SimpleImputer`+`OrdinalEncoder`, no categorical path needed
   since the only non-numeric feature is boolean `is_home`) — close enough
   that the baseline (0.644, n=118) lands within noise of the originally
   reported production number (0.647, n=116), but not an exact
   reproduction. All comparisons in this sweep are internally consistent
   (same preprocessing at every step), so the *relative* deltas are trustworthy
   even though the absolute baseline number differs slightly from the locked
   production report.

**Bottom line:** `batting_order` and the batter's own rolling PA rate are
confirmed load-bearing (removing the latter collapses coverage to zero). The
other 7 features are candidates for dropping from the initial feature-store
build — not because they're proven worthless, but because this sweep found
no evidence they're required at this sample size. Before actually removing
them from production: either re-run this sweep against the larger/multi-season
val sample already flagged as a follow-up in `ROADMAP.md`'s backlog (tighter
CIs would turn "can't tell" into a real answer), or treat the 2-feature set
as the *first* thing served through the feature store and add the other 7
back only if a later test season shows real degradation.
