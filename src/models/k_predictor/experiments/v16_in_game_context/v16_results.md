# v16 Results — k_predictor (in-game context features)

**Date:** 2026-09-03
**Task:** Binary classification (PA grain) — diagnostic only, not a production-candidate pass
**Target:** Will this plate appearance end in a strikeout?
**Primary metric:** PR-AUC / ROC-AUC (PA grain) — no game-grain check this version (see "Why no game-grain check" below)
**Data:** `experiments/count_distribution_check/_model_cache/model_df_v6.parquet` (cached, not rebuilt from S3 — see "Operational note")

## What this pass adds

Every prior k_predictor feature (v1-v15) is knowable *before first pitch*, because the
real use case (a total-strikeouts prop) needs a prediction before the game starts. This
pass deliberately breaks that rule and asks a narrower, purely diagnostic question:
once a game IS in progress, does knowing what's actually happened so far sharpen the
prediction for the very next batter?

Directly prompted by an existing, previously-unexploited finding:
`experiments/count_distribution_check/within_game_correlation_check.py` +
`icc_check.py` found a real "hot/cold tonight" signal — splitting each start's PAs into
odd/even halves by real chronological order, residuals correlate (r=0.105, p=8.8e-13,
n=4,629 starts; per-PA ICC=0.0100) — but that was only ever used to estimate a variance-
inflation factor. Nobody had fed the pitcher's actual within-game running state to the
model as literal features to see if a shallow tree can use it PA-by-PA. This is that
check.

New TDD'd `processing/features/in_game_context.py` (7 tests in
`tests/k_predictor/test_in_game_context.py`):

1. **`build_pitcher_in_game_running_stats`** — realized within-game state as of
   *strictly before* each PA, ordered by real chronological `play_id` within
   `(gamepk, pitcher_id)` (same ordering convention `within_game_correlation_check.py`
   already established). Adds `pitcher_pa_faced_this_game_so_far`,
   `pitcher_k_this_game_so_far`, `pitcher_k_this_game_so_far_rate` (NaN at the first PA
   — no prior PAs to compute a rate from).
2. **`build_pitcher_in_game_hot_cold_gap`** — is this pitcher over/under-performing his
   own established pre-game rate *tonight*? Adds `pitcher_expected_k_this_game_so_far`
   (`pitcher_roll_season_pa_strikeout_rate * PAs faced so far`) and
   `pitcher_hot_cold_gap_this_game_so_far` (actual − expected). Deliberately built from
   the pitcher's own already-existing, already-point-in-time-safe pre-game rate column
   rather than v6's own predicted probabilities — using a model's own in-sample
   predictions as a feature into itself (when fit on the same rows the model saw) would
   be circular/leaky; a pre-game rate constant carries no such risk.

Tests cover: point-in-time correctness within a game, no cross-contamination between
two pitchers in the same `gamepk` (the exact team-scoping bug class already found
elsewhere in this repo, `build_batter_slot_expansion`), order-by-`play_id` not
input-row-order, and reset at the start of a new game for the same pitcher.

## Why no game-grain check

Every prior version's verdict comes from `run_pa_vs_game_grain_check` — aggregating
PA-level probabilities into one pre-game "1+ strikeout this game" prediction. There's no
single pre-game probability left to aggregate here: each PA's in-game features are
different by construction, and the whole feature set doesn't exist before first pitch
anyway. This version is scored PA-by-PA only.

## Operational note — a real infra failure, worth remembering

The first attempt copy-adapted `v6_tuned/train.py`'s full raw-S3 rebuild (same shape as
`score_2026_test_dates.py`: load 7 seasons of pbp/boxscore, build the 42-feature frame
from scratch). It was **killed by the OS partway through the multi-season feature
build** — no traceback, just terminated. This is the same real memory-thrashing failure
mode v6's own changelog entry already documented once ("the run itself hit real system
memory thrashing on its first attempt... needed a kill + restart").

Fixed by loading `experiments/count_distribution_check/_model_cache/model_df_v6.parquet`
instead (733,275 rows, all 7 relevant seasons, v6's 42 features already built, plus
`gamepk`/`pitcher_id`/`play_id`/`is_strikeout` — everything `in_game_context.py` needs).
This is exactly the reuse that cache was built for (see its own ROADMAP note: "any
future k_predictor-v6-feature-set diagnostic... should load that cache instead of
rebuilding from S3"). Runs in under a minute this way instead of ~20-25 minutes of
S3 reads plus the memory risk. **Any future k_predictor diagnostic that only needs v6's
existing 42 features (not new raw pbp/boxscore columns) should load this cache too.**

## Results (evaluated on val season 2024, 105,265 PAs)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| v6 (pre-game only, 42 features) | 0.2838 | 0.5996 |
| v16 (+ 5 in-game context features, 47 total) | 0.2852 | 0.6024 |
| delta | +0.0014 | +0.0028 |

Restricted to the exact regime these features were built for — pitcher has already
faced **≥3 batters this game** (90,909 of 105,265 rows, 86.4%):

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| v6 | 0.2825 | 0.5998 |
| v16 | 0.2839 | 0.6028 |
| delta | +0.0014 | +0.0030 |

**Flat** — both deltas, in both cuts, are well under this project's ~0.005 "real"
threshold. Restricting to the subset with the most information available to exploit
doesn't change the magnitude at all.

## Feature importance

| Feature | Rank (of 47) | Importance |
|---|---|---|
| `pitcher_hot_cold_gap_this_game_so_far` | #20 | 0.0100 |
| `pitcher_pa_faced_this_game_so_far` | #24 | 0.0084 |
| `pitcher_expected_k_this_game_so_far` | #31 | 0.0064 |
| `pitcher_k_this_game_so_far` | #40 | 0.0058 |
| `pitcher_k_this_game_so_far_rate` | **#47 (last)** | **0.0000** |

**2 of 5 new columns get real, top-half usage** — `pitcher_hot_cold_gap_this_game_so_far`
and `pitcher_pa_faced_this_game_so_far` are genuinely used, not dead weight. The plain
K-rate-so-far ratio is completely unused (0.0000, dead last) — the same "derived ratio
adds nothing a tree can't reconstruct from the raw pieces already present" pattern
`interaction_feats.py` and several prior versions (v7's pitch-count trend ratio, v9's
ratio/direction columns) have already documented independently in this repo.

## Interpretation

**Closes the loop on the icc_check.py finding**: the "hot/cold tonight" signal is real
enough that the model both discovers and uses it (2 of 5 features rank top-half of 47),
but not large enough to move the aggregate PA-grain metric. This is consistent with
icc_check.py's own read — a per-PA ICC of 0.0100 implies only a ~1.10x SD inflation over
pure independence, real but too small to explain a start with a genuinely dominant or
disastrous outing on its own.

This also means the in-game features are correctly scoped as diagnostic-only: even if
the magnitude had been larger, they cannot feed the actual pre-game total-strikeouts
prop v6 serves, since none of this state exists before first pitch. The value here is
purely in confirming/quantifying the within-game correlation mechanism, not in
producing a deployable feature.

**v6's tuned XGBoost (PR-AUC 0.2838, `real_improvement` at game grain, max_depth=2,
learning_rate=0.03) remains the standing production candidate.**

## Next steps

- `in_game_context.py`'s two functions are real, tested, reusable infrastructure
  (any future within-game diagnostic — e.g. a genuine live/in-play model, flagged but
  explicitly deferred during this session's scoping discussion — starts from here)
  regardless of this result. Kept, not reverted.
- Not pursued further under the "more in-game features" hypothesis — the two real
  columns already found (hot/cold gap, PAs faced so far) are the most direct
  operationalization of the icc_check.py mechanism; a third in-game feature is unlikely
  to add much given how small the underlying effect already is.
- If a genuine live/in-play use case comes up later (update the total-K prediction AS a
  game progresses, not just diagnose it after the fact), that's a materially different
  build — its own evaluation framework, not just "add these columns to v6" — flagged
  during this session's scoping discussion, not started.
