from models.hit_predictor.processing.schema import PBP

# n_pa_predictor operates at batter-GAME grain (one row per starter per
# game), not per-PA — a different grain from hit_predictor's is_hit model.
#
# PA_OUTCOMES here starts from hit_predictor's PBP.PA_OUTCOMES (the shared
# "what counts as a plate appearance" set) but adds two play_result values
# missing from it: 'Field Error' (reached on a fielding error — indisputably
# a completed PA, batter put the ball in play) and 'Sac Fly Double Play'
# (same category as 'Sac Fly', which IS included). Discovered 2026-08-22 by
# comparing this model's starter-game count against hit_predictor's own
# create_pa_outcome output for the same season: without these two, 4 starters
# whose ONLY plate appearance in a game was one of these two outcomes were
# silently dropped entirely (label had no row for them), and every batter who
# reached on error at least once was undercounted by 1 PA. 'Field Error'
# alone is ~1,100 PAs/season (2024) — not a rounding error.
#
# This is a real gap in hit_predictor's canonical PBP.PA_OUTCOMES too (it
# affects hit_predictor's own season/rolling "PA denominator" stats and TTO
# capping the same way), but fixing that shared definition is a bigger-
# blast-radius change belonging to hit_predictor's own maintainer, not made
# unilaterally here — flagged in ROADMAP.md instead. n_pa_predictor's own
# label needs the accurate count regardless, so it's extended locally.
PA_OUTCOMES = PBP.PA_OUTCOMES | {'Field Error', 'Sac Fly Double Play'}

__all__ = ['PBP', 'PA_OUTCOMES']
