from models.hit_predictor.processing.schema import PBP

# Strikeout as a completed PA outcome (play_result values). 'Strikeout Double
# Play' is the same event category as 'Strikeout' — a K where the runner is
# also erased.
STRIKEOUTS = {"Strikeout", "Strikeout Double Play"}

# Extends hit_predictor's PBP.PA_OUTCOMES with two play_result values it's
# missing ('Field Error', 'Sac Fly Double Play' — both completed PAs, see
# ROADMAP.md). Same local-fix pattern n_pa_predictor already applied for its
# own PA denominator, applied here for k_predictor's.
PA_OUTCOMES = PBP.PA_OUTCOMES | {'Field Error', 'Sac Fly Double Play'}

__all__ = ['PBP', 'STRIKEOUTS', 'PA_OUTCOMES']
