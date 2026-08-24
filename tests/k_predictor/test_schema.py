from models.hit_predictor.processing.schema import PBP
from models.k_predictor.processing.schema import STRIKEOUTS, PA_OUTCOMES


def test_strikeouts_is_strikeout_and_strikeout_double_play():
    assert STRIKEOUTS == {"Strikeout", "Strikeout Double Play"}


def test_pa_outcomes_extends_hit_predictor_pa_outcomes_with_field_error_and_sac_fly_dp():
    """Same known gap n_pa_predictor already worked around locally (see
    ROADMAP.md): hit_predictor's shared PBP.PA_OUTCOMES is missing 'Field
    Error' and 'Sac Fly Double Play' — both are completed PAs. k_predictor
    needs the accurate PA denominator for its own rate features, same as
    n_pa_predictor needed it for its own n_pa label."""
    assert PA_OUTCOMES == PBP.PA_OUTCOMES | {'Field Error', 'Sac Fly Double Play'}
