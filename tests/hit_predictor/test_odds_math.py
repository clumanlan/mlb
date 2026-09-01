import pytest

from models.hit_predictor.utils.odds_math import (
    american_to_implied_prob,
    devig_two_way,
    compute_edge,
)


def test_american_to_implied_prob_negative_odds():
    """-110 -> 110/210, the standard textbook conversion."""
    assert american_to_implied_prob(-110) == pytest.approx(110 / 210)


def test_american_to_implied_prob_positive_odds():
    """+120 -> 100/220."""
    assert american_to_implied_prob(120) == pytest.approx(100 / 220)


def test_american_to_implied_prob_matches_benchmarks_cited_example():
    """BENCHMARKS.md cites -115 implies ~53.5% breakeven -- a sanity check
    on the sign convention against a number already vetted in this repo."""
    assert american_to_implied_prob(-115) == pytest.approx(0.535, abs=0.001)


def test_devig_two_way_balanced_market_sums_to_one():
    """A symmetric -110/-110 market devigs to an exact 50/50 split."""
    p_over, p_under = devig_two_way(-110, -110)
    assert p_over == pytest.approx(0.5)
    assert p_under == pytest.approx(0.5)
    assert p_over + p_under == pytest.approx(1.0)


def test_devig_two_way_asymmetric_market_preserves_ratio_and_sums_to_one():
    """Proportional (multiplicative) devig: the devigged pair always sums
    to 1.0, and preserves the same ratio as the raw (vig-inflated) implied
    probabilities."""
    raw_over = american_to_implied_prob(-150)
    raw_under = american_to_implied_prob(130)

    p_over, p_under = devig_two_way(-150, 130)

    assert p_over + p_under == pytest.approx(1.0)
    assert p_over / p_under == pytest.approx(raw_over / raw_under)


def test_compute_edge_is_model_minus_market():
    assert compute_edge(0.65, 0.55) == pytest.approx(0.10)
    assert compute_edge(0.40, 0.55) == pytest.approx(-0.15)
