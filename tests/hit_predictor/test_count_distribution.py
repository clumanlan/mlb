import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from models.hit_predictor.utils.count_distribution import (
    poisson_binomial_pmf,
    poisson_binomial_mixture_pmf,
    prob_exceeds_line,
    negative_binomial_pmf,
)


def test_poisson_binomial_pmf_single_trial_is_bernoulli():
    """N=1 is the trivial case: the distribution over 0/1 successes is just
    the Bernoulli distribution itself."""

    result = poisson_binomial_pmf([0.3])

    assert result == pytest.approx([0.7, 0.3])


def test_poisson_binomial_pmf_two_trials_different_probabilities():
    """The property a standard binomial CANNOT give: two trials with
    DIFFERENT probabilities. Hand-derived: P(0)=0.8*0.5=0.40,
    P(2)=0.2*0.5=0.10, P(1)=1-0.40-0.10=0.50 (0.2*0.5 + 0.8*0.5)."""

    result = poisson_binomial_pmf([0.2, 0.5])

    assert result == pytest.approx([0.40, 0.50, 0.10])


def test_poisson_binomial_pmf_matches_binomial_when_probabilities_equal():
    """A known-good special case: when every trial shares the same
    probability, this must reduce exactly to the textbook Binomial(3, 0.5)
    distribution — catches an off-by-one or misweighted convolution the
    other hand-derived cases might miss."""

    result = poisson_binomial_pmf([0.5, 0.5, 0.5])

    assert result == pytest.approx([0.125, 0.375, 0.375, 0.125])


def test_poisson_binomial_pmf_sums_to_one():
    """A real probability distribution over 0..N successes must sum to 1.0,
    for any N and any mix of probabilities — a standing regression guard."""

    probabilities = [0.1, 0.25, 0.4, 0.55, 0.7, 0.15, 0.3, 0.6, 0.05, 0.9]

    result = poisson_binomial_pmf(probabilities)

    assert sum(result) == pytest.approx(1.0)
    assert len(result) == len(probabilities) + 1


def test_poisson_binomial_mixture_pmf_reduces_to_plain_pmf_when_n_is_certain():
    """When N (how many of `probabilities`' trials actually occur) is
    certain — a one-hot n_pmf at n=len(probabilities) — the mixture must
    reduce exactly to poisson_binomial_pmf's own result. Proves the fixed-N
    combinator k_predictor's existing diagnostic already trusts is the
    special case of this more general function, not a separate code path
    that could drift from it."""

    probabilities = [0.2, 0.5, 0.7]
    n_pmf = [0.0, 0.0, 0.0, 1.0]

    result = poisson_binomial_mixture_pmf(probabilities, n_pmf)

    assert result == pytest.approx(poisson_binomial_pmf(probabilities))


def test_poisson_binomial_mixture_pmf_hand_derived_two_possible_n_values():
    """Hand-derived, independent of the reduction test above: two equally
    likely values of N. 50% chance N=1 (only the first slot happens — pmf
    [0.5, 0.5, 0]), 50% chance N=2 (both slots happen — pmf
    [0.25, 0.5, 0.25]). Weighted sum: 0.5*[0.5,0.5,0] + 0.5*[0.25,0.5,0.25]
    = [0.375, 0.5, 0.125]."""

    probabilities = [0.5, 0.5]
    n_pmf = [0.0, 0.5, 0.5]

    result = poisson_binomial_mixture_pmf(probabilities, n_pmf)

    assert result == pytest.approx([0.375, 0.5, 0.125])


def test_prob_exceeds_line_half_point_line():
    """DK lines are always half-points (e.g. 1.5). P(K > 1.5) means K>=2,
    i.e. indices 2 and 3 of the [0.125, 0.375, 0.375, 0.125] pmf
    (poisson_binomial_pmf([0.5, 0.5, 0.5]))."""
    pmf = [0.125, 0.375, 0.375, 0.125]

    result = prob_exceeds_line(pmf, 1.5)

    assert result == pytest.approx(0.375 + 0.125)


def test_prob_exceeds_line_lower_line_sums_more_mass():
    """A lower line (0.5) should sum more of the pmf's tail than a higher
    one (1.5) -- P(K > 0.5) means K>=1, indices 1, 2, and 3."""
    pmf = [0.125, 0.375, 0.375, 0.125]

    result = prob_exceeds_line(pmf, 0.5)

    assert result == pytest.approx(0.375 + 0.375 + 0.125)


def test_poisson_binomial_mixture_pmf_sums_to_one():
    """A real probability distribution over 0..N successes must sum to 1.0
    regardless of how uncertain N itself is — a standing regression guard,
    same shape as poisson_binomial_pmf's own sum-to-one test."""

    probabilities = [0.1, 0.25, 0.4, 0.55, 0.7, 0.15, 0.3, 0.6, 0.05, 0.9]
    n_pmf = [0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]

    result = poisson_binomial_mixture_pmf(probabilities, n_pmf)

    assert sum(result) == pytest.approx(1.0)
    assert len(result) == len(probabilities) + 1


class TestNegativeBinomialPmf:
    def test_negative_binomial_pmf_matches_scipy_nbinom_at_known_params(self):
        """Pins the NB2 parameterization convention (variance = mean +
        alpha*mean^2, i.e. n=1/alpha, p=n/(n+mean)) against scipy's own
        (n, p) convention, so a future reader can't silently flip mean/
        variance."""
        mean, alpha, max_k = 5.0, 0.3, 20
        n = 1.0 / alpha
        p = n / (n + mean)
        expected = nbinom.pmf(np.arange(max_k + 1), n, p)

        result = negative_binomial_pmf(mean, alpha, max_k)

        assert result == pytest.approx(expected)

    def test_negative_binomial_pmf_small_alpha_approximates_poisson(self):
        """As alpha -> 0, NB2 converges to Poisson(mean) -- a real limiting
        property. Also implicitly guards against a division-by-zero on
        n = 1/alpha blowing up numerically for small-but-nonzero alpha."""
        mean, max_k = 4.0, 20
        expected = poisson.pmf(np.arange(max_k + 1), mean)

        result = negative_binomial_pmf(mean, 1e-6, max_k)

        assert result == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize("mean", [2.0, 5.0, 9.0])
    @pytest.mark.parametrize("alpha", [0.05, 0.3, 1.0])
    def test_negative_binomial_pmf_is_a_valid_distribution(self, mean, alpha):
        """0.95 rather than 0.99 -- at the most dispersed corner of this grid
        (alpha=1.0, mean=9.0, variance=90) NB2's tail genuinely carries more
        than 1% of its mass past k=30 (verified against scipy.stats.nbinom's
        own CDF directly, not an implementation artifact), so 0.99 would fail
        on correct output. 0.95 still catches a badly broken pmf (wrong
        normalization, NaNs) while accommodating that real truncation loss."""
        max_k = 30

        result = negative_binomial_pmf(mean, alpha, max_k)

        assert len(result) == max_k + 1
        assert (result >= 0).all()
        assert sum(result) >= 0.95
