import pytest

from models.hit_predictor.utils.count_distribution import (
    poisson_binomial_pmf,
    poisson_binomial_mixture_pmf,
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


def test_poisson_binomial_mixture_pmf_sums_to_one():
    """A real probability distribution over 0..N successes must sum to 1.0
    regardless of how uncertain N itself is — a standing regression guard,
    same shape as poisson_binomial_pmf's own sum-to-one test."""

    probabilities = [0.1, 0.25, 0.4, 0.55, 0.7, 0.15, 0.3, 0.6, 0.05, 0.9]
    n_pmf = [0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]

    result = poisson_binomial_mixture_pmf(probabilities, n_pmf)

    assert sum(result) == pytest.approx(1.0)
    assert len(result) == len(probabilities) + 1
