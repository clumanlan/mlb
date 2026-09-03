import math

import numpy as np
from scipy.stats import nbinom, poisson


def poisson_binomial_pmf(probabilities: list[float]) -> np.ndarray:
    """Exact discrete probability distribution over the number of successes
    (0..N) from N independent Bernoulli trials with DIFFERENT probabilities
    p_1..p_N — unlike a standard binomial, which requires identical p across
    trials. This is the combinator any total-count model built on
    game_context.build_batter_slot_expansion needs: each synthetic batter
    slot has its own distinct predicted event probability (K, BB, hit, ...),
    so a plain binomial can't combine them.

    Standard DP/convolution recursion: build the distribution incrementally,
    one trial at a time, each step convolving in that trial's own (1-p, p).
    O(N^2), exact — no simulation, no approximation.
    """

    pmf = np.array([1.0])
    for p in probabilities:
        pmf = np.concatenate([pmf, [0.0]]) * (1 - p) + np.concatenate([[0.0], pmf]) * p
    return pmf


def poisson_binomial_mixture_pmf(probabilities: list[float], n_pmf: np.ndarray) -> np.ndarray:
    """Total-count distribution when N (how many of `probabilities`'s trials
    actually occur) is itself uncertain, not fixed — generalizes
    poisson_binomial_pmf for a starter whose batters-faced count is a real
    distribution (game_context.build_batters_faced_distribution), not a
    point estimate.

    probabilities: per-slot P(success) for up to max_slots slots, in
    slot/lineup-cycle order (build_batter_slot_expansion's own ordering) —
    the first n of these are exactly the n trials that occur when N=n, since
    a start with N batters faced works through the lineup in that same
    order from the top.

    n_pmf: P(N=n) for n=0..len(probabilities), length len(probabilities)+1
    (build_batters_faced_distribution's own contract: always the full
    max_slots+1 length, zero-padded where a start count has no mass).

    Reduces exactly to poisson_binomial_pmf(probabilities) when n_pmf is a
    one-hot at n=len(probabilities) — the fixed-N combinator is the special
    case of this function, not a separate code path.
    """

    n_total = len(probabilities)
    mixture = np.zeros(n_total + 1)
    for n, weight in enumerate(n_pmf):
        if weight == 0:
            continue
        sub_pmf = poisson_binomial_pmf(probabilities[:n])
        mixture[: len(sub_pmf)] += weight * sub_pmf
    return mixture


def negative_binomial_pmf(mean: float, alpha: float, max_k: int) -> np.ndarray:
    """Exact NB2-parameterized negative binomial pmf array over k = 0..max_k,
    for a predicted mean mu and fitted dispersion alpha where
    variance = mu + alpha * mu^2 (statsmodels' NegativeBinomial NB2
    convention). Converts to scipy.stats.nbinom's own (n, p) convention
    (n = 1/alpha, p = n / (n + mean)) so nothing downstream needs to know or
    care which pmf builder produced the array -- same drop-in contract as
    poisson_binomial_pmf's output."""

    n = 1.0 / alpha
    p = n / (n + mean)
    return nbinom.pmf(np.arange(max_k + 1), n, p)


def poisson_pmf(mean: float, max_k: int) -> np.ndarray:
    """Poisson pmf array over k = 0..max_k for a predicted mean. Thin wrapper
    around scipy.stats.poisson, same drop-in contract as
    negative_binomial_pmf/poisson_binomial_pmf's output.

    k_predictor v13's batter-grain model relies on a real, closed-form
    property of the Poisson distribution to get here cheaply: a sum of
    independent Poisson(mean_i) variables is itself exactly
    Poisson(sum(mean_i)) — so combining several batters' predicted means
    into one start's total-K distribution needs no new combination
    algorithm, only this pmf builder applied to the already-summed mean."""
    return poisson.pmf(np.arange(max_k + 1), mean)


def prob_exceeds_line(pmf, line):
    """P(total > line) from a total-count pmf (poisson_binomial_pmf's own
    output, or any array indexed 0..N by count). DK totals lines are always
    half-points (e.g. 5.5), so "exceeds" always means the next whole number
    up -- floor(line) + 1 -- with no ties to worry about."""
    threshold = math.floor(line) + 1
    return float(sum(pmf[threshold:]))
