"""
Market-odds math, model-agnostic -- American odds in, probabilities out.
Nothing here knows about strikeouts, hits, or any specific prop; any model
comparing its own probability to a two-sided DK line can use this.
"""


def american_to_implied_prob(odds):
    """American odds -> raw (vig-inflated) implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def devig_two_way(odds_a, odds_b):
    """Remove the vig from a two-sided market via the proportional
    (multiplicative) method: scale both sides' raw implied probabilities so
    they sum to exactly 1.0, preserving their relative ratio. See
    BENCHMARKS.md's vig-math citation -- a balanced -110/-110 market holds
    ~4.8%; devigging is required before comparing a market line to a
    calibrated model probability."""
    p_a = american_to_implied_prob(odds_a)
    p_b = american_to_implied_prob(odds_b)
    total = p_a + p_b
    return p_a / total, p_b / total


def compute_edge(model_prob_over, market_prob_over):
    """Model's probability minus the devigged market's probability, for the
    SAME side of the SAME line. Positive means the model thinks the outcome
    is more likely than the market prices it; negative means less likely."""
    return model_prob_over - market_prob_over
