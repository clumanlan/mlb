"""
Devig + edge math for comparing a model's predicted probability against a DraftKings
prop line. This is the "economic eval layer" CLAUDE.md flags as deferred, not-yet-built
infrastructure — built now because the math itself doesn't depend on whether the model
probability feeding it is real inference or a placeholder (see starting_pitcher_predictions.py).

Devig method: simple multiplicative normalization (raw implied probabilities, scaled
down so the two-way pair sums to exactly 1). This is the standard baseline devig
approach — good enough to remove the bulk of the vig; more sophisticated methods
(power, Shin) are a possible future refinement, not needed to get a real edge number.
"""


def american_odds_to_probability(odds: int) -> float:
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def devig_two_way(over_odds: int, under_odds: int) -> dict:
    raw_over = american_odds_to_probability(over_odds)
    raw_under = american_odds_to_probability(under_odds)
    overround = raw_over + raw_under
    return {
        "fair_prob_over": raw_over / overround,
        "fair_prob_under": raw_under / overround,
    }


def calculate_edge(model_prob_over: float, fair_prob_over: float) -> float:
    return model_prob_over - fair_prob_over
