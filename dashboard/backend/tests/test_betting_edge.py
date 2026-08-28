import pytest

import betting_edge


class TestAmericanOddsToProbability:
    def test_negative_odds(self):
        # -148 means bet 148 to win 100 -> implied prob = 148 / (148 + 100)
        assert betting_edge.american_odds_to_probability(-148) == pytest.approx(0.5968, abs=1e-4)

    def test_positive_odds(self):
        # +116 means bet 100 to win 116 -> implied prob = 100 / (116 + 100)
        assert betting_edge.american_odds_to_probability(116) == pytest.approx(0.4630, abs=1e-4)

    def test_even_odds_is_fifty_percent(self):
        assert betting_edge.american_odds_to_probability(100) == pytest.approx(0.5, abs=1e-4)


class TestDevigTwoWay:
    def test_fair_probabilities_sum_to_one(self):
        result = betting_edge.devig_two_way(over_odds=116, under_odds=-148)

        assert result["fair_prob_over"] + result["fair_prob_under"] == pytest.approx(1.0, abs=1e-9)

    def test_removes_the_vig_from_raw_implied_probabilities(self):
        # Raw implied probs sum to > 1 (the vig) — devigged should sum to exactly 1
        # and each fair prob should be proportionally smaller than its raw implied prob.
        raw_over = betting_edge.american_odds_to_probability(116)
        result = betting_edge.devig_two_way(over_odds=116, under_odds=-148)

        assert result["fair_prob_over"] < raw_over

    def test_symmetric_odds_devig_to_fifty_fifty(self):
        result = betting_edge.devig_two_way(over_odds=-110, under_odds=-110)

        assert result["fair_prob_over"] == pytest.approx(0.5, abs=1e-9)
        assert result["fair_prob_under"] == pytest.approx(0.5, abs=1e-9)


class TestCalculateEdge:
    def test_positive_edge_when_model_more_confident_than_market(self):
        edge = betting_edge.calculate_edge(model_prob_over=0.52, fair_prob_over=0.431)

        assert edge == pytest.approx(0.089, abs=1e-4)

    def test_negative_edge_when_model_less_confident_than_market(self):
        edge = betting_edge.calculate_edge(model_prob_over=0.30, fair_prob_over=0.431)

        assert edge < 0

    def test_zero_edge_when_model_matches_market(self):
        edge = betting_edge.calculate_edge(model_prob_over=0.431, fair_prob_over=0.431)

        assert edge == pytest.approx(0.0, abs=1e-9)
