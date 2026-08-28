import pytest

import starting_pitcher_predictions

GAMES = [
    {"game_pk": 111, "home_team_name": "Toronto Blue Jays", "away_team_name": "Los Angeles Angels"},
    {"game_pk": 222, "home_team_name": "Baltimore Orioles", "away_team_name": "Oakland Athletics"},
]


class TestGetPlaceholderPredictions:
    def test_returns_two_pitcher_rows_per_game(self):
        result = starting_pitcher_predictions.get_placeholder_predictions(GAMES)

        assert len(result["pitchers"]) == 4

    def test_each_row_carries_the_real_game_pk_and_team_names(self):
        result = starting_pitcher_predictions.get_placeholder_predictions(GAMES)

        rows_for_game_111 = [p for p in result["pitchers"] if p["game_pk"] == 111]
        assert len(rows_for_game_111) == 2
        teams = {p["team"] for p in rows_for_game_111}
        assert teams == {"Toronto Blue Jays", "Los Angeles Angels"}

    def test_each_pitcher_row_has_the_expected_fields(self):
        result = starting_pitcher_predictions.get_placeholder_predictions(GAMES)

        for row in result["pitchers"]:
            for field in (
                "game_pk", "pitcher_name", "team", "opponent",
                "batters_faced_pred", "strikeouts_pred", "early_out_probability",
                "strikeout_line", "strikeout_over_odds", "strikeout_under_odds",
                "model_prob_strikeouts_over", "fair_prob_strikeouts_over", "strikeouts_edge",
            ):
                assert field in row, f"missing field: {field}"

    def test_strikeouts_edge_matches_the_devig_and_edge_math(self):
        import betting_edge

        result = starting_pitcher_predictions.get_placeholder_predictions(GAMES)

        for row in result["pitchers"]:
            fair = betting_edge.devig_two_way(row["strikeout_over_odds"], row["strikeout_under_odds"])
            expected_edge = betting_edge.calculate_edge(row["model_prob_strikeouts_over"], fair["fair_prob_over"])
            assert row["fair_prob_strikeouts_over"] == pytest.approx(fair["fair_prob_over"])
            assert row["strikeouts_edge"] == pytest.approx(expected_edge)

    def test_opponent_is_the_other_team_in_the_same_game(self):
        result = starting_pitcher_predictions.get_placeholder_predictions(GAMES)

        for row in result["pitchers"]:
            assert row["opponent"] != row["team"]

    def test_flags_itself_as_sample_data(self):
        result = starting_pitcher_predictions.get_placeholder_predictions(GAMES)

        assert result["is_sample_data"] is True

    def test_empty_games_list_returns_empty_pitchers_list(self):
        result = starting_pitcher_predictions.get_placeholder_predictions([])

        assert result["pitchers"] == []
        assert result["is_sample_data"] is True
