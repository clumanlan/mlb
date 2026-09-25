import pandas as pd

from batter_predictions import summarize_predictions_by_game

PLAYER_NAMES = {
    680779: "Low PA Guy",
    682927: "Bench Risk",
    687462: "Everyday Player",
}


def _predictions_df(rows):
    return pd.DataFrame(rows)


class TestSummarizePredictionsByGame:
    def test_groups_by_gamepk(self):
        df = _predictions_df([
            {"personId": 680779, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.48, "qualifies": False},
            {"personId": 682927, "gamepk": "824059", "batting_order": 6,
             "predicted_probability": 0.21, "qualifies": False},
        ])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        assert set(result.keys()) == {"823326", "824059"}

    def test_total_batters_counts_all_rows_for_the_game(self):
        df = _predictions_df([
            {"personId": 680779, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.48, "qualifies": False},
            {"personId": 682927, "gamepk": "823326", "batting_order": 6,
             "predicted_probability": 0.21, "qualifies": False},
        ])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        assert result["823326"]["total_batters"] == 2

    def test_qualifying_count_only_counts_qualifies_true(self):
        df = _predictions_df([
            {"personId": 680779, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.91, "qualifies": True},
            {"personId": 682927, "gamepk": "823326", "batting_order": 6,
             "predicted_probability": 0.21, "qualifies": False},
        ])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        assert result["823326"]["qualifying_count"] == 1

    def test_batters_list_only_includes_qualifying_batters(self):
        df = _predictions_df([
            {"personId": 680779, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.91, "qualifies": True},
            {"personId": 682927, "gamepk": "823326", "batting_order": 6,
             "predicted_probability": 0.21, "qualifies": False},
        ])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        person_ids = [b["person_id"] for b in result["823326"]["batters"]]
        assert person_ids == [680779]

    def test_batters_list_sorted_by_probability_descending(self):
        df = _predictions_df([
            {"personId": 680779, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.86, "qualifies": True},
            {"personId": 687462, "gamepk": "823326", "batting_order": 3,
             "predicted_probability": 0.95, "qualifies": True},
        ])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        probs = [b["probability"] for b in result["823326"]["batters"]]
        assert probs == sorted(probs, reverse=True)

    def test_batter_name_comes_from_lookup(self):
        df = _predictions_df([
            {"personId": 680779, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.91, "qualifies": True},
        ])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        assert result["823326"]["batters"][0]["name"] == "Low PA Guy"

    def test_batter_name_falls_back_to_person_id_when_not_in_lookup(self):
        df = _predictions_df([
            {"personId": 999999, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.91, "qualifies": True},
        ])
        result = summarize_predictions_by_game(df, {})
        assert result["823326"]["batters"][0]["name"] == "999999"

    def test_model_name_is_included(self):
        df = _predictions_df([
            {"personId": 680779, "gamepk": "823326", "batting_order": 9,
             "predicted_probability": 0.91, "qualifies": True},
        ])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        assert result["823326"]["model"] == "n_pa_predictor_low_pa"

    def test_empty_dataframe_returns_empty_dict(self):
        df = _predictions_df([])
        result = summarize_predictions_by_game(df, PLAYER_NAMES)
        assert result == {}
