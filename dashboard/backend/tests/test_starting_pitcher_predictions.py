import starting_pitcher_predictions

GAMES = [
    {"game_pk": 111, "home_team_name": "Toronto Blue Jays", "away_team_name": "Los Angeles Angels"},
    {"game_pk": 222, "home_team_name": "Baltimore Orioles", "away_team_name": "Oakland Athletics"},
]

STAT_FIELDS = (
    "batters_faced_pred", "strikeouts_pred", "early_out_probability",
    "strikeout_line", "strikeout_over_odds", "strikeout_under_odds",
    "model_prob_strikeouts_over", "fair_prob_strikeouts_over", "strikeouts_edge",
)


class TestGetPredictions:
    def test_returns_two_pitcher_rows_per_game(self):
        result = starting_pitcher_predictions.get_predictions(GAMES)

        assert len(result["pitchers"]) == 4

    def test_each_row_carries_the_real_game_pk_and_team_names(self):
        result = starting_pitcher_predictions.get_predictions(GAMES)

        rows_for_game_111 = [p for p in result["pitchers"] if p["game_pk"] == 111]
        assert len(rows_for_game_111) == 2
        teams = {p["team"] for p in rows_for_game_111}
        assert teams == {"Toronto Blue Jays", "Los Angeles Angels"}

    def test_each_pitcher_row_has_the_expected_fields(self):
        result = starting_pitcher_predictions.get_predictions(GAMES)

        for row in result["pitchers"]:
            for field in ("game_pk", "pitcher_name", "team", "opponent") + STAT_FIELDS:
                assert field in row, f"missing field: {field}"

    def test_stat_fields_are_none_since_no_model_is_deployed(self):
        result = starting_pitcher_predictions.get_predictions(GAMES)

        for row in result["pitchers"]:
            for field in STAT_FIELDS:
                assert row[field] is None

    def test_opponent_is_the_other_team_in_the_same_game(self):
        result = starting_pitcher_predictions.get_predictions(GAMES)

        for row in result["pitchers"]:
            assert row["opponent"] != row["team"]

    def test_flags_itself_as_having_no_real_predictions(self):
        result = starting_pitcher_predictions.get_predictions(GAMES)

        assert result["has_predictions"] is False

    def test_empty_games_list_returns_empty_pitchers_list(self):
        result = starting_pitcher_predictions.get_predictions([])

        assert result["pitchers"] == []
        assert result["has_predictions"] is False


class TestPitcherNameFromLineup:
    LINEUPS_BY_GAME = {
        111: {"home_pitcher_id": 501, "away_pitcher_id": 502},
    }
    PLAYER_NAMES = {501: "Home Starter", 502: "Away Starter"}

    def test_pitcher_name_resolved_for_home_and_away_independently(self):
        result = starting_pitcher_predictions.get_predictions(
            GAMES, lineups_by_game=self.LINEUPS_BY_GAME, player_names=self.PLAYER_NAMES
        )
        rows = {p["team"]: p["pitcher_name"] for p in result["pitchers"] if p["game_pk"] == 111}
        assert rows["Toronto Blue Jays"] == "Home Starter"
        assert rows["Los Angeles Angels"] == "Away Starter"

    def test_pitcher_name_is_tbd_when_game_not_in_lineups_by_game(self):
        # game_pk 222 has no entry — lineup not confirmed yet
        result = starting_pitcher_predictions.get_predictions(
            GAMES, lineups_by_game=self.LINEUPS_BY_GAME, player_names=self.PLAYER_NAMES
        )
        rows_222 = [p for p in result["pitchers"] if p["game_pk"] == 222]
        assert all(p["pitcher_name"] == "TBD" for p in rows_222)

    def test_pitcher_name_is_tbd_when_lineup_has_no_pitcher_id(self):
        lineups = {111: {"home_pitcher_id": None, "away_pitcher_id": None}}
        result = starting_pitcher_predictions.get_predictions(
            GAMES, lineups_by_game=lineups, player_names=self.PLAYER_NAMES
        )
        rows_111 = [p for p in result["pitchers"] if p["game_pk"] == 111]
        assert all(p["pitcher_name"] == "TBD" for p in rows_111)

    def test_pitcher_name_is_tbd_when_pitcher_id_not_in_player_names(self):
        result = starting_pitcher_predictions.get_predictions(
            GAMES, lineups_by_game=self.LINEUPS_BY_GAME, player_names={}
        )
        rows_111 = [p for p in result["pitchers"] if p["game_pk"] == 111]
        assert all(p["pitcher_name"] == "TBD" for p in rows_111)

    def test_defaults_to_tbd_when_lineups_and_player_names_omitted(self):
        result = starting_pitcher_predictions.get_predictions(GAMES)
        assert all(p["pitcher_name"] == "TBD" for p in result["pitchers"])
