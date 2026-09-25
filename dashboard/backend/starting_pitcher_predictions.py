"""
No real model wired up yet — stat fields are None until batters_faced_predictor,
k_predictor, or short_outing_predictor gets a production inference path (Layer 6
in CLAUDE.md's architecture is "not started" for all three).

Attached to today's REAL schedule (game_pk, team names) so the game <-> pitcher
visual link on the frontend stays buildable/testable against real games.
pitcher_name IS real, though: daily_lineup_fetch already writes real
home_pitcher_id/away_pitcher_id into each confirmed game's lineup file
(raw_data/games/lineups/{year}/{date}/{game_pk}.json), resolved to a name via
lineups_by_game + player_names (both passed in by the caller — this module stays
S3-free). Falls back to "TBD" whenever the lineup isn't confirmed yet, has no
pitcher_id, or the id isn't in the player_names lookup.
"""

STAT_FIELDS = (
    "batters_faced_pred", "strikeouts_pred", "early_out_probability",
    "strikeout_line", "strikeout_over_odds", "strikeout_under_odds",
    "model_prob_strikeouts_over", "fair_prob_strikeouts_over", "strikeouts_edge",
)


def _pitcher_name(pitcher_id, player_names: dict) -> str:
    if pitcher_id is None:
        return "TBD"
    return player_names.get(pitcher_id, "TBD")


def get_predictions(games: list, lineups_by_game: dict = None, player_names: dict = None) -> dict:
    lineups_by_game = lineups_by_game or {}
    player_names = player_names or {}

    pitchers = []
    for game in games:
        game_pk = game["game_pk"]
        lineup = lineups_by_game.get(game_pk, {})
        home_team = game["home_team_name"]
        away_team = game["away_team_name"]
        matchups = [
            (home_team, away_team, lineup.get("home_pitcher_id")),
            (away_team, home_team, lineup.get("away_pitcher_id")),
        ]
        for team, opponent, pitcher_id in matchups:
            pitchers.append({
                "game_pk": game_pk,
                "pitcher_name": _pitcher_name(pitcher_id, player_names),
                "team": team,
                "opponent": opponent,
                **{field: None for field in STAT_FIELDS},
            })

    return {"pitchers": pitchers, "has_predictions": False}
