"""
Placeholder starting-pitcher prediction data — settles Section 4's UI shape ahead of
a real inference path. None of batters_faced_predictor, k_predictor, or
short_outing_predictor have a production inference path yet (Layer 6 in CLAUDE.md's
architecture is "not started" for any model), so there's nothing real to serve here.

Attached to today's REAL schedule (game_pk, team names) so the game <-> pitcher
visual link on the frontend can be built and tried against real games — only the
three prediction numbers themselves are fake. Swap the fake numbers for real
model output once a model is wired up; the game_pk/team plumbing stays as-is.

The strikeout line/odds (strikeout_line, strikeout_over_odds, strikeout_under_odds)
are shaped like a real DraftKings pitcher_strikeouts market row (see
raw_data/odds/player_props/{year}/{date}.parquet) but are also fake for now — the
odds pipeline is mid-outage (see project_odds_fetch_outage memory), so there's no
live line to attach even if a real model existed. model_prob_strikeouts_over is a
placeholder stand-in for "the model's P(actual Ks > line)"; devig_two_way/
calculate_edge (betting_edge.py) are the REAL, tested math — only their inputs are
fake here. Swap in a real market line and a real model probability later; the edge
calculation itself doesn't change.
"""

import betting_edge

_SAMPLE_STAT_SETS = [
    {
        "batters_faced_pred": 24.3, "strikeouts_pred": 5.8, "early_out_probability": 0.18,
        "strikeout_line": 5.5, "strikeout_over_odds": 116, "strikeout_under_odds": -148,
        "model_prob_strikeouts_over": 0.52,
    },
    {
        "batters_faced_pred": 21.7, "strikeouts_pred": 4.9, "early_out_probability": 0.27,
        "strikeout_line": 4.5, "strikeout_over_odds": -116, "strikeout_under_odds": -110,
        "model_prob_strikeouts_over": 0.48,
    },
    {
        "batters_faced_pred": 26.1, "strikeouts_pred": 6.4, "early_out_probability": 0.12,
        "strikeout_line": 6.5, "strikeout_over_odds": 108, "strikeout_under_odds": -138,
        "model_prob_strikeouts_over": 0.61,
    },
    {
        "batters_faced_pred": 19.4, "strikeouts_pred": 3.6, "early_out_probability": 0.35,
        "strikeout_line": 3.5, "strikeout_over_odds": -133, "strikeout_under_odds": 104,
        "model_prob_strikeouts_over": 0.39,
    },
]


def _with_edge(stats: dict) -> dict:
    fair = betting_edge.devig_two_way(stats["strikeout_over_odds"], stats["strikeout_under_odds"])
    edge = betting_edge.calculate_edge(stats["model_prob_strikeouts_over"], fair["fair_prob_over"])
    return {**stats, "fair_prob_strikeouts_over": fair["fair_prob_over"], "strikeouts_edge": edge}


def get_placeholder_predictions(games: list) -> dict:
    pitchers = []
    for i, game in enumerate(games):
        home_team = game["home_team_name"]
        away_team = game["away_team_name"]
        for slot, (team, opponent) in enumerate([(home_team, away_team), (away_team, home_team)]):
            stats = _SAMPLE_STAT_SETS[(i * 2 + slot) % len(_SAMPLE_STAT_SETS)]
            pitchers.append({
                "game_pk": game["game_pk"],
                "pitcher_name": "TBD",
                "team": team,
                "opponent": opponent,
                **_with_edge(stats),
            })

    return {"pitchers": pitchers, "is_sample_data": True}
