import pandas as pd

MODEL_NAME = "n_pa_predictor_low_pa"


def summarize_predictions_by_game(predictions_df: pd.DataFrame, player_names: dict) -> dict:
    """
    Group n_pa_predictor_low_pa's per-batter predictions by game.

    Returns {gamepk_str: {model, total_batters, qualifying_count, batters}} where
    "batters" lists only the qualifying (flagged) batters, sorted highest-risk first.
    """
    if predictions_df.empty:
        return {}

    result = {}
    for gamepk, group in predictions_df.groupby("gamepk"):
        qualifying = group[group["qualifies"]].sort_values(
            "predicted_probability", ascending=False
        )
        result[str(gamepk)] = {
            "model": MODEL_NAME,
            "total_batters": len(group),
            "qualifying_count": len(qualifying),
            "batters": [
                {
                    "person_id": int(row["personId"]),
                    "name": player_names.get(int(row["personId"]), str(int(row["personId"]))),
                    "batting_order": int(row["batting_order"]),
                    "probability": float(row["predicted_probability"]),
                }
                for _, row in qualifying.iterrows()
            ],
        }
    return result
