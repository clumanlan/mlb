import pandas as pd

from batter_lineup import compute_batter_lineup_features_for_date

FEATURE_VIEW = "batter_pa_volume_fv"

# Feast-facing name (what get_online_features returns) vs. the name the
# model was actually trained on — batter_pa_volume.py renames on the way
# INTO the store (see its OUTPUT_COLUMNS comment), so this is the same
# rename applied in reverse before scoring.
FEAST_FEATURE_NAME = "batter_pa_roll_season_avg_n_pa_per_game"
MODEL_FEATURE_NAME = "batter_n_pa_roll_season_avg_n_pa_per_game"

OUTPUT_COLUMNS = [
    "personId", "gamepk", "game_date", "batting_order",
    MODEL_FEATURE_NAME,
    "predicted_probability", "qualifies", "model_version",
]


def predict_for_date(date, feature_store, model_info):
    lineup_df = compute_batter_lineup_features_for_date(date)
    if lineup_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    entity_rows = [{"personId": pid} for pid in lineup_df["personId"]]
    online_df = feature_store.get_online_features(
        features=[f"{FEATURE_VIEW}:{FEAST_FEATURE_NAME}"],
        entity_rows=entity_rows,
    ).to_df()
    online_df = online_df.rename(columns={FEAST_FEATURE_NAME: MODEL_FEATURE_NAME})

    merged = lineup_df.merge(online_df, on="personId", how="left")

    model = model_info["model"]
    X = merged[model.get_booster().feature_names]
    merged["predicted_probability"] = model.predict_proba(X)[:, 1]
    merged["qualifies"] = merged["predicted_probability"] >= model_info["operating_threshold"]
    merged["model_version"] = model_info["version"]

    return merged[OUTPUT_COLUMNS]
