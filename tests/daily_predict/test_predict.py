import importlib.util
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import xgboost as xgb

# Loaded by direct file path, NOT via sys.path — src/lambdas/daily_predict/
# also contains handler.py, the same generic filename every other Lambda
# uses, so it's deliberately never added to the shared/session-lifetime
# sys.path (see conftest.py's comment). predict.py's own top-level import
# of batter_lineup still resolves normally via src/features/transforms,
# which conftest.py does add — that directory has no name-collision risk.
_predict_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../src/lambdas/daily_predict/predict.py")
)
_spec = importlib.util.spec_from_file_location("n_pa_predict_predict", _predict_path)
predict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict)


def _fake_model_info(threshold=0.85, version="2026-09-22T00-00-00"):
    # Trained on the SAME column names the real model was trained on
    # (batter_n_pa_roll_season_avg_n_pa_per_game) — this is deliberately
    # NOT the Feast-facing name (batter_pa_roll_season_avg_n_pa_per_game,
    # see batter_pa_volume.py's rename), to catch predict_for_date if it
    # forgets to rename the online-store column back before predicting.
    X = pd.DataFrame(
        [[1, 4.0], [9, 1.0]] * 3,
        columns=["batting_order", "batter_n_pa_roll_season_avg_n_pa_per_game"],
    )
    y = [0, 1] * 3
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
    model.fit(X, y)
    return {"model": model, "operating_threshold": threshold, "version": version}


def _fake_feature_store(online_rows):
    store = MagicMock()
    store.get_online_features.return_value.to_df.return_value = pd.DataFrame(online_rows)
    return store


def _fake_lineup_df():
    return pd.DataFrame([
        {"personId": "1", "gamepk": "g1", "game_date": "2026-09-22", "batting_order": 1},
        {"personId": "2", "gamepk": "g1", "game_date": "2026-09-22", "batting_order": 9},
    ])


class TestPredictForDate:
    def test_requests_the_right_feast_feature_reference(self):
        feature_store = _fake_feature_store([
            {"personId": "1", "batter_pa_roll_season_avg_n_pa_per_game": 4.0},
            {"personId": "2", "batter_pa_roll_season_avg_n_pa_per_game": 1.0},
        ])
        with patch.object(predict, 'compute_batter_lineup_features_for_date', return_value=_fake_lineup_df()):
            predict.predict_for_date("2026-09-22", feature_store, _fake_model_info())

        requested_features = feature_store.get_online_features.call_args.kwargs["features"]
        assert requested_features == ["batter_pa_volume_fv:batter_pa_roll_season_avg_n_pa_per_game"]

    def test_requests_entity_rows_for_every_batter_in_the_lineup(self):
        feature_store = _fake_feature_store([
            {"personId": "1", "batter_pa_roll_season_avg_n_pa_per_game": 4.0},
            {"personId": "2", "batter_pa_roll_season_avg_n_pa_per_game": 1.0},
        ])
        with patch.object(predict, 'compute_batter_lineup_features_for_date', return_value=_fake_lineup_df()):
            predict.predict_for_date("2026-09-22", feature_store, _fake_model_info())

        entity_rows = feature_store.get_online_features.call_args.kwargs["entity_rows"]
        assert entity_rows == [{"personId": "1"}, {"personId": "2"}]

    def test_renames_feast_column_to_the_models_training_column_name(self):
        # The real gotcha: batter_pa_volume.py renames the training-pipeline
        # column to a different Feast-facing name on the way INTO the store.
        # predict_for_date must rename it back before calling predict_proba,
        # or this raises instead of silently mispredicting.
        feature_store = _fake_feature_store([
            {"personId": "1", "batter_pa_roll_season_avg_n_pa_per_game": 4.0},
            {"personId": "2", "batter_pa_roll_season_avg_n_pa_per_game": 1.0},
        ])
        with patch.object(predict, 'compute_batter_lineup_features_for_date', return_value=_fake_lineup_df()):
            result = predict.predict_for_date("2026-09-22", feature_store, _fake_model_info())

        assert len(result) == 2
        assert result["predicted_probability"].notna().all()

    def test_qualifies_flag_reflects_operating_threshold(self):
        feature_store = _fake_feature_store([
            {"personId": "1", "batter_pa_roll_season_avg_n_pa_per_game": 4.0},
            {"personId": "2", "batter_pa_roll_season_avg_n_pa_per_game": 1.0},
        ])
        with patch.object(predict, 'compute_batter_lineup_features_for_date', return_value=_fake_lineup_df()):
            result = predict.predict_for_date("2026-09-22", feature_store, _fake_model_info(threshold=0.0))

        # threshold=0.0 means every row qualifies, regardless of probability
        assert result["qualifies"].all()

    def test_attaches_model_version_to_every_row(self):
        feature_store = _fake_feature_store([
            {"personId": "1", "batter_pa_roll_season_avg_n_pa_per_game": 4.0},
            {"personId": "2", "batter_pa_roll_season_avg_n_pa_per_game": 1.0},
        ])
        with patch.object(predict, 'compute_batter_lineup_features_for_date', return_value=_fake_lineup_df()):
            result = predict.predict_for_date(
                "2026-09-22", feature_store, _fake_model_info(version="2026-09-22T09-00-00")
            )

        assert (result["model_version"] == "2026-09-22T09-00-00").all()

    def test_empty_lineup_returns_empty_dataframe_with_output_schema(self):
        empty_lineup = pd.DataFrame(columns=["personId", "gamepk", "game_date", "batting_order"])
        feature_store = _fake_feature_store([])
        with patch.object(predict, 'compute_batter_lineup_features_for_date', return_value=empty_lineup):
            result = predict.predict_for_date("2026-09-22", feature_store, _fake_model_info())

        assert result.empty
        assert list(result.columns) == predict.OUTPUT_COLUMNS
        feature_store.get_online_features.assert_not_called()
