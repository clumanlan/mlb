import sys
from pathlib import Path

# Ensure repo root is on sys.path so shared/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from shared.model_dashboard import run_dashboard

_HERE = Path(__file__).parent

CONFIG = {
    "model_name": "k_predictor v6 — pitcher PA strikeout probability",
    "target": "is_strikeout",
    "pred": "pred_prob",
    "entity": "pitcher_name",
    "date": "game_date",
    "slice_cols": [
        "platoon_matchup", "pitcher_throw_hand", "batter_bat_side",
        "expected_pitcher_role", "weather_condition",
    ],
    "interaction_pairs": [
        ("batter_bat_side", "pitcher_throw_hand"),
        ("expected_pitcher_role", "platoon_matchup"),
    ],
    "data_path": str(_HERE / "data.csv"),
}

run_dashboard(CONFIG)
