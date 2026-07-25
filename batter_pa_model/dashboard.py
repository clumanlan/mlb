import sys
from pathlib import Path

# Ensure repo root is on sys.path so shared/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.model_dashboard import run_dashboard

_HERE = Path(__file__).parent

CONFIG = {
    "model_name": "Batter PA — hit/no hit",
    "target": "is_hit",
    "pred": "pred_prob",
    "entity": "batter_name",
    "date": "game_date",
    "slice_cols": ["batSide", "pitcher_hand", "batting_order", "weather_condition"],
    "interaction_pairs": [
        ("batSide", "pitcher_hand"),
        ("batting_order", "pitcher_hand"),
    ],
    "data_path": str(_HERE / "data.csv"),
}

run_dashboard(CONFIG)
