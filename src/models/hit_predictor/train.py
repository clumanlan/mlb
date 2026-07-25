from src.data.modules.ingestion import GetData
import src.data.modules.preprocessing as prep

import pandas as pd
import numpy as np

# NOTES:
# Plate appearance level: rolling avg by position can be done at the game level.


# ---- 5. Plate-appearance outcome (one row per PA with wOBA) ----
def calculate_woba(df, plate_result_col='play_result'):
    """
    Calculate wOBA for each plate appearance.
    Unlisted events are NaN and should be filtered out.
    """
    hits = {'Single': 0.855, 'Double': 1.248, 'Triple': 1.575, 'Home Run': 2.014}
    walks = {'Walk': 0.697, 'Intent Walk': 0.697, 'Hit By Pitch': 0.727, 'Catcher Interference': 0.697}
    outs = [
        'Strikeout', 'Groundout', 'Flyout', 'Lineout', 'Pop Out',
        'Forceout', 'Grounded Into DP', 'Double Play', 'Triple Play',
        'Fielders Choice', 'Fielders Choice Out', 'Field Error',
        'Bunt Groundout', 'Bunt Pop Out', 'Bunt Lineout',
        'Strikeout Double Play', 'Field Out', 'Batter Out'
    ]
    woba_map = {**hits, **walks, **{out: 0 for out in outs}}
    df['woba'] = df[plate_result_col].map(woba_map)
    return df

outcome_cols = ['gamepk', 'inning', 'play_id', 'batter_id', 'pitcher_id', 'play_result']
# Use first row per PA for outcome (play_result is same within a play_id)
pa_outcome = (
    playbyplay[outcome_cols + (['batter_player_name', 'pitcher_player_name'] if 'batter_player_name' in playbyplay.columns else [])]
    .drop_duplicates(subset=['gamepk', 'play_id'], keep='first')
    .reset_index(drop=True)
)
pa_outcome = calculate_woba(pa_outcome)

# ---- 6. Pitch-level aggregates per plate appearance ----
if 'pitch_number' in playbyplay.columns:
    pitch_agg = (
        playbyplay.groupby(['gamepk', 'play_id'])
        .agg(pitch_number=('pitch_number', 'max'))
        .reset_index()
    )
else:
    pitch_agg = playbyplay[['gamepk', 'play_id']].drop_duplicates()
    pitch_agg['pitch_number'] = np.nan

# ---- 7. Build model dataframe: PA-level features + outcome ----
# One row per (gamepk, play_id): context from first row of playbyplay + pitch agg + outcome
pa_context_cols = [
    'gamepk', 'play_id', 'inning', 'half_inning',
    'game_date', 'game_datetime', 'game_type', 'game_season',
    'weather_condition', 'weather_temp', 'weather_wind', 'game_duration_minutes',
    'batter_id', 'pitcher_id', 'batter_team_id', 'pitcher_team_id',
    'batter_team_name', 'pitcher_team_name',
]
# Use names from prepare_playbyplay if present
if 'batter_player_name' in playbyplay.columns:
    pa_context_cols.extend(['batter_player_name', 'pitcher_player_name'])
elif 'batter_name' in playbyplay.columns:
    pa_context_cols.extend(['batter_name', 'pitcher_name'])
pa_context_cols = [c for c in pa_context_cols if c in playbyplay.columns]

pa_context = (
    playbyplay[pa_context_cols]
    .drop_duplicates(subset=['gamepk', 'play_id'], keep='first')
    .reset_index(drop=True)
)
pa_context = pa_context.merge(pitch_agg, on=['gamepk', 'play_id'], how='left')

model_df = pa_outcome[['gamepk', 'play_id', 'play_result', 'woba']].merge(
    pa_context,
    on=['gamepk', 'play_id'],
    how='inner'
)
# Drop PAs with no wOBA (unlisted events) for modeling
model_df = model_df[model_df['woba'].notna()].reset_index(drop=True)

# model_df is ready for feature engineering and training (one row per plate appearance)
