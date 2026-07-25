from src.data.modules.ingestion import GetData
import src.data.modules.preprocessing as prep

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# NOTES:
# Plate appearance level: rolling avg by position can be done at the game level.

pd.set_option('display.max_columns', None)

get_data = GetData()

# ---- 1. Load raw data from S3 ----
game_info_list = []
for season in range(2015, 2025):
    game_info = get_data.get_game_info(season=season)
    game_info_list.append(game_info)
game_info_raw = pd.concat(game_info_list)

player_info_raw = get_data.get_player_info()
schedule_raw = get_data.get_schedule()
playbyplay_raw = get_data.get_playbyplay()

# ---- 2. Process reference data (consistent types and columns) ----
schedule = prep.process_schedule(schedule_raw)
game_info = prep.process_game_info(game_info_raw)
game_info['gamepk'] = game_info['gamePk'].astype(str)
player_info = prep.process_player_info(player_info_raw)

# ---- 3. Join schedule + game info (weather, duration, game_date) ----
schedule = schedule.merge(
    game_info.drop(columns=['gamePk']),
    on='gamepk',
    how='left'
)


# PLAYBYPLAY NEEDS GAME INFO RELELVANT COLUMNS LIKE SEASON, GAME DATE 

# ---- 4. Process play-by-play and attach game context ----
playbyplay_raw['gamepk'] = playbyplay_raw['gamepk'].astype(str)
if 'batter' not in playbyplay_raw.columns and 'batter_id' in playbyplay_raw.columns:
    playbyplay_raw = playbyplay_raw.assign(batter=playbyplay_raw['batter_id'], pitcher=playbyplay_raw['pitcher_id'])
playbyplay = prep.prepare_playbyplay(playbyplay_raw, player_info, schedule)

pbp_verlander = playbyplay[playbyplay['pitcher_name']=='Justin Verlander'].reset_index(drop=True)

rel_cols = ['gamepk', 'play_id', 'pitch_number', 'pitch_type']
rel_pitch_types = ['Four-Seam Fastball', 'Slider', 'Curveball', 'Changeup', 'Sweeper', 'Cutter', 'Sinker']

pbp_verlander = pbp_verlander[pbp_verlander['pitch_type'].isin(rel_pitch_types)][rel_cols].sort_values(['gamepk', 'play_id', 'pitch_number'])

verlander_ab_sequences =  (
    pbp_verlander.groupby(['gamepk', 'play_id'])['pitch_type']
    .apply(list)
    .reset_index()
)

#  sequences we're trying to model
pitch_sequences = verlander_ab_sequences['pitch_type']

# create embedding table
pitch_types = pbp_verlander['pitch_type'].unique()
pitch_type_dict = {}
for i in range(len(pitch_types)):
    pitch_type_dict[i] = pitch_types[i]

pitch_type_dict[7] = 'START'
pitch_type_dict[8] = 'END'


# create vocab mappings 
ptoi = {p:i for i,p in pitch_type_dict.items()}


# create a count matrix
# add start/end tokens > adjust count in matrix

N = torch.zeros((9,9), dtype=torch.int32)

for s in pitch_sequences:
    s = ['START'] + s + ['END']

    for (p1, p2) in zip(s, s[1:]): # this zip trick is really cool because it stops it automatically at end
        ix1 = ptoi[p1]
        ix2 = ptoi[p2]
        N[ix1, ix2] += 1




import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(N, cmap='Blues')
for i in range(9):
    for j in range(9):
        plt.text(j, i, pitch_type_dict[i][0:3] + '/' + pitch_type_dict[j][0:3], 
                ha='center', va='bottom', fontsize=7)
        plt.text(j, i, N[i,j].item(), ha='center', va='top')
plt.axis('off')