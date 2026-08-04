

# NEW MODULE: SEASON STATS HERE -------------------


# ---------------------------------------------------------------------------- #
#                                BUILD FEATURES                                #
# ---------------------------------------------------------------------------- #

# ---------------------------- BATTER SEASON STATS --------------------------- #
season_batter_stats = (
    batter_boxscore
    .groupby(['personId', 'game_season'])
    [['h',  'k', 'bb', 'hr', 'ab', 'plate_appearances']].sum()
    .reset_index()
    .rename(columns={
            'h': 'season_total_h',
            'k': 'season_total_k',
            'bb': 'season_total_bb',
            'hr': 'season_total_hr',
            'ab': 'season_total_ab',
            'plate_appearances': 'season_total_plate_appearances'
        })
    .assign(
        season_ba = lambda x: np.round(x['season_total_h']/x['season_total_ab'], 2)
    )
)

last_season_batter_stats = (
    season_batter_stats
    .assign(
        game_season = lambda x: x['game_season']+1
    )
    .rename(columns={
        col: col.replace('season_', 'last_season_batter_')
        for col in season_batter_stats.columns
        if col.startswith('season_')
    })
)


# --------------------------- PITCHER BOXSCORE SEASON STATS --------------------------- #
season_pitcher_stats = (
    pitcher_boxscore
    .groupby(['personId', 'game_season'])
    [['h', 'r', 'er', 'bb', 'hr', 'k', 'p', 's', 'ip']].sum()
    .reset_index()
    .rename(columns={
        'h': 'season_total_h',
        'r': 'season_total_r',
        'k': 'season_total_k',
        'er': 'season_total_er',
        'bb': 'season_total_bb',
        'hr': 'season_total_hr',
        'p': 'season_total_p',
        's': 'season_total_s',
        'ip': 'season_total_ip'
    })
    .assign(
        season_whip = lambda x: np.round(
            (x['season_total_bb'] + x['season_total_h'])
            / x['season_total_ip'].replace(0, np.nan), 2),
        season_k_rate = lambda x: np.round(
            x['season_total_k']
            / x['season_total_ip'].replace(0, np.nan), 2),
        season_bb_rate = lambda x: np.round(
            x['season_total_bb']
            / x['season_total_ip'].replace(0, np.nan), 2),
        season_strike_rate = lambda x: np.round(
            x['season_total_s']
            / x['season_total_p'].replace(0, np.nan), 2),
        season_hr_rate = lambda x: np.round(
            x['season_total_hr']
            / x['season_total_ip'].replace(0, np.nan), 2),
    )
)
last_season_pitcher_stats = (
    season_pitcher_stats
    .assign(
        game_season = lambda x: x['game_season']+1
    )
    .rename(columns={
        col: col.replace('season_', 'last_season_pitcher_')
        for col in season_pitcher_stats.columns 
        if col.startswith('season_')
    })
)
