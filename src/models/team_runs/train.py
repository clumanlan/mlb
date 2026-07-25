import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
import plotly.express as px
from data.modules.ingestion import GetData
pd.set_option("display.max_columns", None)

get_data = GetData()
playbyplay = get_data.get_playbyplay(season=2025)

game_info = get_data.get_game_info()
schedule = get_data.get_schedule()
player_info = get_data.get_player_info()

player_info.columns.tolist()


_XGB_PARAMS = dict(
    objective='count:poisson',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    min_child_weight=5,
    random_state=42,
)

TRAIN_SEASONS = [2018, 2019, 2021, 2022, 2023]
VAL_SEASONS   = [2024]
PITCHER_SEASONS = [2018, 2019, 2021, 2022, 2023, 2024]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batter_data(seasons) -> pd.DataFrame:
    """Load and concatenate batter boxscore for each season.

    Processed batter data already includes game_date (datetime) and game_season (Int32).
    No schedule merge needed.
    # TODO: processed batter_boxscore has no team_name column (only player_name).
    #       build_team_batting agg will fail on team_name=('team_name', 'first').
    #       Add team_name to the processing step or remove from agg.
    """

    get_data = GetData()

    batter_dfs = []
    for _season in seasons:
        batter_dfs.append(get_data.get_batter_boxscore(season=_season))
    batter_boxscore = pd.concat(batter_dfs, ignore_index=True)
    return batter_boxscore


def load_pitcher_data() -> pd.DataFrame:
    """Load and concatenate pitcher boxscore for each season.

    Processed pitcher data already includes game_date (datetime).
    game_season is derived from game_date since it is absent from the processed schema.
    # TODO: processed pitcher_boxscore has no team_id column.
    #       add_sp_features starter fallback and SP join on ['gamepk', 'team_id'] will break.
    #       Add team_id to the processing step.
    """
    get_data = GetData()
    

    pitcher_dfs = []
    for _season in seasons:
        pitcher_dfs.append(get_data.get_pitcher_boxscore(season=_season))
    pitcher_raw = pd.concat(pitcher_dfs, ignore_index=True)

    return pitcher_raw


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_team_batting(batter_boxscore: pd.DataFrame) -> pd.DataFrame:
    """Aggregate batter data to team level; add rate stats and wOBA."""
    player_cols_to_drop = [
        'namefield', 'personId', 'batting_order', 'substitution', 'note',
        'name', 'nameposition', 'position', 'avg', 'obp', 'slg', 'ops',
    ]
    batter = batter_boxscore.drop(columns=[c for c in player_cols_to_drop if c in batter_boxscore.columns])

    counting_stats = ['ab', 'r', 'h', 'doubles', 'triples', 'hr', 'rbi', 'bb', 'k', 'lob', 'sb']
    team_batting = (
        batter
        .groupby(['team_id', 'gamepk', 'game_season'], as_index=False)
        .agg(
            # Fix 4: removed team_name — not in processed batter schema, would KeyError at runtime
            game_date=('game_date', 'first'),
            **{col: (col, 'sum') for col in counting_stats},
        )
    )

    # SLG numerator: singles=1, doubles=2, triples=3, hr=4
    # singles = h - doubles - triples - hr  →  total_bases = h + doubles + 2*triples + 3*hr
    team_batting['total_bases'] = (
        team_batting['h']
        + team_batting['doubles']
        + 2 * team_batting['triples']
        + 3 * team_batting['hr']
    )

    # OBP: (H + BB) / (AB + BB) — simplified; HBP and SF not available in this dataset
    team_batting['obp'] = (team_batting['h'] + team_batting['bb']) / (team_batting['ab'] + team_batting['bb'])
    team_batting['slg'] = team_batting['total_bases'] / team_batting['ab']
    team_batting['avg'] = team_batting['h'] / team_batting['ab']
    team_batting['ops'] = team_batting['obp'] + team_batting['slg']

    rate_cols = ['obp', 'slg', 'avg', 'ops']
    team_batting[rate_cols] = team_batting[rate_cols].replace([np.inf, -np.inf], np.nan)

    # wOBA (Weighted On-Base Average, FanGraphs weights)
    # HBP is not available in this dataset; set to 0
    # wOBA = (0.69*BB + 0.72*HBP + 0.89*1B + 1.27*2B + 1.62*3B + 2.10*HR) / (AB + BB + HBP)
    singles = team_batting['h'] - team_batting['doubles'] - team_batting['triples'] - team_batting['hr']
    hbp = 0
    team_batting['woba'] = (
        0.69 * team_batting['bb']
        + 0.72 * hbp
        + 0.89 * singles
        + 1.27 * team_batting['doubles']
        + 1.62 * team_batting['triples']
        + 2.10 * team_batting['hr']
    ) / (team_batting['ab'] + team_batting['bb'] + hbp)
    team_batting['woba'] = team_batting['woba'].replace([np.inf, -np.inf], np.nan)
    
    # PA (needed as denominator for rate stats)
    team_batting['pa'] = team_batting['ab'] + team_batting['bb']
    

    # Plate discipline rates
    team_batting['bb_pct']  = team_batting['bb'] / team_batting['pa']
    team_batting['k_pct']   = team_batting['k']  / team_batting['pa']
    team_batting['bb_k']    = team_batting['bb'] / team_batting['k'].replace(0, np.nan)

    # Power
    team_batting['iso']   = team_batting['slg'] - team_batting['avg']

    # Luck signals
    team_batting['babip']   = (
        (team_batting['h'] - team_batting['hr'])
        / (team_batting['ab'] - team_batting['k'] - team_batting['hr'])
    )
    team_batting['lob_pct'] = (
        (team_batting['h'] + team_batting['bb'] - team_batting['r'])
        / (team_batting['h'] + team_batting['bb'] - 1.4 * team_batting['hr'])
    )

    # Clip / clean
    for col in ['bb_pct','k_pct','bb_k','iso','babip','lob_pct']:
        team_batting[col] = team_batting[col].replace([np.inf, -np.inf], np.nan)
        
    return team_batting

def add_opponent_features(team_batting: pd.DataFrame) -> pd.DataFrame:
    """Self-join on gamepk to add opponent stats; compute Pythagorean Expectation and opp rolling features."""
    # ---- Opponent stats join ----
    # All opponent columns are prefixed opp_
    opp_col_renames = {
        'r':   'opp_r',
        'h':   'opp_h',
        'hr':  'opp_hr',
        'bb':  'opp_bb',
        'k':   'opp_k',
        'obp': 'opp_obp',
        'slg': 'opp_slg',
        'ops': 'opp_ops',
    }
    opp = (
        team_batting[['gamepk', 'team_id'] + list(opp_col_renames.keys())]
        .rename(columns={**opp_col_renames, 'team_id': 'opp_team_id'})
    )

    print(f"\nShape before opponent join: {team_batting.shape}")
    team_batting = (
        team_batting
        .merge(opp, on='gamepk', how='left')
        .query('opp_team_id != team_id')
        .copy()
    )
    print(f"Shape after opponent join:  {team_batting.shape}")

    # Sanity: sample rows to visually confirm the join
    print("\nSample (team_id | gamepk | r | opp_r):")
    print(team_batting[['team_id', 'gamepk', 'r', 'opp_r']].sample(3, random_state=42).to_string(index=False))

    # Sanity: confirm no gamepk has more than 2 teams (would indicate a bad join)
    teams_per_game = team_batting.groupby('gamepk')['team_id'].nunique()
    bad_games = teams_per_game[teams_per_game != 2]
    if bad_games.empty:
        print(f"\nOK: all {teams_per_game.shape[0]} gamepks have exactly 2 teams")
    else:
        print(f"\nWARNING: {len(bad_games)} gamepks do not have exactly 2 teams:\n{bad_games}")

    # ---- Pythagorean Expectation ----
    rs_pow = team_batting['r'] ** 1.83
    ra_pow = team_batting['opp_r'] ** 1.83
    team_batting['pyth_exp'] = rs_pow / (rs_pow + ra_pow)
    team_batting['pyth_exp'] = team_batting['pyth_exp'].replace([np.inf, -np.inf], np.nan)

    # ---- Opponent rolling features ----
    # opp_team_id and opp_* cols are already in team_batting from the self-join.
    # Sort per opp_team_id's own game timeline, roll, then join back on gamepk + opp_team_id.
    opp_src_cols = ['opp_r', 'opp_h', 'opp_hr', 'opp_bb', 'opp_k']

    opp_hist = (
        team_batting[['gamepk', 'opp_team_id', 'game_season', 'game_date'] + opp_src_cols]
        .sort_values(['opp_team_id', 'game_season', 'game_date'])
        .reset_index(drop=True)
    )

    for src_col in opp_src_cols:
        for w in [7, 14, 30]:
            feat_name = f'rolling_{w}_{src_col}'
            opp_hist[feat_name] = (
                opp_hist
                .groupby(['opp_team_id', 'game_season'])[src_col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    opp_roll_feat_cols = [f'rolling_{w}_{col}' for col in opp_src_cols for w in [7, 14, 30]]
    opp_roll = opp_hist[['gamepk', 'opp_team_id'] + opp_roll_feat_cols]
    team_batting = team_batting.merge(opp_roll, on=['gamepk', 'opp_team_id'], how='left')

    print("\nNull counts for opponent rolling features:")
    print(team_batting[opp_roll_feat_cols].isnull().sum())

    print("\nSample (team_id | opp_team_id | r | rolling_30_opp_r):")
    print(team_batting[['team_id', 'opp_team_id', 'r', 'rolling_30_opp_r']].sample(5, random_state=42).to_string(index=False))

    return team_batting

def add_team_rolling_features(team_batting: pd.DataFrame) -> pd.DataFrame:
    """Add per-team rolling features for batting stats, wOBA, and Pythagorean Expectation."""
    # Drop 2020 (COVID season — abnormal run environment)
    team_batting = team_batting[team_batting['game_season'] != 2020].copy()

    # Sort by team, season, game_date for correct rolling window order
    team_batting = team_batting.sort_values(['team_id', 'game_season', 'game_date']).reset_index(drop=True)

    roll_cols = [
        'h', 'hr', 'bb', 'k',
        'avg', 'obp', 'slg', 'ops',
        'bb_pct', 'k_pct', 'bb_k',   # discipline
        'iso',                         # power
        'babip', 'lob_pct',            # luck signals
    ]
    
    windows   = [7, 14, 30]

    for col in roll_cols:
        for w in windows:
            feat_name = f'rolling_{w}_{col}'
            team_batting[feat_name] = (
                team_batting
                .groupby(['team_id', 'game_season'])[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    # Naive baseline feature: 30-game rolling mean of r per team-season (shift(1) applied)
    team_batting['rolling_30_r'] = (
        team_batting
        .groupby(['team_id', 'game_season'])['r']
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )

    # wOBA and Pythagorean Expectation rolling features (14 and 30 game windows, shift(1) before rolling)
    team_batting['woba_14g'] = (
        team_batting
        .groupby(['team_id', 'game_season'])['woba']
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    )
    team_batting['woba_30g'] = (
        team_batting
        .groupby(['team_id', 'game_season'])['woba']
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )
    team_batting['pyth_exp_14g'] = (
        team_batting
        .groupby(['team_id', 'game_season'])['pyth_exp']
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    )
    team_batting['pyth_exp_30g'] = (
        team_batting
        .groupby(['team_id', 'game_season'])['pyth_exp']
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )

    new_feat_cols = ['woba_14g', 'woba_30g', 'pyth_exp_14g', 'pyth_exp_30g']
    print("\nNull counts for new engineered features:")
    print(team_batting[new_feat_cols].isnull().sum())

    return team_batting

def add_sp_features(team_batting: pd.DataFrame, pitcher_raw: pd.DataFrame) -> pd.DataFrame:
    """Identify starters, compute per-game rate stats, roll them, and join to team_batting via opp_team_id."""
    pitcher = pitcher_raw.copy()

    def mlb_ip_to_true(ip_series) -> pd.Series:
        """Convert MLB innings-pitched notation (6.1 = 6⅓) to true decimal IP."""
        ip_f  = ip_series.astype(float)
        whole = ip_f.apply(lambda x: int(x) if pd.notna(x) else np.nan)
        outs  = ip_f.apply(lambda x: round((x - int(x)) * 10) if pd.notna(x) else np.nan)
        return whole + outs / 3.0

    if 'team_id' not in pitcher.columns:
        raise ValueError(
            "pitcher_raw is missing 'team_id' column. "
            "Add team_id to the pitcher boxscore processing step before calling add_sp_features()."
        )

    ip_col = next((c for c in ['ip', 'inningsPitched'] if c in pitcher.columns), None)
    k_col  = next((c for c in ['so', 'k', 'strikeOuts'] if c in pitcher.columns), None)
    bb_col = next((c for c in ['bb', 'baseOnBalls'] if c in pitcher.columns), None)
    h_col  = next((c for c in ['h', 'hits'] if c in pitcher.columns), None)
    er_col = next((c for c in ['er', 'earnedRuns'] if c in pitcher.columns), None)
    hr_col = next((c for c in ['hr', 'homeRuns'] if c in pitcher.columns), None)  

    if ip_col is None:
        raise ValueError("No IP column found in pitcher_boxscore — check schema printed above.")

    pitcher['ip_true'] = mlb_ip_to_true(pitcher[ip_col])

    # Identify starters: prefer gameSequence==0 or position=='SP'; fall back to max IP per team per game
    if 'gameSequence' in pitcher.columns:
        starters = pitcher[pitcher['gameSequence'] == 0].copy()
        print("\nStarter identification: gameSequence == 0")
    elif 'position' in pitcher.columns and pitcher['position'].astype(str).str.upper().isin(['SP']).any():
        starters = pitcher[pitcher['position'].astype(str).str.upper() == 'SP'].copy()
        print("\nStarter identification: position == 'SP'")
    else:
        starters = (
            pitcher
            .sort_values('ip_true', ascending=False)
            .drop_duplicates(subset=['gamepk', 'team_id'])  
            .copy()
        )
        print("\nStarter identification: pitcher with max IP per (gamePk, team_id)")

    print(f"Total starter rows: {len(starters):,}")
    sample_disp = ['gamePk', 'personId', 'team_id', 'ip_true']
    avail_disp  = [c for c in sample_disp + ([k_col] if k_col else []) + ([bb_col] if bb_col else []) if c in starters.columns]
    print("\nSample starters (3 rows):")
    print(starters[avail_disp].head(3).to_string(index=False))

    # Per-game rate stats (guard against ip_true == 0)
    safe_ip = starters['ip_true'].replace(0, np.nan)

    starters['era_gm']  = (
        (starters[er_col].astype(float) / safe_ip * 9).clip(upper=27)
        if er_col else np.nan
    )
    starters['whip_gm'] = (
        ((starters[h_col].astype(float) + starters[bb_col].astype(float)) / safe_ip).clip(upper=9)
        if (h_col and bb_col) else np.nan
    )
    starters['k9_gm']   = (
        (starters[k_col].astype(float) / safe_ip * 9).clip(upper=27)
        if k_col else np.nan
    )
    starters['bb9_gm']  = (
        (starters[bb_col].astype(float) / safe_ip * 9).clip(upper=27)
        if bb_col else np.nan
    )
    starters['hr9_gm']  = (
        (starters[hr_col].astype(float) / safe_ip * 9).clip(upper=9)
        if hr_col else np.nan
    )

    if hr_col is None:
        print("\nWARNING: no HR column found in pitcher data — sp_hr9 features will be NaN")
    else:
        print(f"\nHR column found: '{hr_col}' — sp_hr9 features will be computed")

    # Rolling features: shift(1) then roll, per pitcher per season
    starters = starters.sort_values(['personId', 'game_season', 'game_date']).reset_index(drop=True)

    _sp_rate_map = {
        'era_gm':  'sp_era',
        'whip_gm': 'sp_whip',
        'k9_gm':   'sp_k9',
        'bb9_gm':  'sp_bb9',
        'ip_true': 'sp_ip',
        'hr9_gm':  'sp_hr9',   # NEW
    }
    _sp_windows = [5, 10]

    for src, prefix in _sp_rate_map.items():
        for w in _sp_windows:
            feat = f'{prefix}_{w}s'
            starters[feat] = (
                starters
                .groupby(['personId', 'game_season'])[src]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    sp_feature_cols = [f'{prefix}_{w}s' for prefix in _sp_rate_map.values() for w in _sp_windows]

    # Join to team_batting: sp_join.team_id is the team whose SP pitched
    # Each team_batting row has opp_team_id → we want that team's starting pitcher's rolling stats
    sp_join = (
        starters[['gamepk', 'team_id'] + sp_feature_cols]  # Fix 1: was gamePk; must match team_batting's lowercase gamepk
        .rename(columns={'team_id': 'opp_team_id'})
    )
    team_batting = team_batting.merge(sp_join, on=['gamepk', 'opp_team_id'], how='left')  # Fix 1: was gamePk; silent null bug — lowercase gamepk required

    print("\nNull counts for SP rolling features after join:")
    print(team_batting[sp_feature_cols].isnull().sum().to_string())

    return team_batting

# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def build_train_val_split(team_batting: pd.DataFrame, feature_cols: list) -> tuple:
    """Split into train/val by season; return (X_train, y_train, X_val, y_val, val_df)."""
    train = team_batting[team_batting['game_season'].isin(TRAIN_SEASONS)].dropna(subset=feature_cols + ['r'])
    val   = team_batting[team_batting['game_season'].isin(VAL_SEASONS)].dropna(subset=feature_cols + ['r'])

    X_train, y_train = train[feature_cols], train['r']
    X_val,   y_val   = val[feature_cols],   val['r']

    return X_train, y_train, X_val, y_val, val


def train_model(X_train, y_train, X_val, y_val):
    """Fit an XGBoost Poisson regressor and return the trained model."""
    model = xgb.XGBRegressor(**_XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(model, X_val, y_val, val_df: pd.DataFrame, label: str) -> dict:
    """Print MAE, naive rolling baseline, and lift; return metrics dict."""
    preds       = model.predict(X_val)
    mae         = mean_absolute_error(y_val, preds)
    val_baseline = val_df['rolling_30_r'].fillna(y_val.mean())
    mae_baseline = mean_absolute_error(y_val, val_baseline)
    lift         = mae_baseline - mae

    print(f"MAE {label}: {mae:.4f}")
    print(f"Naive baseline MAE (rolling_30_r): {mae_baseline:.4f}")
    print(f"Improvement over baseline ({label}): {lift:.4f} runs")

    return {'mae': mae, 'mae_baseline': mae_baseline, 'lift': lift, 'preds': preds}



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# 1. Load data
batter_boxscore  = load_batter_data(TRAIN_SEASONS + VAL_SEASONS)
pitcher_boxscore = load_pitcher_data(TRAIN_SEASONS + VAL_SEASONS)



# 2. Build team-level features
team_batting = build_team_batting(batter_boxscore)
team_batting = add_opponent_features(team_batting)

team_batting = add_team_rolling_features(team_batting)

team_batting = add_sp_features(team_batting, pitcher_boxscore)

batter_boxscore.columns.tolist()


# 3. Define feature column sets
_roll_cols       = ['h', 'hr', 'bb', 'k', 'avg', 'obp', 'slg', 'ops']
_windows         = [7, 14, 30]
_new_feat_cols   = ['woba_14g', 'woba_30g', 'pyth_exp_14g', 'pyth_exp_30g']
_opp_src_cols    = ['opp_r', 'opp_h', 'opp_hr', 'opp_bb', 'opp_k']
_opp_roll_feats  = [f'rolling_{w}_{col}' for col in _opp_src_cols for w in [7, 14, 30]]
_sp_prefixes     = ['sp_era', 'sp_whip', 'sp_k9', 'sp_bb9', 'sp_ip']
_sp_feature_cols = [f'{prefix}_{w}s' for prefix in _sp_prefixes for w in [5, 10]]

feature_cols = (
    [f'rolling_{w}_{col}' for col in _roll_cols for w in _windows]
    + _new_feat_cols
    + _opp_roll_feats
)

# 4. Without SP features
X_train, y_train, X_val, y_val, val = build_train_val_split(team_batting, feature_cols)
print(f"Train rows (without SP): {len(X_train)}  |  Val rows: {len(X_val)}")

model_base    = train_model(X_train, y_train, X_val, y_val)
results_base  = evaluate(model_base, X_val, y_val, val, label="without SP features (val 2024)")

# 5. With SP features
feature_cols_with_sp = feature_cols + _sp_feature_cols
X_train_sp, y_train_sp, X_val_sp, y_val_sp, val_sp = build_train_val_split(team_batting, feature_cols_with_sp)

model_sp   = train_model(X_train_sp, y_train_sp, X_val_sp, y_val_sp)
results_sp = evaluate(model_sp, X_val_sp, y_val_sp, val_sp, label="with SP features (val 2024)")
print(f"MAE before → after: {results_base['mae']:.4f} → {results_sp['mae']:.4f}  (delta: {results_sp['mae'] - results_base['mae']:+.4f} runs)")

# 6. Permutation Importance (SP model)
perm_result = permutation_importance(
    model_sp, X_val_sp, y_val_sp,
    n_repeats=20,
    scoring='neg_mean_absolute_error',
    random_state=42,
)

importance_df = pd.DataFrame({
    'feature':        X_val_sp.columns,
    'importance_mean': perm_result.importances_mean,
    'importance_std':  perm_result.importances_std,
}).sort_values('importance_mean', ascending=False)


# RESIDUAL IMPORTANCE ---------------------------

X_val_sp['pred'] = model_sp.predict(X_val_sp)
X_val_sp['residual'] = y_val_sp - X_val_sp['pred']

X_val_sp
px.line(X_val_sp, x='')


px.histogram(X_val_sp['residual'])


pitcher_boxscore.columns.tolist()
playbyplay.columns.tolist()

