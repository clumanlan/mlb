import numpy as np
import pandas as pd

def mlb_ip_to_true(ip: pd.Series) -> pd.Series:
    """
    Convert MLB innings pitched notation to true decimal innings.
    MLB uses 6.1 to mean 6 and 1/3 innings, 6.2 to mean 6 and 2/3 innings.
    """
    whole     = ip.apply(lambda x: int(x) if pd.notna(x) else np.nan)
    fraction  = ip.apply(lambda x: round(x % 1, 1) if pd.notna(x) else np.nan)
    return whole + (fraction / 0.3).round(6)
 
 
def identify_starters_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the starting pitcher for each team in each game.
    Starter = first pitcher to appear for each team, determined by
    sorting on inning + event_index and taking the first row per
    (gamepk, pitcher_team_id).
 
    Returns DataFrame with columns:
        gamepk, team_id, starter_personId, starter_name
    """
    starters = (
        pbp
        .sort_values(["gamepk", "pitcher_team_id", "inning", "event_index"])
        .groupby(["gamepk", "pitcher_team_id"], as_index=False)
        .first()
        [["gamepk", "pitcher_team_id", "pitcher_id", "pitcher_name"]]
        .rename(columns={
            "pitcher_team_id": "team_id",
            "pitcher_id":      "starter_personId",
            "pitcher_name":    "starter_name",
        })
    )
    return starters
 
 