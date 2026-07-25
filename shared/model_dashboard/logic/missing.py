import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    pct = df.isnull().mean()
    result = pct.reset_index()
    result.columns = ["feature", "pct_missing"]
    return result.sort_values("pct_missing", ascending=False).reset_index(drop=True)
