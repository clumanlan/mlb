import pandas as pd


def stratified_subsample(df: pd.DataFrame, target_col: str, n: int, random_state: int = 42) -> pd.DataFrame:
    """Sample down to at most n rows, preserving target_col's class proportions.

    Used to bring a training set under a model's hard row-count ceiling
    (e.g. TabPFN) without shifting the class balance the model trains on.
    """
    if len(df) <= n:
        return df

    frac = n / len(df)
    parts = [group.sample(frac=frac, random_state=random_state) for _, group in df.groupby(target_col)]
    return pd.concat(parts)
