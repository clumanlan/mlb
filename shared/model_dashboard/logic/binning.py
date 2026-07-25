import pandas as pd


def bin_feature_by_outcome(
    df: pd.DataFrame,
    feature_col: str,
    outcome_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    col = df[feature_col].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if is_numeric:
        bins, labels = pd.qcut(df[feature_col], q=n_bins, retbins=True, duplicates="drop")
        grouped = df.groupby(bins, observed=True)[outcome_col].agg(["mean", "count"]).reset_index()
        grouped.columns = ["bin_label", "mean_outcome", "n"]
        grouped["bin_label"] = grouped["bin_label"].astype(str)
    else:
        grouped = (
            df.groupby(df[feature_col].astype(str), observed=True)[outcome_col]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped.columns = ["bin_label", "mean_outcome", "n"]

    return grouped.reset_index(drop=True)
