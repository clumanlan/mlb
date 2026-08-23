"""
utils/matchup_slicing.py

Cross-tabs PA-grain predictions by two matchup-quality features (batter
strength, pitcher dominance) instead of by predicted-probability quantile —
per BENCHMARKS.md §2's noise-at-n=1 caveat, this only means anything once
many PAs matching an archetype (e.g. "weak batter vs dominant pitcher") are
pooled together, same aggregation logic as the PA-vs-game-grain check, just
sliced by an interpretable feature pair instead of by n_pa.
"""
import pandas as pd


def slice_by_matchup_extremity(
    df: pd.DataFrame,
    batter_col: str,
    pitcher_col: str,
    outcome_col: str,
    pred_col: str,
    n_bins: int = 3,
    min_n: int = 0,
) -> pd.DataFrame:
    """
    Bins batter_col and pitcher_col each into n_bins quantile groups and
    reports n / mean_pred / obs_rate / reliable for every (batter_bin,
    pitcher_bin) cell.

    batter_col is assumed higher-is-better (stronger batter) — bin labels
    range low->high as weak/.../strong. pitcher_col is assumed lower-is-
    better (e.g. hit rate allowed — more dominant pitcher) — bin labels
    range low->high as dominant/.../weak. Rows with a NaN in any of the four
    columns are dropped before binning.
    """
    df = df[[batter_col, pitcher_col, outcome_col, pred_col]].dropna().copy()

    def _labels(n, low_to_high):
        if n == 1:
            return [low_to_high[len(low_to_high) // 2]]
        if n == 2:
            return [low_to_high[0], low_to_high[-1]]
        step = (len(low_to_high) - 1) / (n - 1)
        return [low_to_high[round(i * step)] for i in range(n)]

    def _bin(col, low_to_high):
        codes, edges = pd.qcut(df[col], q=n_bins, duplicates="drop", retbins=True)
        actual_n = len(edges) - 1
        labels = _labels(actual_n, low_to_high)
        return codes.cat.rename_categories(labels)

    df["batter_bin"] = _bin(batter_col, ["weak", "below_avg", "avg", "above_avg", "strong"])
    df["pitcher_bin"] = _bin(pitcher_col, ["dominant", "above_avg", "avg", "below_avg", "weak"])

    grouped = (
        df.groupby(["batter_bin", "pitcher_bin"], observed=True)
        .agg(n=(outcome_col, "size"), obs_rate=(outcome_col, "mean"), mean_pred=(pred_col, "mean"))
        .reset_index()
    )
    grouped["reliable"] = grouped["n"] >= min_n

    return grouped
