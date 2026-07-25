import numpy as np
import pandas as pd


def reliability_bins(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(df[pred_col].clip(0, 1), edges[1:-1])

    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        subset = df[mask]
        if len(subset) == 0:
            continue
        rows.append({
            "bin_mid": round((edges[i] + edges[i + 1]) / 2, 4),
            "pred_mean": subset[pred_col].mean(),
            "actual_rate": subset[target_col].mean(),
            "n": len(subset),
        })

    return pd.DataFrame(rows, columns=["bin_mid", "pred_mean", "actual_rate", "n"])
