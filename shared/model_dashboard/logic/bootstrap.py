import numpy as np
import pandas as pd


def bootstrap_ci(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    n_iter: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    y = df[target_col].to_numpy(dtype=np.float64)
    p = np.clip(df[pred_col].to_numpy(dtype=np.float64), 1e-15, 1 - 1e-15)
    row_losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    point = float(row_losses.mean())

    rng = np.random.default_rng(seed)
    n = len(row_losses)
    # Generate all sample indices at once: shape (n_iter, n)
    indices = rng.integers(0, n, size=(n_iter, n))
    # Mean log loss for each bootstrap sample: shape (n_iter,)
    boot_means = row_losses[indices].mean(axis=1)

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return point, lo, hi
