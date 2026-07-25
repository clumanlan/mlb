import numpy as np


def per_row_loss(y: int, p: float, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def slice_log_loss(df, target_col: str, pred_col: str) -> float:
    return df.apply(lambda r: per_row_loss(r[target_col], r[pred_col]), axis=1).mean()
