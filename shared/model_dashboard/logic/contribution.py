import pandas as pd

from shared.model_dashboard.logic.metrics import per_row_loss


def compute_contribution(
    df: pd.DataFrame,
    slices: list[dict],
    target_col: str,
    pred_col: str,
    min_n: int = 50,
) -> pd.DataFrame:
    overall_loss = df.apply(lambda r: per_row_loss(r[target_col], r[pred_col]), axis=1).mean()
    total_n = len(df)

    rows = []
    for s in slices:
        feature, value = s["feature"], s["value"]
        mask = df[feature] == value
        subset = df[mask]
        n = len(subset)
        if n < min_n:
            continue
        slice_loss = subset.apply(lambda r: per_row_loss(r[target_col], r[pred_col]), axis=1).mean()
        pct = n / total_n
        delta = slice_loss - overall_loss
        rows.append({
            "feature": feature,
            "value": value,
            "n": n,
            "pct": pct,
            "slice_loss": slice_loss,
            "delta": delta,
            "contribution": pct * delta,
        })

    result = pd.DataFrame(rows, columns=["feature", "value", "n", "pct", "slice_loss", "delta", "contribution"])
    return result.sort_values("contribution", ascending=False).reset_index(drop=True)
