import pandas as pd


def generate_single_slices(df: pd.DataFrame, slice_cols: list[str]) -> list[dict]:
    slices = []
    for col in slice_cols:
        if col not in df.columns:
            continue
        for val in sorted(df[col].dropna().unique(), key=str):
            slices.append({"feature": col, "value": val})
    return slices


def generate_interaction_slices(df: pd.DataFrame, interaction_pairs: list[tuple]) -> list[dict]:
    slices = []
    for col_a, col_b in interaction_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        combos = df[[col_a, col_b]].dropna().drop_duplicates()
        for _, row in combos.iterrows():
            slices.append({
                "feature": f"{col_a}×{col_b}",
                "value": f"{row[col_a]}×{row[col_b]}",
            })
    return slices
