"""
utils/model_prep.py

Model-agnostic prep utilities shared across experiments: missing-value
indicators, train-fit/both-transform imputation + encoding, and a
feature-importance plot for tree-based models. Kept separate from
processing/features/*.py (which computes the features themselves) since
this module operates on an already-assembled model_df / feature matrix.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def get_columns_with_nulls(df, cols):
    """Return the subset of `cols` that have at least one null in `df`.

    Meant to be called on the training split only — the resulting list is
    then applied identically to both train and val via
    add_missing_indicators, so indicator columns never diverge between
    splits based on val's own missingness pattern.
    """
    return [c for c in cols if df[c].isnull().any()]


def add_missing_indicators(df, indicator_cols):
    """Add a boolean `{col}_isnull` column for each col in indicator_cols.

    Returns a copy — the caller's frame is left untouched. Apply the same
    `indicator_cols` list (derived from train via get_columns_with_nulls) to
    both train and val so the two splits never end up with different
    indicator columns.
    """
    indicators = pd.DataFrame(
        {f"{col}_isnull": df[col].isnull() for col in indicator_cols}, index=df.index
    )
    return pd.concat([df, indicators], axis=1)


def impute_and_encode(X_train, X_val, num_cols, cat_cols):
    """Impute + encode; fit on train only, transform both.

    Numeric columns: median imputation (train median applied to val).
    Categorical columns: nulls filled with an explicit "missing" sentinel
    category (not most-frequent — that would silently fold missingness
    into the mode instead of preserving it as its own category), then
    OrdinalEncoder fit on train. A category never seen in train (e.g. a
    val-only value) is mapped to -1 rather than raising.

    Returns (X_train_out, X_val_out) as DataFrames with num_cols + cat_cols
    as columns, so feature names stay attached all the way to a fitted
    tree model's feature_importances_.
    """
    # keep_empty_features=True: without it, SimpleImputer silently drops any
    # column that's entirely null in X_train (e.g. a rare rolling/hand-split
    # stat with no observed values for a given FIT_SEASONS window), which
    # would desync the output from num_cols. An all-null column is filled
    # with 0 rather than a median (which is undefined for it).
    num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Xtr_num = pd.DataFrame(
        num_imputer.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index
    ) if num_cols else pd.DataFrame(index=X_train.index)
    Xval_num = pd.DataFrame(
        num_imputer.transform(X_val[num_cols]), columns=num_cols, index=X_val.index
    ) if num_cols else pd.DataFrame(index=X_val.index)

    if cat_cols:
        Xtr_cat_filled = X_train[cat_cols].fillna("missing").astype(object)
        Xval_cat_filled = X_val[cat_cols].fillna("missing").astype(object)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr_cat = pd.DataFrame(
            encoder.fit_transform(Xtr_cat_filled), columns=cat_cols, index=X_train.index
        )
        Xval_cat = pd.DataFrame(
            encoder.transform(Xval_cat_filled), columns=cat_cols, index=X_val.index
        )
    else:
        Xtr_cat = pd.DataFrame(index=X_train.index)
        Xval_cat = pd.DataFrame(index=X_val.index)

    return pd.concat([Xtr_num, Xtr_cat], axis=1), pd.concat([Xval_num, Xval_cat], axis=1)


def plot_feature_importance(importances, save_path, top_n=20):
    """Save a horizontal bar chart of the top-N feature importances.

    `importances` is a pd.Series indexed by feature name (e.g.
    pd.Series(model.feature_importances_, index=feature_names)).
    """
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.35)))
    ax.barh(top.index, top.values, color="steelblue")
    ax.set_title(f"Feature importance (top {top_n})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
