import pandas as pd

from models.hit_predictor.utils.model_prep import (
    add_missing_indicators,
    get_columns_with_nulls,
    impute_and_encode,
    plot_feature_importance,
)


def test_get_columns_with_nulls_returns_only_cols_with_at_least_one_null():
    """Indicator columns are derived from which train columns actually have
    nulls — a column with no nulls in train gets no indicator, even if it's
    fully populated by coincidence in this particular frame."""

    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3], "c": [None, None, None]})

    assert get_columns_with_nulls(df, ["a", "b", "c"]) == ["a", "c"]


def test_add_missing_indicators_adds_bool_isnull_columns():
    """A rookie player with no last-season stats is systematically different
    from a veteran — imputing the median throws that away unless the fact
    of missingness is preserved as its own column. Must return a copy, not
    mutate the caller's frame in place."""

    df = pd.DataFrame({"a": [1, None, 3]})

    result = add_missing_indicators(df, ["a"])

    assert list(result["a_isnull"]) == [False, True, False]
    assert "a_isnull" not in df.columns


def test_impute_and_encode_uses_train_median_not_val_median():
    """Imputation must be fit on train only. val here has no observed value
    of its own to derive a median from, so if the function were (buggily)
    computing a per-split median instead of applying train's, this would
    surface as NaN/an error rather than a silently-plausible wrong number —
    still a real check on the fit-train/transform-both wiring."""

    X_train = pd.DataFrame({"n": [1.0, 3.0, None]})  # train median = 2.0
    X_val = pd.DataFrame({"n": [None]})

    _, X_val_out = impute_and_encode(X_train, X_val, num_cols=["n"], cat_cols=[])

    assert X_val_out["n"].iloc[0] == 2.0


def test_impute_and_encode_keeps_columns_that_are_entirely_null_in_train():
    """SimpleImputer silently DROPS any column that's entirely null at fit
    time (sklearn's default keep_empty_features=False) — e.g. a rare
    rolling/hand-split stat that happens to be all-null for a given
    FIT_SEASONS window. Without keep_empty_features=True, the output has
    fewer columns than num_cols, which breaks the DataFrame(columns=...)
    reconstruction with a shape mismatch rather than a clear error."""

    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [None, None, None]})
    X_val = pd.DataFrame({"a": [4.0], "b": [None]})

    X_train_out, X_val_out = impute_and_encode(
        X_train, X_val, num_cols=["a", "b"], cat_cols=[]
    )

    assert list(X_train_out.columns) == ["a", "b"]
    assert list(X_val_out.columns) == ["a", "b"]


def test_impute_and_encode_handles_missing_sentinel_and_unseen_val_category():
    """Missing categoricals get an explicit 'missing' sentinel category
    rather than most-frequent imputation — most-frequent would silently
    fold 'unknown hand' into whichever hand is most common, throwing away
    the same missingness signal the numeric indicator columns exist to
    preserve. A category never seen in train (val's 'S') must not raise —
    OrdinalEncoder needs handle_unknown configured for this."""

    X_train = pd.DataFrame({"hand": ["L", "R", None]})
    X_val = pd.DataFrame({"hand": ["S"]})

    X_train_out, X_val_out = impute_and_encode(
        X_train, X_val, num_cols=[], cat_cols=["hand"]
    )

    assert X_train_out["hand"].nunique() == 3  # L, R, "missing" each get a distinct code
    assert X_val_out["hand"].iloc[0] == -1  # unseen category -> configured unknown code


def test_plot_feature_importance_saves_file(tmp_path):
    """v1 left feature importance as a TODO comment and never built it —
    this is the one concrete check that it actually produces a file, not
    that the plot looks any particular way."""

    importances = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    save_path = tmp_path / "fi.png"

    plot_feature_importance(importances, save_path=save_path, top_n=2)

    assert save_path.exists()
