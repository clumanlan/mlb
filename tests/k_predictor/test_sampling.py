import numpy as np
import pandas as pd
import pytest

from models.k_predictor.utils.sampling import stratified_subsample


def _make_df(n=5000, positive_rate=0.22, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "is_strikeout": rng.binomial(1, positive_rate, size=n),
        "feature_a": rng.normal(size=n),
    })


def test_sample_size_does_not_exceed_n():
    df = _make_df(n=5000)
    result = stratified_subsample(df, target_col="is_strikeout", n=1000, random_state=42)
    assert len(result) <= 1000


def test_returns_full_df_unchanged_when_len_le_n():
    df = _make_df(n=500)
    result = stratified_subsample(df, target_col="is_strikeout", n=1000, random_state=42)
    assert len(result) == 500
    assert sorted(result.index) == sorted(df.index)


def test_preserves_class_balance_within_tolerance():
    df = _make_df(n=20000, positive_rate=0.22)
    original_rate = df["is_strikeout"].mean()
    result = stratified_subsample(df, target_col="is_strikeout", n=4000, random_state=42)
    sampled_rate = result["is_strikeout"].mean()
    assert abs(sampled_rate - original_rate) < 0.02


def test_reproducible_with_fixed_random_state():
    df = _make_df(n=20000)
    result_a = stratified_subsample(df, target_col="is_strikeout", n=3000, random_state=42)
    result_b = stratified_subsample(df, target_col="is_strikeout", n=3000, random_state=42)
    assert sorted(result_a.index) == sorted(result_b.index)
