import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_pa_data():
    """8-row deterministic PA dataset for fast unit tests."""
    return pd.DataFrame({
        "is_hit":       [0, 1, 0, 1, 0, 1, 0, 1],
        "pred_prob":    [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        "batSide":      ["L", "L", "R", "R", "L", "L", "R", "R"],
        "pitcher_hand": ["L", "R", "L", "R", "L", "R", "L", "R"],
    })


@pytest.fixture
def biased_slice_data():
    """Data where batSide=L slice has clearly worse log loss than overall.
    Used to verify slice ranking puts the bad slice on top."""
    rng = np.random.default_rng(42)
    n = 1000
    bat_side = rng.choice(["L", "R"], size=n)
    is_hit = rng.binomial(1, 0.3, size=n)
    pred_prob = np.where(
        bat_side == "L",
        rng.uniform(0.7, 0.9, size=n),
        rng.uniform(0.2, 0.4, size=n),
    )
    return pd.DataFrame({
        "is_hit": is_hit, "pred_prob": pred_prob, "batSide": bat_side,
    })
