import sys
import os
import math
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/features/transforms"))

from utils import mlb_ip_to_true


def test_mlb_ip_to_true_handles_nulls():
    result = mlb_ip_to_true(pd.Series([6.1, None]))
    assert math.isnan(result.iloc[1])


def test_mlb_ip_to_true_converts_fractional_notation():
    result = mlb_ip_to_true(pd.Series([6.1, 6.2, 7.0]))
    assert abs(result.iloc[0] - 6.333333) < 0.0001
    assert abs(result.iloc[1] - 6.666667) < 0.0001
    assert result.iloc[2] == 7.0
