"""Исправление: x и y регрессии выравниваются по строкам (патч после гл. 13).

Раньше мост брал x и y через numeric_column, который выбрасывает пропуски
в каждой колонке отдельно: при пропусках в разных строках пары сдвигались
(или длины не совпадали и метод падал).
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from core.stats.correlation import pearson_correlation
from core.stats.regression import simple_linear_regression
from core.stats.regression_checks import regression_diagnostics
from core.validation.parser import parse_file

CSV = b"x,y\n1,2.1\n2,\n3,6.2\n,7.9\n5,10.1\n6,11.8\n7,14.2\n8,15.9\n"
X_OK = np.array([1, 3, 5, 6, 7, 8], float)
Y_OK = np.array([2.1, 6.2, 10.1, 11.8, 14.2, 15.9])


def test_numeric_column_misaligns_pairs():
    table = parse_file(CSV)
    x, y = table.numeric_column("x"), table.numeric_column("y")
    assert len(x) == len(y) == 7  # равные длины, но пары сдвинуты
    assert (x[1], y[1]) == (2.0, 6.2)  # x строки 2 с y строки 3


def test_aligned_numeric_keeps_rows():
    x, y = parse_file(CSV).aligned_numeric(["x", "y"])
    assert len(x) == len(y) == 8
    assert np.isnan(y[1]) and np.isnan(x[3])


def test_regression_on_aligned_pairs_matches_complete_rows():
    x, y = parse_file(CSV).aligned_numeric(["x", "y"])
    res = simple_linear_regression(x, y)
    ref = sm.OLS(Y_OK, sm.add_constant(X_OK)).fit()
    slope = next(e for e in res.estimates if e.name == "slope")
    assert slope.value == pytest.approx(ref.params[1])
    assert res.n == 6
    r = pearson_correlation(x, y)
    assert r.statistics["r"] == pytest.approx(np.corrcoef(X_OK, Y_OK)[0, 1])


def test_diagnostics_report_file_rows():
    x, y = parse_file(CSV).aligned_numeric(["x", "y"])
    rows = [o["row"] for o in regression_diagnostics(x, y).parameters["observations"]]
    assert rows == [1, 3, 5, 6, 7, 8]
