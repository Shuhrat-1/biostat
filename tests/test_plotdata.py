"""Тесты подготовки данных для графиков."""

from __future__ import annotations

import numpy as np
import pytest

from core.stats.plotdata import (
    bar_means_data,
    boxplot_data,
    histogram_data,
    scatter_regression_data,
)


def test_histogram_basic():
    data = list(range(100))
    h = histogram_data(data, bins=10)
    assert len(h["counts"]) == 10
    assert len(h["bin_edges"]) == 11
    assert sum(h["counts"]) == 100


def test_histogram_normal_curve_present():
    rng = np.random.default_rng(0)
    h = histogram_data(rng.normal(0, 1, 200).tolist())
    assert h["normal_curve"] is not None
    assert len(h["normal_curve"]["x"]) == len(h["normal_curve"]["y"])


def test_histogram_empty():
    h = histogram_data([])
    assert h["counts"] == []
    assert h["normal_curve"] is None


def test_histogram_drops_nan():
    h = histogram_data([1.0, 2.0, float("nan"), 4.0], bins=2)
    assert sum(h["counts"]) == 3


def test_boxplot_structure():
    groups = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
    b = boxplot_data(groups)
    assert b["labels"] == ["a", "b"]
    assert len(b["boxes"]) == 2
    # [min, Q1, median, Q3, max] — median в середине.
    assert b["boxes"][0][2] == pytest.approx(3.0)


def test_boxplot_detects_outlier():
    """Явный выброс попадает в список outliers."""
    groups = {"a": [1, 2, 3, 4, 5, 100]}
    b = boxplot_data(groups)
    assert len(b["outliers"]) == 1
    assert b["outliers"][0][1] == 100.0


def test_boxplot_whiskers_within_data():
    groups = {"a": [2, 4, 6, 8, 10]}
    b = boxplot_data(groups)
    box = b["boxes"][0]
    assert box[0] >= 2.0  # ус не ниже минимума
    assert box[4] <= 10.0  # ус не выше максимума


def test_scatter_regression_line():
    x = [0, 1, 2, 3, 4]
    y = [1, 3, 5, 7, 9]  # y = 2x + 1
    s = scatter_regression_data(x, y, slope=2.0, intercept=1.0)
    assert len(s["points"]) == 5
    # Линия: крайние точки по x=0 и x=4.
    assert s["line"][0] == pytest.approx([0.0, 1.0])
    assert s["line"][1] == pytest.approx([4.0, 9.0])


def test_scatter_drops_nan_pairs():
    x = [1, 2, float("nan"), 4]
    y = [2, 4, 6, float("nan")]
    s = scatter_regression_data(x, y, slope=2.0, intercept=0.0)
    assert len(s["points"]) == 2


def test_bar_means():
    groups = {"a": [4, 5, 6], "b": [10, 11, 12]}
    b = bar_means_data(groups)
    assert b["labels"] == ["a", "b"]
    assert b["means"][0] == pytest.approx(5.0)
    assert b["means"][1] == pytest.approx(11.0)
    assert all(e > 0 for e in b["errors"])  # SE положительна


def test_bar_means_single_value_zero_error():
    b = bar_means_data({"a": [5.0]})
    assert b["errors"][0] == 0.0


def test_serializable():
    import json

    groups = {"a": [1, 2, 3], "b": [4, 5, 6]}
    for data in [
        histogram_data([1, 2, 3, 4, 5]),
        boxplot_data(groups),
        scatter_regression_data([1, 2, 3], [2, 4, 6], 2.0, 0.0),
        bar_means_data(groups),
    ]:
        json.dumps(data)  # не должно бросить
