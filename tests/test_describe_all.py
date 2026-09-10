"""Тесты сводной описательной статистики describe_all."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.stats.descriptive import describe_all

# Колонка с известными свойствами (та же, что в test_descriptive).
SIMPLE = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


def test_summary_structure():
    r = describe_all({"a": SIMPLE, "b": [1.0, 2.0, 3.0, 4.0]})
    summary = r.parameters["summary"]
    assert set(summary) == {"a", "b"}
    assert r.parameters["columns"] == ["a", "b"]


def test_summary_values_match_manual():
    r = describe_all({"x": SIMPLE})
    m = r.parameters["summary"]["x"]
    assert m["mean"] == pytest.approx(5.0)
    assert m["median"] == pytest.approx(4.5)
    assert m["variance"] == pytest.approx(32.0 / 7.0)
    assert m["std"] == pytest.approx(math.sqrt(32.0 / 7.0))


def test_variance_std_consistent():
    """std² должно равняться variance — обе выборочные (ddof=1)."""
    r = describe_all({"x": SIMPLE})
    m = r.parameters["summary"]["x"]
    assert m["std"] ** 2 == pytest.approx(m["variance"])


def test_zero_metrics():
    r = describe_all({"z": [0.0, 0.0, 1.0, 2.0, 0.0]})
    m = r.parameters["summary"]["z"]
    assert m["zero_count"] == 3
    assert m["zero_pct"] == pytest.approx(60.0)


def test_outliers_iqr():
    """Явный выброс ловится правилом IQR."""
    r = describe_all({"x": [1, 2, 3, 4, 5, 6, 7, 8, 100]})
    m = r.parameters["summary"]["x"]
    assert m["outliers_iqr"] >= 1


def test_zscore_outliers():
    data = list(range(50)) + [1000]  # явный z-выброс
    r = describe_all({"x": data})
    m = r.parameters["summary"]["x"]
    assert m["outliers_zscore"] >= 1


def test_mad():
    """MAD = медиана абсолютных отклонений от медианы."""
    r = describe_all({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    m = r.parameters["summary"]["x"]
    # медиана 3, отклонения [2,1,0,1,2], медиана отклонений = 1
    assert m["mad"] == pytest.approx(1.0)


def test_cv_pct():
    r = describe_all({"x": [10.0, 12.0, 8.0, 10.0]})
    m = r.parameters["summary"]["x"]
    expected_cv = np.std([10, 12, 8, 10], ddof=1) / 10 * 100
    assert m["cv_pct"] == pytest.approx(expected_cv)


def test_missing_counted_per_column():
    r = describe_all({"x": [1.0, 2.0, np.nan, 4.0]})
    m = r.parameters["summary"]["x"]
    assert m["missing"] == 1
    assert m["n"] == 3


def test_empty_column_flagged():
    r = describe_all({"good": [1.0, 2.0, 3.0], "bad": [np.nan, np.nan]})
    assert any(w.code == "empty_columns" for w in r.warnings)
    assert r.parameters["summary"]["bad"]["n"] == 0


def test_no_columns():
    r = describe_all({})
    assert any(w.code == "empty" for w in r.warnings)


def test_relative_iqr():
    r = describe_all({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    m = r.parameters["summary"]["x"]
    # median 3, iqr = q3-q1 = 4-2 = 2, relative = 2/3*100
    assert m["relative_iqr_pct"] == pytest.approx(2.0 / 3.0 * 100)


def test_serializable():
    import json

    r = describe_all({"a": SIMPLE, "b": [1.0, 2.0, 3.0]})
    parsed = json.loads(r.to_json())
    assert parsed["method"] == "describe_all"
    assert "summary" in parsed["parameters"]
