"""Тесты описательной статистики: сверка с эталоном, не с самим ядром."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.stats.descriptive import describe, mean_ci

# Датасет с известными вручную посчитанными свойствами.
SIMPLE = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
# mean = 40/8 = 5.0; var(ddof=1): сумма кв.откл. = 32, /7 => 32/7


def test_describe_mean_median():
    r = describe(SIMPLE)
    assert r.statistics["mean"] == pytest.approx(5.0)
    assert r.statistics["median"] == pytest.approx(4.5)


def test_describe_variance_ddof1():
    """Дисперсия с делением на (n-1), сверка с ручным расчётом."""
    r = describe(SIMPLE)
    assert r.statistics["variance"] == pytest.approx(32.0 / 7.0)
    assert r.statistics["std"] == pytest.approx(math.sqrt(32.0 / 7.0))


def test_describe_matches_numpy_reference():
    """Сверка со свежим независимым вызовом numpy."""
    rng = np.random.default_rng(42)
    data = rng.normal(10, 3, size=200)
    r = describe(data)
    assert r.statistics["mean"] == pytest.approx(np.mean(data))
    assert r.statistics["variance"] == pytest.approx(np.var(data, ddof=1))
    assert r.statistics["median"] == pytest.approx(np.median(data))


def test_describe_drops_nan_and_warns():
    r = describe([1.0, 2.0, np.nan, 4.0])
    assert r.n == 3
    assert any(w.code == "missing_dropped" for w in r.warnings)
    assert r.statistics["mean"] == pytest.approx(7.0 / 3.0)


def test_describe_empty():
    r = describe([])
    assert r.n == 0
    assert any(w.code == "empty" for w in r.warnings)


def test_describe_single_value_no_spread():
    r = describe([5.0])
    assert r.statistics["mean"] == pytest.approx(5.0)
    assert "variance" not in r.statistics
    assert any(w.code == "n_too_small" for w in r.warnings)


def test_mean_ci_known_values():
    """CI для среднего против ручного расчёта через t.

    data = [1,2,3,4,5]: mean=3, sd=sqrt(2.5), sem=sqrt(0.5),
    t(0.975, df=4)=2.7764. half = 2.7764*sqrt(0.5).
    """
    r = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0], level=0.95)
    est = r.estimates[0]
    half = 2.7764451051977987 * math.sqrt(0.5)
    assert est.value == pytest.approx(3.0)
    assert est.ci_lower == pytest.approx(3.0 - half)
    assert est.ci_upper == pytest.approx(3.0 + half)
    assert est.ci_level == 0.95


def test_mean_ci_wider_for_higher_level():
    """99% интервал шире 95% — базовое свойство."""
    d = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0]
    r95 = mean_ci(d, 0.95).estimates[0]
    r99 = mean_ci(d, 0.99).estimates[0]
    w95 = r95.ci_upper - r95.ci_lower
    w99 = r99.ci_upper - r99.ci_lower
    assert w99 > w95


def test_mean_ci_invalid_level():
    with pytest.raises(ValueError):
        mean_ci([1.0, 2.0, 3.0], level=1.5)


def test_mean_ci_too_small():
    r = mean_ci([5.0])
    assert not r.estimates
    assert any(w.code == "n_too_small" for w in r.warnings)


def test_result_json_roundtrip():
    """Результат сериализуется в JSON без ошибок."""
    import json

    r = describe(SIMPLE)
    parsed = json.loads(r.to_json())
    assert parsed["method"] == "describe"
    assert parsed["statistics"]["mean"] == 5.0
