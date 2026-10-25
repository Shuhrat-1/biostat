"""Тесты трансформаций отклика (гл. 6): Table 6.4, Example 6.1A–C."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from core.stats.transforms import (
    TRANSFORMS,
    back_transform,
    suggest_log_offset,
    transform,
    transformed_means,
)

# Welham et al. (2015), Table 6.1 (стр. 117), файл beetles.dat: число яиц,
# 1 = P. vit. × inter, 2 = P. vit. × intra, 3 = P. vulg. × inter, 4 = P. vulg. × intra.
BEETLES = {
    "1": [57, 15, 40, 34, 42, 19, 43, 39, 36, 24],
    "2": [90, 80, 101, 59, 73, 51, 43, 57, 42, 66],
    "3": [82, 91, 66, 98, 82, 134, 51, 96, 52, 91],
    "4": [136, 117, 181, 41, 89, 106, 133, 98, 106, 79],
}


def _beetles(kind: str = "log10"):
    return transformed_means(*BEETLES.values(), labels=list(BEETLES), transform_kind=kind)


def _codes(result) -> set[str]:
    return {w.code for w in result.warnings}


def test_book_example_6_1a_anova_on_log10():
    """Table 6.2 (стр. 118): SS 1.3751 / 0.8571, F = 19.254 на (3, 36) df."""
    r = _beetles()
    s = r.statistics
    assert s["ss_between"] == pytest.approx(1.3751, abs=5e-5)
    assert s["ss_within"] == pytest.approx(0.8571, abs=5e-5)
    assert s["ms_within"] == pytest.approx(0.0238, abs=5e-5)
    assert s["F"] == pytest.approx(19.254, abs=5e-4)
    assert (s["df_between"], s["df_within"]) == (3, 36)
    assert r.p_value < 0.001


def test_book_table_6_3_means_sem_sed_on_log_scale():
    """Table 6.3 (стр. 118): средние 1.513, 1.804, 1.909, 2.008; SEM 0.0488; SED 0.0690."""
    tr = _beetles().parameters["transformed"]
    assert list(tr["means"].values()) == pytest.approx([1.513, 1.804, 1.909, 2.008], abs=5e-4)
    assert tr["sem"]["mean_1"] == pytest.approx(0.0488, abs=5e-5)
    assert tr["statistics"]["sed"] == pytest.approx(0.0690, abs=5e-5)


def test_book_table_6_5_back_transformed_means_and_ci():
    """Table 6.5 (стр. 123): средние и 95 % ДИ на исходной шкале."""
    expected = {
        "1": (25.9, 32.6, 40.9),
        "2": (50.7, 63.6, 79.9),
        "3": (64.6, 81.1, 101.8),
        "4": (81.2, 102.0, 128.1),
    }
    r = _beetles()
    for lab, (low, mid, high) in expected.items():
        est = next(e for e in r.estimates if e.name == f"mean_{lab}")
        assert (est.ci_lower, est.value, est.ci_upper) == pytest.approx((low, mid, high), abs=0.05)
        assert est.std_error is None  # SE обратно не переводится (стр. 122)


def test_book_example_6_1c_ratio_and_lsd_factor():
    """Example 6.1C (стр. 125–126): 4 к 1 = 3.13, 95 % ДИ (2.27, 4.32), множитель LSD 1.38."""
    r = _beetles()
    ratio = next(c for c in r.parameters["ratios"] if c["group1"] == "1" and c["group2"] == "4")
    assert ratio["ratio"] == pytest.approx(3.13, abs=5e-3)
    assert (ratio["ci_lower"], ratio["ci_upper"]) == pytest.approx((2.27, 4.32), abs=5e-3)
    assert r.statistics["lsd_factor"] == pytest.approx(1.38, abs=5e-3)


def test_back_transformed_log_mean_is_geometric_mean():
    """Обратно переведённое среднее log = геометрическое ≤ арифметического (стр. 124)."""
    r = _beetles("ln")
    for lab, values in BEETLES.items():
        est = next(e for e in r.estimates if e.name == f"mean_{lab}")
        assert est.value == pytest.approx(stats.gmean(values))
        assert est.value < np.mean(values)


def test_log_base_does_not_change_test_or_back_means():
    """log10 и ln дают одинаковые F и средние на исходной шкале."""
    a, b = _beetles("log10"), _beetles("ln")
    assert a.statistics["F"] == pytest.approx(b.statistics["F"])
    assert [e.value for e in a.estimates] == pytest.approx([e.value for e in b.estimates])


@pytest.mark.parametrize("kind", TRANSFORMS)
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_round_trip_table_6_4(kind, offset):
    """back_transform(transform(y)) = y для всех пар Table 6.4."""
    y = np.array([3.0, 40.0, 75.0]) if kind == "logit_pct" else np.array([0.2, 0.5, 0.8])
    if kind in ("log10", "ln", "sqrt"):
        y = y * 10
    assert back_transform(transform(y, kind, offset), kind, offset) == pytest.approx(y)


def test_logit_pct_back_transform_matches_book_formula():
    """P = [(100 + c)e^z − c]/(1 + e^z) (Table 6.4)."""
    z, c = 1.3, 0.7
    expected = ((100 + c) * np.exp(z) - c) / (1 + np.exp(z))
    assert back_transform(z, "logit_pct", c) == pytest.approx(expected)
    assert back_transform(0.0, "logit") == pytest.approx(0.5)  # p = 0.5 ↔ logit 0 (стр. 120)


def test_suggest_log_offset_rules():
    """Сдвиг (стр. 116): 1 для целых счётов с нулями, ½ мин. положительного иначе."""
    assert suggest_log_offset(np.array([0, 3, 7, 12])) == 1.0
    assert suggest_log_offset(np.array([0.0, 0.1, 0.4])) == pytest.approx(0.05)
    assert suggest_log_offset(np.array([2.0, 5.0])) == 0.0


def test_zero_counts_without_offset_is_error_with_hint():
    r = transformed_means([0, 3, 5], [2, 4, 9], transform_kind="log10")
    warn = next(w for w in r.warnings if w.code == "transform_domain_log")
    assert warn.params["suggested_offset"] == 1.0
    assert r.estimates == []


def test_zero_counts_with_offset_works():
    r = transformed_means([0, 3, 5], [2, 4, 9], transform_kind="log10", offset=1.0)
    assert r.p_value is not None
    est = next(e for e in r.estimates if e.name == "mean_g1")
    assert est.value == pytest.approx(10 ** np.mean(np.log10([1, 4, 6])) - 1)


def test_logit_rejects_proportions_outside_unit_interval():
    r = transformed_means([0.1, 0.4, 1.0], [0.2, 0.3, 0.5], transform_kind="logit")
    assert "transform_domain" in _codes(r)


def test_sqrt_lower_limit_clamped_and_flagged():
    """Нижняя граница ниже 0 на шкале √ прижимается к 0 с предупреждением."""
    r = transformed_means([0.0, 0.0, 0.0, 9.0], [4.0, 9.0, 16.0, 25.0], transform_kind="sqrt")
    low = next(e for e in r.estimates if e.name == "mean_g1").ci_lower
    assert low == pytest.approx(0.0)
    assert "back_transform_out_of_range" in _codes(r)


def test_ratios_only_for_log():
    assert "ratios" not in _beetles("sqrt").parameters


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        transformed_means([1, 2, 3], [4, 5, 6], transform_kind="boxcox")


def test_result_json_safe():
    json.dumps(_beetles().to_dict(), allow_nan=False)
