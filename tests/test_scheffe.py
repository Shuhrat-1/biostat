"""scheffe_test — метод Шеффе [Witte, стр. 314; предл.: Scheffé 1953].

Witte называет Шеффе консервативнее Тьюки для пар и пригодным для сложных
контрастов («0 часов против объединённых 24 и 48»), но чисел не даёт; сверка —
с данными гл. 16 через тождества метода. Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest
from scipy import stats

from core.stats.anova import one_way_anova, tukey_hsd
from core.stats.oneway import scheffe_test
from tests import witte_data as W


def _rows(res) -> dict[tuple[str, str], dict]:
    return {(c["group1"], c["group2"]): c for c in res.parameters["comparisons"]}


def test_sleep_critical_s_and_pairs():
    """Witte, стр. 302, 306 (Tables 16.3, 16.5): MS_within = 3.67, F_crit(2, 6) = 5.14 →
    S = √(2 · 5.143) = 3.207; пара 0–48: SE = √(3.67 · 2/3) = 1.563, ДИ 6 ± 3.207 · 1.563."""
    res = scheffe_test(*W.SLEEP_B, labels=W.SLEEP_LABELS)
    assert res.statistics["critical_S"] == pytest.approx((2 * stats.f.ppf(0.95, 2, 6)) ** 0.5)
    assert round(res.statistics["critical_S"], 3) == 3.207
    row = _rows(res)[("0", "48")]
    assert row["mean_diff"] == 6
    assert row["ci_lower"] == pytest.approx(6 - 3.207258 * (22 / 6 * 2 / 3) ** 0.5, abs=1e-5)
    assert round(row["cohens_d"], 2) == 3.13  # как ф. 16.10


def test_extreme_contrast_reproduces_overall_f_table_16_9():
    """Свойство Шеффе: максимум F_S по всем контрастам = общий F. Для исхода B контраст
    0–48 пропорционален отклонениям средних (−3, 0, 3), поэтому его P = P ANOVA = .024
    (Minitab, Table 16.9, стр. 317)."""
    res = scheffe_test(*W.SLEEP_B, labels=W.SLEEP_LABELS)
    anova = one_way_anova(*W.SLEEP_B)
    row = _rows(res)[("0", "48")]
    assert row["F_scheffe"] == pytest.approx(anova.statistics["F"])
    assert row["p_adj"] == pytest.approx(anova.p_value)


def test_complex_contrast_witte_example():
    """Witte, стр. 314: «среднее 0 ч против объединённого среднего 24 и 48 ч».
    ψ = 2 − (5 + 8)/2 = −4.5; SE = √(3.67 · (1 + 1/4 + 1/4)/3) = 1.354; F_S = (4.5/1.354)²/2 = 5.52."""
    res = scheffe_test(*W.SLEEP_B, labels=W.SLEEP_LABELS, contrasts="0 vs 24 + 48")
    row = res.parameters["contrasts"][0]
    assert row["coefficients"] == {"0": 1.0, "24": -0.5, "48": -0.5}
    assert row["estimate"] == pytest.approx(-4.5)
    assert row["se"] == pytest.approx((22 / 6 * 1.5 / 3) ** 0.5)
    assert row["F"] == pytest.approx(4.5**2 / (22 / 6 * 0.5) / 2)
    assert row["p_adjusted"] == pytest.approx(stats.f.sf(row["F"], 2, 6))
    assert row["significant"]
    est = res.estimates[0]
    assert (est.value, est.ci_lower, est.ci_upper) == (row["estimate"], row["ci_lower"], row["ci_upper"])


def test_more_conservative_than_tukey_for_pairs():
    """Witte, стр. 314: Шеффе «лучше защищает от ложных тревог» ценой мощности —
    для пар его P не меньше, чем у Тьюки (зрительные контакты, Progress Check 16.7)."""
    labels = ["0", "1", "2", "3"]
    sch = _rows(scheffe_test(*W.EYE_CONTACTS, labels=labels))
    tuk = _rows(tukey_hsd(*W.EYE_CONTACTS, labels=labels))
    for key, row in sch.items():
        assert row["p_adj"] >= tuk[key]["p_adj"] - 1e-12
        assert row["ci_upper"] - row["ci_lower"] > tuk[key]["ci_upper"] - tuk[key]["ci_lower"]


def test_two_groups_equals_student_t():
    """При k = 2 S = t_crit и P Шеффе = P t-теста Стьюдента (EPO, Table 14.1: p = .0569)."""
    res = scheffe_test(W.EPO_X1, W.EPO_X2)
    assert res.statistics["critical_S"] == pytest.approx(stats.t.ppf(0.975, 10))
    assert round(res.parameters["comparisons"][0]["p_adj"], 4) == 0.0569


def test_unequal_n():
    """Review 16.10 (стр. 319): неравные n — SE пары из своих n; d от поправки не зависит."""
    row = _rows(scheffe_test(*W.SLEEP_UNEQUAL, labels=W.SLEEP_LABELS))[("24", "48")]
    ms = one_way_anova(*W.SLEEP_UNEQUAL).statistics["ms_within"]
    assert row["se"] == pytest.approx((ms * (1 / 3 + 1 / 4)) ** 0.5)
    assert round(row["cohens_d"], 2) == 2.12


def test_guards():
    assert scheffe_test([1, 2, 3]).warnings[0].code == "too_few_groups"
    assert scheffe_test([1, 2], []).warnings[0].code == "empty_group"
    assert scheffe_test([1], [2]).warnings[0].code == "n_too_small"
    assert scheffe_test([1, 1], [2, 2]).warnings[0].code == "scheffe_zero_variance"
    bad = scheffe_test(*W.SLEEP_B, labels=W.SLEEP_LABELS, contrasts="0 vs 72")
    assert bad.warnings[0].code == "contrast_invalid"
    with pytest.raises(ValueError):
        scheffe_test([1, 2], [3, 4], alpha=5)
