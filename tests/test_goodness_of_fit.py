"""chi2_goodness_of_fit — одномерный χ² согласия (Witte & Witte, 11th ed., гл. 19.1–19.5;
пакет W3). Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest
from scipy import stats

from core.stats.goodness_of_fit import chi2_goodness_of_fit, parse_proportions

_BLOOD = (["O", "A", "B", "AB"], [38, 38, 20, 4])


def _cells(res) -> dict[str, dict]:
    return {c["category"]: c for c in res.parameters["cells"]}


def test_blood_types_tables_19_1_19_2():
    """Witte 19.3–19.5, стр. 368–369 (Tables 19.1–19.2): f_e = 44, 41, 10, 5;
    χ² = .82 + .22 + 10.00 + .20 = 11.24, df 3, крит. 7.81 → H0 отвергнута. Точно χ² = 11.238."""
    res = chi2_goodness_of_fit(*_BLOOD, proportions="O=.44; A=.41; B=.10; AB=.05")
    cells = _cells(res)
    assert [cells[k]["expected"] for k in ("O", "A", "B", "AB")] == pytest.approx([44, 41, 10, 5])
    assert res.statistics["chi2"] == pytest.approx(11.2377, abs=1e-4)
    assert res.statistics["df"] == 3
    assert res.statistics["chi2"] > stats.chi2.ppf(0.95, 3)
    assert round(stats.chi2.ppf(0.95, 3), 2) == 7.81
    # Главный вклад — тип B (стр. 371): стандартизованный остаток (20 − 10)/√10.
    assert cells["B"]["std_residual"] == pytest.approx(10 / 10**0.5)
    assert not res.warnings  # все f_e ≥ 5


def test_equal_proportions_progress_check_19_2():
    """Witte, стр. 371 (Progress Check 19.2), ответ стр. 452: шесть желаний, n = 90, f_e = 15,
    χ² = 23.33 > 11.07 (df 5), p < .001."""
    res = chi2_goodness_of_fit(["love", "wealth", "power", "health", "fame", "family"],
                                [25, 10, 5, 25, 10, 15])
    assert res.statistics["chi2"] == pytest.approx(70 / 3)
    assert res.statistics["df"] == 5
    assert res.p_value < 0.001
    assert all(c["expected"] == 15 for c in res.parameters["cells"])


def test_holiday_review_19_10_with_phi_c_and_binomial():
    """Witte, стр. 383 (Review 19.10), ответ стр. 453–454: 33 до и 70 после праздника, f_e = 51.5,
    χ² = 13.29 > 3.84, p < .001; «[χ²(1, n = 103) = 13.29, p < .001, φ²c = .13]».
    [предл.] Точный биномиальный P = .00034 (две категории)."""
    res = chi2_goodness_of_fit(["before", "after"], [33, 70])
    st = res.statistics
    assert st["chi2"] == pytest.approx(13.2913, abs=1e-4)
    assert round(st["phi_c_squared"], 2) == 0.13
    assert st["binomial_p"] == pytest.approx(stats.binomtest(33, 103).pvalue)
    assert st["binomial_p"] < 0.001


def test_margin_of_error_section_12_7():
    """Witte 12.7, стр. 228: 64 % из 1500 «±3 %» при 95 %. [предл.] ДИ Уилсона 61.5 %…66.4 %
    (полуширина ≈ 2.4 %; книга округляет вверх до 3 %)."""
    cell = _cells(chi2_goodness_of_fit(["yes", "no"], [960, 540]))["yes"]
    assert cell["p_observed"] == pytest.approx(0.64)
    assert (cell["ci_upper"] - cell["ci_lower"]) / 2 == pytest.approx(0.0243, abs=5e-4)


def test_small_expected_review_19_16_and_combining():
    """Witte, стр. 385 (Review 19.16): n = 60, f_e(AB) = .05 · 60 = 3 < 5 → предупреждение;
    после объединения B и AB все f_e ≥ 5 (f_e = 26.4, 24.6, 9)."""
    raw = chi2_goodness_of_fit(["O", "A", "B", "AB"], [27, 26, 4, 3],
                                proportions="O=.44; A=.41; B=.10; AB=.05")
    warn = next(w for w in raw.warnings if w.code == "gof_small_expected")
    assert warn.params == {"levels": "AB"}
    merged = chi2_goodness_of_fit(["O", "A", "B+AB"], [27, 26, 7],
                                   proportions="O=.44; A=.41; B+AB=.15")
    assert [c["expected"] for c in merged.parameters["cells"]] == pytest.approx([26.4, 24.6, 9])
    assert not merged.warnings
    assert merged.p_value > 0.01  # H0 сохраняется на уровне .01


def test_ratios_normalized_and_missing_category_counts_zero():
    """Доли можно задать отношениями (9:3:3:1) — нормируются с info; категория из H0, которой нет
    в данных, получает f_o = 0, а не выпадает."""
    res = chi2_goodness_of_fit(["YR", "YW", "GR"], [315, 108, 101],
                                proportions="YR=9; YW=3; GR=3; GW=1")
    assert "gof_normalized" in [w.code for w in res.warnings]
    cells = _cells(res)
    assert cells["GW"]["observed"] == 0
    assert cells["GW"]["expected"] == pytest.approx(524 / 16)
    assert res.statistics["df"] == 3


def test_g2_and_cohens_w():
    """[предл.] G² = 2Σ f_o ln(f_o/f_e) и w = √(χ²/n) — для групп крови."""
    res = chi2_goodness_of_fit(*_BLOOD, proportions="O=.44; A=.41; B=.10; AB=.05")
    obs, exp = [38, 38, 20, 4], [44, 41, 10, 5]
    g2 = stats.power_divergence(obs, exp, lambda_="log-likelihood").statistic
    assert res.statistics["g2"] == pytest.approx(g2)
    assert res.statistics["cohens_w"] == pytest.approx((res.statistics["chi2"] / 100) ** 0.5)


def test_rows_of_raw_labels_equal_counts():
    """Сырые метки по строкам = сведённые частоты."""
    labels = ["O"] * 38 + ["A"] * 38 + ["B"] * 20 + ["AB"] * 4 + ["", None]
    raw = chi2_goodness_of_fit(labels, proportions="O=.44; A=.41; B=.10; AB=.05")
    agg = chi2_goodness_of_fit(*_BLOOD, proportions="O=.44; A=.41; B=.10; AB=.05")
    assert raw.statistics["chi2"] == pytest.approx(agg.statistics["chi2"])
    assert raw.n == 100


def test_guards():
    assert chi2_goodness_of_fit([]).warnings[0].code == "gof_empty"
    assert chi2_goodness_of_fit(["a", "a"]).warnings[0].code == "gof_one_category"
    unknown = chi2_goodness_of_fit(["a", "b", "c"], proportions="a=1; b=1")
    assert unknown.warnings[0].code == "gof_unknown_levels"
    assert chi2_goodness_of_fit(["a", "b"], proportions="a=1; b=0").warnings[0].code == \
        "gof_bad_proportions"
    with pytest.raises(ValueError):
        parse_proportions("a=1")
    with pytest.raises(ValueError):
        chi2_goodness_of_fit(["a", "b"], alpha=0)
