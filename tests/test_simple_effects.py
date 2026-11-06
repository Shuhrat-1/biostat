"""simple_effects — простые эффекты двухфакторного ANOVA (Witte & Witte, 11th ed., гл. 18.9;
пакет W4). Book-тесты по точным значениям; книжные округления — в докстрингах.
Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest
from scipy import stats

from core.stats.anova import one_way_anova, tukey_hsd
from core.stats.simple_effects import simple_effects
from tests import witte_data as W

_NAMES = {"factor_a": "crowd", "factor_b": "danger"}


def _effects(res) -> dict[tuple[str, str], dict]:
    return {(r["effect_of"], r["level"]): r for r in res.parameters["simple_effects"]}


def _pairs(res) -> dict[tuple[str, str], dict]:
    return {(c["group1"], c["group2"]): c for c in res.parameters["comparisons"]}


def test_smoke_crowd_at_danger_formula_18_6():
    """Witte 18.9, стр. 356–357 (ф. 18.5–18.6, Table 18.7): SS_se(толпа | неопасно) =
    20²/2 + 34²/2 + 42²/2 − 96²/6 = 124, MS_se = 62, F_se = 62 / 5.33 = 11.63 (p < .01, крит. 10.92);
    при «опасно» F_se = 0.38 (ns). Точно 11.625 и 0.375 (книга делит на округлённое 5.33)."""
    res = simple_effects(*W.SMOKE, column_names=_NAMES)
    eff = _effects(res)
    non, dang = eff[("crowd", "nondangerous")], eff[("crowd", "dangerous")]
    assert (non["ss"], non["df"], non["ms"]) == pytest.approx((124, 2, 62))
    assert non["F"] == pytest.approx(11.625)
    assert non["p"] < 0.01
    assert dang["F"] == pytest.approx(0.375)
    assert dang["p"] > 0.05
    assert res.statistics["ms_within"] == pytest.approx(32 / 6)
    assert res.statistics["df_within"] == 6


def test_smoke_danger_at_crowd_progress_check_18_6():
    """Witte, стр. 358 (Progress Check 18.6), ответ стр. 449: F_se(опасность | 0) = 4 / 5.33 = 0.75,
    при 2 — 100 / 5.33 = 18.76, при 4 — 144 / 5.33 = 27.02; df 1 и 6. Точно 0.75, 18.75, 27.00."""
    eff = _effects(simple_effects(*W.SMOKE, column_names=_NAMES))
    assert eff[("danger", "0")]["F"] == pytest.approx(0.75)
    assert eff[("danger", "2")]["F"] == pytest.approx(18.75)
    assert eff[("danger", "4")]["F"] == pytest.approx(27.0)
    assert eff[("danger", "0")]["df"] == 1


def test_smoke_tukey_in_cells_page_357():
    """Witte, стр. 357: HSD в «неопасно» с n = 2 ячейки: q(.01; 3, 6) = 6.33 → 10.32, разность 0–4 = 11
    значима (p < .01); q(.05) = 4.34 → 7.07, разность 0–2 = 7 «пограничная» (в отчёте «p = .05»,
    стр. 359–360). Точный P Тьюки для 0–2 = .052 — чуть выше .05."""
    pairs = _pairs(simple_effects(*W.SMOKE, column_names=_NAMES))
    p04 = pairs[("0 @ nondangerous", "4 @ nondangerous")]
    p02 = pairs[("0 @ nondangerous", "2 @ nondangerous")]
    assert p04["mean_diff"] == 11 and p04["p_adj"] < 0.01
    assert p02["mean_diff"] == 7
    assert 0.05 < p02["p_adj"] < 0.06
    hsd05 = stats.studentized_range.ppf(0.95, 3, 6) * (16 / 6) ** 0.5
    assert p02["mean_diff"] - p02["ci_lower"] == pytest.approx(hsd05)
    assert round(hsd05, 2) == 7.09  # книга 7.07: q и √(5.33/2) округлены


def test_smoke_cohens_d_pages_358_and_449():
    """Witte, стр. 358: d(4, 0 | неопасно) = 11 / √5.33 = 4.76, d(2, 0) = 7 / 2.31 = 3.03;
    ответ 18.6 (стр. 450): d(опасность | 2) = 10 / √5.33 = 4.33, d(| 4) = 12 / √5.33 = 5.19
    (точно 5.196; книга делит на 2.31)."""
    pairs = _pairs(simple_effects(*W.SMOKE, column_names=_NAMES))
    assert round(pairs[("0 @ nondangerous", "4 @ nondangerous")]["cohens_d"], 2) == 4.76
    assert round(pairs[("0 @ nondangerous", "2 @ nondangerous")]["cohens_d"], 2) == 3.03
    assert round(pairs[("dangerous @ 2", "nondangerous @ 2")]["cohens_d"], 2) == 4.33
    assert pairs[("dangerous @ 4", "nondangerous @ 4")]["cohens_d"] == pytest.approx(12 / (32 / 6) ** 0.5)


def test_interaction_context_and_bonferroni():
    """F взаимодействия = 5.25 (Table 18.6) — для контекста; гейтинга нет: оба набора всегда.
    [предл.] P по Бонферрони в пределах набора: толпа | неопасно — 2 · .0086."""
    res = simple_effects(*W.SMOKE, column_names=_NAMES)
    assert res.statistics["F_interaction"] == pytest.approx(5.25)
    eff = _effects(res)
    non = eff[("crowd", "nondangerous")]
    assert non["p_bonferroni"] == pytest.approx(2 * non["p"])
    assert eff[("danger", "4")]["p_bonferroni"] == pytest.approx(3 * eff[("danger", "4")]["p"])
    assert {r["family"] for r in res.parameters["simple_effects"]} == {"crowd | danger", "danger | crowd"}


def test_single_simple_effect_vs_one_way_with_pooled_error():
    """SS_se = SS_between однофакторного ANOVA по ячейкам этого уровня; отличается только
    знаменатель (общий MS_within) и df ошибки (стр. 356)."""
    values, a, b = W.SMOKE
    sub = [[v for v, x, y in zip(values, a, b) if y == "nondangerous" and x == lev] for lev in ("0", "2", "4")]
    one = one_way_anova(*sub)
    eff = _effects(simple_effects(*W.SMOKE, column_names=_NAMES))[("crowd", "nondangerous")]
    assert eff["ss"] == pytest.approx(one.statistics["ss_between"])
    assert eff["F"] != pytest.approx(one.statistics["F"])


def test_unequal_n_tukey_kramer():
    """Неравные n: SS_se = Σ n(ȳ − ȳ_уровня)²; при одном уровне второго фактора пары совпадают
    с tukey_hsd на тех же ячейках, если MS_within общий (здесь — равен однофакторному)."""
    values = [1, 3, 6, 2, 1, 4, 7, 5, 7, 12, 10, 9] + [2, 4, 3, 5, 6, 8, 9, 7, 6, 8, 10, 11]
    a = ["0"] * 5 + ["24"] * 3 + ["48"] * 4 + ["0"] * 4 + ["24"] * 4 + ["48"] * 4
    b = ["x"] * 12 + ["y"] * 12
    res = simple_effects(values, a, b)
    eff = _effects(res)[("A", "x")]
    grand = sum(values[:12]) / 12
    means = {"0": sum(values[:5]) / 5, "24": sum(values[5:8]) / 3, "48": sum(values[8:12]) / 4}
    assert eff["ss"] == pytest.approx(5 * (means["0"] - grand) ** 2 + 3 * (means["24"] - grand) ** 2
                                      + 4 * (means["48"] - grand) ** 2)
    assert len(res.parameters["comparisons"]) == 3 + 3 + 3  # 2 набора A | B + 3 пары B | A
    tk = tukey_hsd(values[:5], values[5:8], values[8:12], labels=["0", "24", "48"])
    assert tk.parameters["comparisons"][0]["mean_diff"] == pytest.approx(
        _pairs(res)[("0 @ x", "24 @ x")]["mean_diff"])


def test_guards():
    values, a, b = W.SMOKE
    assert simple_effects(values, ["0"] * 12, b).warnings[0].code == "too_few_groups"
    gap = [i for i, (x, y) in enumerate(zip(a, b)) if not (x == "4" and y == "dangerous")]
    res = simple_effects([values[i] for i in gap], [a[i] for i in gap], [b[i] for i in gap])
    assert res.warnings[0].code == "simple_empty_cell"
    assert res.warnings[0].params == {"missing": 1}
    single = simple_effects([1, 2, 3, 4], ["a", "a", "b", "b"], ["x", "y", "x", "y"])
    assert single.warnings[0].code == "n_too_small"
    with pytest.raises(ValueError):
        simple_effects(values, a, b, alpha=2)
