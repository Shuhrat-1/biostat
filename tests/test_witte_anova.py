"""Book-тесты Witte & Witte (11th ed., 2017), гл. 16: однофакторный ANOVA.

one_way_anova и tukey_hsd — из Welham (гл. 4, 8); Witte добавляет ω², объединённое
SD и d для пар новыми ключами (пакет W2). Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest

from core.stats.anova import omega_squared, one_way_anova, tukey_hsd
from core.stats.hypothesis import two_sample_t
from tests import witte_data as W


def test_sleep_outcome_b_tables_16_3_16_6_16_9():
    """Witte 16.4–16.7, стр. 302 (Table 16.3): SS_between = 54, SS_within = 22, SS_total = 76;
    стр. 307 (Table 16.6): MS = 27 и 3.67, F = 7.36, df 2 и 6;
    стр. 317 (Table 16.9, Minitab): p = 0.024, Pooled StDev = 1.915."""
    res = one_way_anova(*W.SLEEP_B, labels=W.SLEEP_LABELS)
    st = res.statistics
    assert (st["ss_between"], st["ss_within"], st["ss_total"]) == (54, 22, 76)
    assert (st["df_between"], st["df_within"]) == (2, 6)
    assert round(st["ms_within"], 2) == 3.67
    assert round(st["F"], 2) == 7.36
    assert round(res.p_value, 3) == 0.024
    assert round(st["pooled_sd"], 3) == 1.915  # Progress Check 16.8b, ответ стр. 445


def test_sleep_eta_squared_formula_16_8():
    """Witte 16.9, стр. 310 (формула 16.8, Fig. 16.3 III): η² = 54/76 = .71."""
    res = one_way_anova(*W.SLEEP_B)
    assert round(res.statistics["eta_squared"], 2) == 0.71


def test_sleep_omega_squared_smaller_than_eta():
    """Witte, стр. 311: ω² — «более точный конкурент» η² (формулы нет).
    [предл.: Hays 1963] ω² = (54 − 2 · 3.667) / (76 + 3.667) = .586 < η² = .711."""
    res = one_way_anova(*W.SLEEP_B)
    assert res.statistics["omega_squared"] == pytest.approx((54 - 2 * 22 / 6) / (76 + 22 / 6))
    assert res.statistics["omega_squared"] < res.statistics["eta_squared"]


def test_sleep_outcome_a_retained_review_16_9():
    """Witte, стр. 297–298: исход A — F = 0.75 < 5.14, H0 сохранена (Review Question 16.9).
    F < 1 → ω² отрицателен по формуле, усечён до 0."""
    res = one_way_anova(*W.SLEEP_A)
    assert res.statistics["F"] == pytest.approx(0.75)
    assert res.p_value > 0.05
    assert res.statistics["omega_squared"] == 0
    assert omega_squared(6, 30, 4, 2) == 0  # (6 − 8) / 34 < 0 → 0


def test_eye_contacts_progress_checks_16_5_16_6():
    """Witte, стр. 307–308 (Progress Check 16.5), ответы стр. 444: SS = 154.12 / 132.75 / 286.87,
    df 3 и 28, MS 51.37 и 4.74, F = 10.84; 16.6: η² = 154.12 / 286.87 = .54."""
    res = one_way_anova(*W.EYE_CONTACTS)
    st = res.statistics
    assert round(st["ss_between"], 3) == 154.125
    assert round(st["ss_within"], 2) == 132.75
    assert round(st["F"], 2) == 10.84
    assert round(st["eta_squared"], 2) == 0.54


def test_unequal_n_review_16_10():
    """Witte, стр. 319 (Review Question 16.10), ответ стр. 445: SS 106.05 / 34.87 / 140.92,
    df 2 и 9, MS 53.03 и 3.87, F = 13.70, η² = .75. Книга делит округлённые MS: точно F = 13.687."""
    res = one_way_anova(*W.SLEEP_UNEQUAL)
    st = res.statistics
    assert round(st["ss_between"], 2) == 106.05
    assert round(st["ss_within"], 2) == 34.87
    assert round(st["F"], 3) == 13.687
    assert round(st["eta_squared"], 2) == 0.75


def test_vodka_and_weight_loss_review_16_11_16_13():
    """Witte, стр. 320 (Review 16.11): T = 15, 20, 26, 56, 74, G = 191, ΣX² = 2371 →
    SS_total = 2371 − 191²/25 = 911.76; стр. 321 (Review 16.13): T = 22, 12, 63, G = 97,
    ΣX² = 961, N = 23 → F(2, 20) = 2.67 < 3.49, H0 сохранена."""
    vodka = one_way_anova(*W.VODKA).statistics
    assert vodka["ss_total"] == pytest.approx(2371 - 191**2 / 25)
    weight = one_way_anova(*W.WEIGHT_LOSS)
    assert weight.statistics["ss_total"] == pytest.approx(961 - 97**2 / 23)
    assert round(weight.statistics["F"], 2) == 2.67
    assert weight.p_value > 0.05


def test_f_equals_t_squared_for_two_groups():
    """Witte 16.8, стр. 308: при двух группах F = t² (t Стьюдента). EPO (Table 14.1): t = 2.152."""
    f = one_way_anova(W.EPO_X1, W.EPO_X2)
    t = two_sample_t(W.EPO_X1, W.EPO_X2, equal_var=True)
    assert f.statistics["F"] == pytest.approx(t.statistics["t"] ** 2)
    assert f.p_value == pytest.approx(t.p_value)


def _rows(res) -> dict[tuple[str, str], dict]:
    return {(c["group1"], c["group2"]): c for c in res.parameters["comparisons"]}


def test_sleep_tukey_table_16_8_and_minitab_16_9():
    """Witte 16.10, стр. 313 (Table 16.8): HSD = 4.34 · √(3.67/3) = 4.77, значима только
    разность 6 (0 и 48 ч); стр. 317 (Table 16.9, Minitab): q = 4.34, ДИ 0–48: −10.798…−1.202,
    0–24: −7.798…1.798 (знак «столбец − строка»; у tukey_hsd — group2 − group1).
    Точно HSD = 4.797: книга округляет √1.22 до 1.10; Minitab печатает пределы на 0.001
    больше точных (10.797 → 10.798), отсюда допуск 1e-3."""
    rows = _rows(tukey_hsd(*W.SLEEP_B, labels=W.SLEEP_LABELS))
    assert [k for k, c in rows.items() if c["reject"]] == [("0", "48")]
    assert rows[("0", "48")]["ci_lower"] == pytest.approx(1.202, abs=1e-3)
    assert rows[("0", "48")]["ci_upper"] == pytest.approx(10.798, abs=1e-3)
    assert rows[("0", "24")]["ci_lower"] == pytest.approx(-1.798, abs=1e-3)
    hsd = rows[("0", "48")]["mean_diff"] - rows[("0", "48")]["ci_lower"]
    assert round(hsd, 3) == 4.797


def test_sleep_tukey_cohens_d_formula_16_10():
    """Witte, стр. 313–314 (формула 16.10): d = 6 / √3.67 = 3.13; знак — как у mean_diff."""
    res = tukey_hsd(*W.SLEEP_B, labels=W.SLEEP_LABELS)
    assert round(_rows(res)[("0", "48")]["cohens_d"], 2) == 3.13
    assert round(res.statistics["pooled_sd"], 3) == 1.915


def test_eye_contacts_tukey_progress_check_16_7():
    """Witte, стр. 314 (Progress Check 16.7), ответ стр. 444: значимы 0–2 (3.37), 0–3 (5.87),
    1–3 (4.37); d = 1.55, 2.69, 2.00. Книга: средние округлены до сотых, √4.74 = 2.18,
    q взят для df = 24 (нет 28 в таблице). Точно d(0, 3) = 5.875 / 2.177 = 2.698."""
    rows = _rows(tukey_hsd(*W.EYE_CONTACTS, labels=["0", "1", "2", "3"]))
    assert sorted(k for k, c in rows.items() if c["reject"]) == [("0", "2"), ("0", "3"), ("1", "3")]
    assert round(rows[("0", "2")]["cohens_d"], 2) == 1.55
    assert round(rows[("0", "3")]["cohens_d"], 3) == 2.698
    assert round(rows[("1", "3")]["cohens_d"], 2) == 2.01


def test_unequal_n_tukey_kramer_review_16_10():
    """Witte, стр. 319 (Review 16.10), ответ стр. 445: HSD с n = 4 → 3.87, значимы 0–48 (6.90)
    и 24–48 (4.17); d = 3.50 и 2.12. Тьюки–Крамер (точные n = 5, 3, 4) даёт для 24–48
    p = .052 — НЕ значимо: подстановка n̄ в книге занижает HSD для пары с n = 3.
    d от поправки не зависит: 3.506 и 2.117."""
    rows = _rows(tukey_hsd(*W.SLEEP_UNEQUAL, labels=W.SLEEP_LABELS))
    assert rows[("0", "48")]["reject"]
    assert not rows[("24", "48")]["reject"]
    assert round(rows[("24", "48")]["p_adj"], 3) == 0.052
    assert round(rows[("0", "48")]["cohens_d"], 2) == 3.51
    assert round(rows[("24", "48")]["cohens_d"], 2) == 2.12
