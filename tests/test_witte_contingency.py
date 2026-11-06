"""Book-тесты Witte & Witte (11th ed., 2017), гл. 19.6–19.15: χ² независимости.

contingency_test — из Welham (стр. 513); Witte добавляет φ²c, χ² без поправки Йейтса,
ДИ отношения шансов и отношение рисков новыми ключами (пакет W3). Witte поправку Йейтса
не применяет; поведение по умолчанию (correction = auto) не меняется — сравнение с книгой
по chi2_uncorrected. Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest

from core.stats.categorical import contingency_test


def _run(table: dict[tuple[str, str], int], **kwargs):
    rows = [r for r, _ in table]
    cols = [c for _, c in table]
    return contingency_test(rows, cols, count=list(table.values()), **kwargs)


LETTERS = {("no", "campus"): 19, ("no", "downtown"): 21, ("no", "suburbia"): 40,
           ("yes", "campus"): 51, ("yes", "downtown"): 39, ("yes", "suburbia"): 30}

# Review 19.14 (стр. 384): Титаник; первый столбец (событие) — «lived» < «perished».
TITANIC = {("cabin", "lived"): 299, ("cabin", "perished"): 280,
           ("steerage", "lived"): 186, ("steerage", "perished"): 526}


def test_lost_letters_tables_19_5_and_sas_19_8():
    """Witte 19.8–19.10, стр. 375 (Table 19.5): χ² = 14.02, df 2 > 5.99; стр. 381 (Table 19.8, SAS):
    Chi-Square 14.018, Prob .0009, Likelihood Ratio 14.049, Cramer's V .265."""
    st = _run(LETTERS).statistics
    assert st["chi2"] == pytest.approx(14.018, abs=1e-3)
    assert st["df"] == 2
    assert round(st["p"], 4) == 0.0009
    assert st["g2"] == pytest.approx(14.049, abs=1e-3)
    assert round(st["cramers_v"], 3) == 0.265


def test_lost_letters_phi_c_squared_formula_19_6():
    """Witte 19.11, стр. 378 (ф. 19.6): φ²c = 14.02 / (200 · 1) = .07;
    Progress Check 19.8c (ответ стр. 453): квадрат SAS «Cramer's V» .265² = .07."""
    st = _run(LETTERS).statistics
    assert st["cramers_v_squared"] == pytest.approx(st["chi2_uncorrected"] / 200)
    assert st["cramers_v_squared"] == pytest.approx(st["cramers_v"] ** 2)
    assert round(st["cramers_v_squared"], 2) == 0.07


def test_poison_oak_progress_checks_19_4_19_5():
    """Witte, стр. 376 (Progress Check 19.4), ответы стр. 452–453: χ² = 15.28, df 3 > 11.34 (α = .01);
    19.5: φ²c = 15.28 / (300 · 1) = .05."""
    table = {("no rash", "black"): 40, ("no rash", "blond"): 30, ("no rash", "brown"): 30,
             ("no rash", "red"): 20, ("rash", "black"): 80, ("rash", "blond"): 30,
             ("rash", "brown"): 60, ("rash", "red"): 10}
    st = _run(table).statistics
    assert st["chi2"] == pytest.approx(15.278, abs=1e-3)
    assert st["p"] < 0.01
    assert round(st["cramers_v_squared"], 2) == 0.05


def test_titanic_review_19_14_without_yates():
    """Witte, стр. 384 (Review 19.14), ответ стр. 454–455: χ² = 88.65 (без поправки Йейтса),
    φ²c = 88.65 / 1291 = .07, OR = 1.07 / .35 = 3.06. По умолчанию contingency_test для 2 × 2
    применяет Йейтса (87.56) — книжное число в chi2_uncorrected. Точно OR = 3.02: книга делит
    округлённые шансы."""
    st = _run(TITANIC).statistics
    assert st["yates"] is True
    assert st["chi2"] == pytest.approx(87.564, abs=1e-3)
    assert st["chi2_uncorrected"] == pytest.approx(88.649, abs=1e-3)
    assert round(st["cramers_v_squared"], 2) == 0.07
    assert st["odds_ratio"] == pytest.approx(299 * 526 / (280 * 186))
    assert round(st["odds_ratio"], 2) == 3.02
    no_yates = _run(TITANIC, correction="no").statistics
    assert no_yates["chi2"] == pytest.approx(st["chi2_uncorrected"])


def test_titanic_or_ci_and_risk_ratio():
    """[предл.: Woolf 1955; Katz 1978] ДИ OR 2.39…3.81 и RR = (299/579)/(186/712) = 1.98,
    ДИ 1.71…2.29 — как statsmodels Table2x2. Книга: «в 3.06 раза вероятнее выжил» (стр. 455) —
    так читается RR, а не OR: вероятность выжить выше в 1.98 раза."""
    res = _run(TITANIC)
    st = res.statistics
    assert (st["or_ci_lower"], st["or_ci_upper"]) == pytest.approx((2.39099, 3.81412), abs=1e-4)
    assert st["risk_ratio"] == pytest.approx((299 / 579) / (186 / 712))
    assert (st["rr_ci_lower"], st["rr_ci_upper"]) == pytest.approx((1.70735, 2.28874), abs=1e-4)
    assert res.parameters["ratio_orientation"] == {"event": "lived", "group1": "cabin",
                                                   "group2": "steerage"}


def test_aspirin_table_19_7_and_campus_progress_check_19_6():
    """Witte, стр. 379 (Table 19.7): OR = (189/10 845)/(104/10 933) = 1.83; χ²(1) = 25.01, φ²c = .001
    (стр. 378). Progress Check 19.6 (ответ стр. 453): кампус против остальных, χ²(1) = 7.42,
    OR = (51/19)/(69/61) = 2.37."""
    aspirin = _run({("aspirin", "attack"): 104, ("aspirin", "no attack"): 10933,
                    ("placebo", "attack"): 189, ("placebo", "no attack"): 10845}).statistics
    assert 1 / aspirin["odds_ratio"] == pytest.approx(1.83, abs=5e-3)
    assert aspirin["chi2_uncorrected"] == pytest.approx(25.01, abs=5e-3)
    assert round(aspirin["cramers_v_squared"], 3) == 0.001
    campus = _run({("campus", "returned"): 51, ("campus", "unreturned"): 19,
                   ("off", "returned"): 69, ("off", "unreturned"): 61}).statistics
    assert campus["chi2_uncorrected"] == pytest.approx(7.42, abs=5e-3)
    assert round(campus["odds_ratio"], 2) == 2.37


def test_religion_review_19_13_large_table():
    """Witte, стр. 383–384 (Review 19.13), ответ стр. 454: χ² = 220, df 12, φ²c = 220/(500 · 3) = .15."""
    counts = [[30, 30, 40, 60, 40], [10, 10, 40, 20, 20], [10, 10, 20, 20, 40], [0, 0, 0, 0, 100]]
    rows, cols = ["a_dem", "b_rep", "c_ind", "d_oth"], ["1bud", "2jew", "3pro", "4rc", "5oth"]
    table = {(r, c): counts[i][j] for i, r in enumerate(rows) for j, c in enumerate(cols)}
    st = _run(table).statistics
    assert st["chi2"] == pytest.approx(220)
    assert st["df"] == 12
    assert round(st["cramers_v_squared"], 2) == 0.15
    assert "or_ci_lower" not in st  # ДИ отношений — только для 2 × 2


def test_large_n_review_19_17_19_18():
    """Witte, стр. 385 (Review 19.17–19.18): те же доли, n × 10 → χ² × 10 (без Йейтса),
    φ²c неизменен."""
    small = _run({("f", "favor"): 30, ("f", "oppose"): 10, ("m", "favor"): 40, ("m", "oppose"): 20})
    large = _run({("f", "favor"): 300, ("f", "oppose"): 100, ("m", "favor"): 400, ("m", "oppose"): 200})
    assert large.statistics["chi2_uncorrected"] == pytest.approx(10 * small.statistics["chi2_uncorrected"])
    assert small.statistics["p_uncorrected"] > 0.05 > large.statistics["p_uncorrected"]
    assert large.statistics["cramers_v_squared"] == pytest.approx(small.statistics["cramers_v_squared"])


def test_zero_cell_haldane_for_intervals():
    """Нулевая ячейка: ДИ по Холдейну (+0.5), конечные."""
    st = _run({("a", "x"): 0, ("a", "y"): 10, ("b", "x"): 5, ("b", "y"): 5}).statistics
    assert 0 < st["or_ci_lower"] < st["or_ci_upper"] < float("inf")
