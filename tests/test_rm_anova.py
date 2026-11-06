"""rm_anova — ANOVA повторных измерений (Witte & Witte, 11th ed., гл. 17; пакет W4).

Book-тесты по точным значениям; книжные округления — в докстрингах. Поправки на
сферичность [предл.: Greenhouse & Geisser 1959; Huynh & Feldt 1976] и тест Маучли
сверены с pingouin 0.7.0 (его Маучли совпадает с R mauchly.test) и проверены по
инвариантам. Страница PDF = печатная + 18.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from core.stats.anova import one_way_anova
from core.stats.hypothesis import paired_t
from core.stats.repeated import rm_anova, sphericity
from tests import witte_data as W


def _long(matrix: list[list[float]], labels: list[str] | None = None) -> tuple[list, list, list]:
    """Матрица субъекты × условия → (values, condition, subject)."""
    m = np.asarray(matrix, dtype=float)
    labels = labels or [f"c{j}" for j in range(m.shape[1])]
    values = m.ravel().tolist()
    cond = [labels[j] for _ in range(m.shape[0]) for j in range(m.shape[1])]
    subj = [f"s{i}" for i in range(m.shape[0]) for _ in range(m.shape[1])]
    return values, cond, subj


def _pairs(res, block: str = "comparisons") -> dict[tuple[str, str], dict]:
    return {(c["group1"], c["group2"]): c for c in res.parameters[block]}


# --- Witte: сон, Tables 17.3, 17.5, 17.6, 17.8 ------------------------------------------

def test_sleep_tables_17_3_and_17_5():
    """Witte 17.4–17.7, стр. 328 (Table 17.3): SS_between = 54, SS_within = 22, SS_subject = 18,
    SS_error = 4, SS_total = 76; стр. 329 (Table 17.4): df 2, 2, 4; стр. 331 (Table 17.5):
    MS 27 и 1, F = 27 > 18.00 (α = .01, df 2 и 4). Точно p = .0048."""
    res = rm_anova(*_long(W.RM_SLEEP, W.SLEEP_LABELS))
    st = res.statistics
    assert (st["ss_between"], st["ss_subject"], st["ss_error"]) == pytest.approx((54, 18, 4))
    assert (st["ss_within"], st["ss_total"]) == pytest.approx((22, 76))
    assert (st["df_between"], st["df_subject"], st["df_error"]) == (2, 2, 4)
    assert (st["ms_between"], st["ms_error"], st["F"]) == pytest.approx((27, 1, 27))
    assert res.p_value == pytest.approx(0.004756, abs=1e-6)
    assert [r["term"] for r in res.parameters["anova_table"]] == ["Between", "Subject", "Error", "Total"]


def test_sleep_more_powerful_than_one_factor_table_17_6():
    """Witte, стр. 332 (Table 17.6): на тех же числах F = 7.36 (независимые) против 27."""
    rm = rm_anova(*_long(W.RM_SLEEP, W.SLEEP_LABELS))
    one = one_way_anova(*np.array(W.RM_SLEEP).T)
    assert one.statistics["F"] == pytest.approx(7.3636, abs=1e-4)
    assert rm.statistics["F"] > one.statistics["F"]
    assert rm.statistics["ss_within"] == pytest.approx(one.statistics["ss_within"])


def test_sleep_partial_eta_formula_17_6():
    """Witte 17.8, стр. 333 (формула 17.6): η²_p = 54 / (54 + 4) = .93 (против η² = .71)."""
    res = rm_anova(*_long(W.RM_SLEEP, W.SLEEP_LABELS))
    assert res.statistics["eta_squared_partial"] == pytest.approx(54 / 58)


def test_sleep_tukey_table_17_8():
    """Witte 17.9, стр. 334 (ф. 17.7, Table 17.8): q(3, 4) = 5.04, HSD = 5.04 · √(1/3) = 2.87;
    все три пары значимы; стр. 335 (ф. 17.8): d = 6/√1 = 6 и 3/√1 = 3.
    Точно HSD = 2.910: книга округляет √(1/3) = .577 до .57 — допуск к точному значению."""
    rows = _pairs(rm_anova(*_long(W.RM_SLEEP, W.SLEEP_LABELS)))
    assert all(r["reject"] for r in rows.values())
    hsd = rows[("0", "48")]["mean_diff"] - rows[("0", "48")]["ci_lower"]
    assert hsd == pytest.approx(2.910, abs=5e-4)
    assert rows[("0", "48")]["cohens_d"] == pytest.approx(6)
    assert rows[("0", "24")]["cohens_d"] == pytest.approx(3)


def test_sleep_degenerate_sphericity_lower_bound():
    """В Table 17.1 профиль каждого субъекта линеен (0-3-6, 4-6-8, 2-6-10): квадратичный контраст
    у всех нулевой, ковариация контрастов ранга 1 → ε_ГГ = нижней границе 1/(k − 1) = .5,
    тест Маучли не определён (info), P_ГГ = .035 — вывод не меняется."""
    res = rm_anova(*_long(W.RM_SLEEP, W.SLEEP_LABELS))
    st = res.statistics
    assert st["epsilon_gg"] == pytest.approx(0.5)
    assert st["epsilon_lower_bound"] == 0.5
    assert st["mauchly_w"] is None
    assert "rm_mauchly_undefined" in [w.code for w in res.warnings]
    assert st["p_gg"] == pytest.approx(0.035099, abs=1e-6)


# --- Witte: шум, Progress Checks 17.2–17.4 ----------------------------------------------

def test_noise_progress_checks_17_2_to_17_4():
    """Witte, стр. 332 (Progress Check 17.2), ответы стр. 446: SS 70.33 / 27.83 / 40.34, df 2, 5, 10,
    MS 35.17 и 4.03, F = 8.73 > 4.10; 17.3: η²_p = .64; 17.4: HSD = 3.88 · .82 = 3.18, значима
    только тишина–рок (4.84), d = 4.84/√4.03 = 2.41.
    Точно: SS_error = 40.333 (книга 40.34 — вычитание округлённых), F = 8.719 (книга делит
    округлённые MS), HSD = 3.179 — допуск approx к точным значениям."""
    res = rm_anova(*_long(W.RM_NOISE, W.RM_NOISE_LABELS))
    st = res.statistics
    assert st["ss_between"] == pytest.approx(70.333, abs=1e-3)
    assert st["ss_subject"] == pytest.approx(27.833, abs=1e-3)
    assert st["ss_error"] == pytest.approx(40.333, abs=1e-3)
    assert st["F"] == pytest.approx(8.719, abs=1e-3)
    assert round(st["eta_squared_partial"], 2) == 0.64
    rows = _pairs(res)
    assert [k for k, r in rows.items() if r["reject"]] == [("silence", "rock")]
    hsd = rows[("silence", "rock")]["mean_diff"] - rows[("silence", "rock")]["ci_lower"]
    assert hsd == pytest.approx(3.179, abs=1e-3)
    assert round(abs(rows[("silence", "rock")]["cohens_d"]), 2) == 2.41


def test_noise_sphericity_matches_pingouin():
    """Шум (k = 3, n = 6): pingouin 0.7.0 — W = .145345, P_Маучли = .021125, ε_ГГ = .539184,
    ε_ХФ = .569942, P_ГГ = .027958. Маучли значим, но вывод по P не меняется — переключения нет."""
    res = rm_anova(*_long(W.RM_NOISE, W.RM_NOISE_LABELS))
    st = res.statistics
    assert st["mauchly_w"] == pytest.approx(0.145345, abs=1e-6)
    assert st["mauchly_p"] == pytest.approx(0.021125, abs=1e-6)
    assert st["epsilon_gg"] == pytest.approx(0.539184, abs=1e-6)
    assert st["epsilon_hf"] == pytest.approx(0.569942, abs=1e-6)
    assert st["p_gg"] == pytest.approx(0.027958, abs=1e-6)
    assert "rm_sphericity_conflict" not in [w.code for w in res.warnings]


# --- Witte: Review 17.6 (ошибка книги) и 17.7 -------------------------------------------

def test_eye_review_17_6_book_error():
    """Witte, стр. 337 (Review 17.6), ответ стр. 447. Книга: SS_between = 154.12, SS_subject = 67.87,
    SS_error = 64.88 = 132.75 − 67.87, F = 16.62, η²_p = .70. Ошибка книги: SS_within = 132.75
    взят из 16.5, но столбец THREE в 17.6 другой (см. witte_data.RM_EYE). Для данных 17.6 точно:
    SS_within = 72.75, SS_error = 4.875, F = 221.3, η²_p = .969. Тест — по точным числам; не
    «исправлять» под книгу. SS_between и SS_subject в книге верны."""
    res = rm_anova(*_long(W.RM_EYE))
    st = res.statistics
    assert st["ss_between"] == pytest.approx(154.125)
    assert st["ss_subject"] == pytest.approx(67.875)
    assert st["ss_within"] == pytest.approx(72.75)
    assert st["ss_error"] == pytest.approx(4.875)
    assert st["F"] == pytest.approx(221.3077, abs=1e-4)
    assert st["ss_error"] != pytest.approx(64.88, abs=1)  # книжное значение
    three_17_6 = [row[3] for row in W.RM_EYE]
    assert sum(three_17_6) == sum(W.EYE_CONTACTS[3]) == 60  # суммы совпадают
    assert three_17_6 != W.EYE_CONTACTS[3]  # а значения — нет


def test_vodka_review_17_7_sphericity_k5_matches_pingouin():
    """Witte, стр. 338 (Review 17.7): водка, k = 5, n = 5. SS_between = 523.36 (как в 16.11).
    pingouin 0.7.0: W = .000315, χ² = 19.483, P_Маучли = .049593 (с поправкой второго порядка,
    как R), ε_ГГ = .333489, ε_ХФ = .437894, P_ГГ = .040501."""
    st = rm_anova(*_long(W.RM_VODKA)).statistics
    assert st["ss_between"] == pytest.approx(523.36)
    assert st["mauchly_w"] == pytest.approx(0.000315, abs=1e-6)
    assert st["mauchly_chi2"] == pytest.approx(19.48275, abs=1e-5)
    assert st["mauchly_p"] == pytest.approx(0.049593, abs=1e-6)
    assert st["epsilon_gg"] == pytest.approx(0.333489, abs=1e-6)
    assert st["epsilon_hf"] == pytest.approx(0.437894, abs=1e-6)
    assert st["p_gg"] == pytest.approx(0.040501, abs=1e-6)


# --- Инварианты поправок ----------------------------------------------------------------

def _compound_symmetric(n: int, k: int, seed: int) -> np.ndarray:
    """Данные, у которых выборочная ковариация контрастов в точности ∝ I (сферичность)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k - 1))
    z -= z.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z, rowvar=False)).T)
    helmert = np.array([[1.0] * i + [-float(i)] + [0.0] * (k - i - 1) for i in range(1, k)])
    helmert /= np.linalg.norm(helmert, axis=1, keepdims=True)
    return z @ helmert + rng.normal(size=(n, 1)) * 5 + np.arange(k)


def test_compound_symmetry_gives_epsilon_one():
    """Идеальная сферичность: ε_ГГ = ε_ХФ = 1, W = 1, P_ГГ = P."""
    y = _compound_symmetric(10, 4, seed=3)
    eps = sphericity(y)
    assert eps["epsilon_gg"] == pytest.approx(1)
    assert eps["epsilon_hf"] == 1
    assert eps["mauchly_w"] == pytest.approx(1)
    res = rm_anova(*_long(y.tolist()))
    assert res.statistics["p_gg"] == pytest.approx(res.p_value)


@pytest.mark.parametrize("seed", range(20))
def test_epsilon_bounds(seed):
    """1/(k − 1) ≤ ε_ГГ ≤ ε_ХФ ≤ 1 на случайных данных (k = 3…6)."""
    rng = np.random.default_rng(seed)
    k = 3 + seed % 4
    y = rng.normal(size=(6 + seed, k)) * rng.uniform(0.5, 5, size=k)
    eps = sphericity(y)
    assert 1 / (k - 1) - 1e-12 <= eps["epsilon_gg"] <= eps["epsilon_hf"] <= 1


# --- Блок парных t, k = 2, предупреждения -----------------------------------------------

def test_paired_block_is_holm_over_paired_t():
    """[предл.: Maxwell & Delaney 2004] Парные t по разностям = paired_t; P — по Холму; d_z = D̄/s_D."""
    res = rm_anova(*_long(W.RM_NOISE, W.RM_NOISE_LABELS))
    rows = _pairs(res, "paired_comparisons")
    m = np.array(W.RM_NOISE, float)
    raw = []
    for (i, a), (j, b) in [((0, "silence"), (1, "white")), ((0, "silence"), (2, "rock")),
                           ((1, "white"), (2, "rock"))]:
        ref = paired_t(m[:, j], m[:, i])
        assert rows[(a, b)]["t"] == pytest.approx(ref.statistics["t"])
        assert rows[(a, b)]["cohens_dz"] == pytest.approx(ref.statistics["cohens_dz"])
        raw.append(ref.p_value)
    holm = multipletests(raw, method="holm")[1]
    assert [r["p_adj"] for r in rows.values()] == pytest.approx(list(holm))
    assert res.parameters["comparisons_assumption"] == "sphericity"


def test_two_conditions_trivial_sphericity_equals_paired_t():
    """k = 2: ε = 1 без Маучли (info, не ошибка), F = t² парного t (EPO, Table 15.2: t = 7.319)."""
    y = np.column_stack([W.EPO_X1, W.EPO_X2]).tolist()
    res = rm_anova(*_long(y, ["epo", "control"]))
    assert res.statistics["epsilon_gg"] == 1
    assert "mauchly_w" not in res.statistics
    assert "rm_two_conditions" in [w.code for w in res.warnings]
    t = paired_t(W.EPO_X2, W.EPO_X1)
    assert res.statistics["F"] == pytest.approx(t.statistics["t"] ** 2)
    assert res.p_value == pytest.approx(t.p_value)
    assert len(res.parameters["paired_comparisons"]) == 1


def test_sphericity_conflict_warning():
    """Обычный P < .05, а P_ГГ > .05 → предупреждение с обоими P."""
    y = [[-6.3, -4.7, -6.0, 0.2], [-0.6, -1.6, -0.2, 1.5], [-4.8, -4.7, -5.2, 5.9],
         [1.6, 4.8, 3.7, 5.1], [-0.6, 2.1, 0.8, 11.3], [1.1, 2.3, 1.2, 17.4],
         [0.2, 0.5, 2.3, -3.3], [-0.3, -1.6, 0.7, -5.5]]
    res = rm_anova(*_long(y))
    assert res.p_value < 0.05 < res.statistics["p_gg"]
    warn = next(w for w in res.warnings if w.code == "rm_sphericity_conflict")
    assert warn.params["p_gg"] == round(res.statistics["p_gg"], 4)


def test_incomplete_subjects_dropped_with_mixed_model_hint():
    values, cond, subj = _long(W.RM_NOISE, W.RM_NOISE_LABELS)
    values[0] = float("nan")  # у первого субъекта нет «тишины»
    res = rm_anova(values, cond, subj)
    warn = next(w for w in res.warnings if w.code == "rm_incomplete_subjects")
    assert warn.params == {"dropped": 1}
    assert "mixed_model" in warn.message
    assert res.statistics["n_subjects"] == 5


def test_profiles_matrix():
    res = rm_anova(*_long(W.RM_NOISE, W.RM_NOISE_LABELS))
    prof = res.parameters["profiles"]
    assert prof["conditions"] == W.RM_NOISE_LABELS
    assert prof["values"] == [[float(v) for v in row] for row in W.RM_NOISE]
    assert len(prof["subjects"]) == 6


def test_guards():
    values, cond, subj = _long(W.RM_NOISE, W.RM_NOISE_LABELS)
    with pytest.raises(ValueError):
        rm_anova(values + [1.0], cond + ["rock"], subj + ["s0"])  # повтор ячейки
    with pytest.raises(ValueError):
        rm_anova(values, cond, subj, alpha=0)
    assert rm_anova([1, 2], ["a", "a"], ["s1", "s2"]).warnings[-1].code == "too_few_groups"
    assert rm_anova([1, 2], ["a", "b"], ["s1", "s1"]).warnings[-1].code == "n_too_small"
    flat = rm_anova(*_long([[1, 2, 3], [2, 3, 4], [5, 6, 7]]))
    assert flat.warnings[-1].code == "rm_zero_error"
