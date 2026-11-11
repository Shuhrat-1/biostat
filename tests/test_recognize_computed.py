"""Расчётные распознаватели карты (этап 2): допущения, форма связи, GLM, DIAG.

Где можно — данные книги с известным выводом: яйца жуков Table 6.1
(дисперсия растёт со средним → log, гл. 6), трансекта Table 12.7
(неадекватность F = 4.321, p = 0.013), температура по месяцам (гл. 17,
периодичность), тля (счёты со сверхдисперсией, гл. 18). Остальное —
синтетика с заданным свойством: детерминированные квантили вместо
случайных чисел, чтобы тесты не зависели от генератора.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.validation.recognize import COMPUTED, STRUCTURAL, recognize
from tests.book_ch12_data import TRANSECT
from tests.book_ch17_data import TEMPERATURE
from tests.test_recognize import APHIDS, CALCIUM, _potato, _table
from tests.test_regression_checks import ELISA_ABS, ELISA_CONC
from tests.test_transforms import BEETLES

NORMAL10 = stats.norm.ppf(np.linspace(0.05, 0.95, 10))
SKEWED10 = stats.expon.ppf(np.linspace(0.05, 0.95, 10))
# Те же квантили в «перемешанном» порядке: без тренда и серий по строкам.
MIXED10 = NORMAL10[[3, 7, 0, 5, 9, 2, 6, 1, 8, 4]]


def _computed(columns: dict[str, list], **roles: str) -> dict[str, dict]:
    out = recognize(_table(columns), roles, cls=COMPUTED)
    return {f["id"]: f for f in out["fired"]}


def _grouped(groups: dict[str, np.ndarray]) -> dict[str, list]:
    return {"g": [k for k, v in groups.items() for _ in v],
            "y": [round(float(x), 4) for v in groups.values() for x in v]}


def _params(fired: dict, rid: str) -> dict:
    return fired[rid]["message"]["params"]


# --- Классы и вызов ------------------------------------------------------------------

def test_cls_filter_splits_structural_and_computed():
    table = _table(_potato())
    roles = {"value": "yield", "group": "treatment", "block": "block"}
    structural = recognize(table, roles, cls=STRUCTURAL)["fired"]
    both = recognize(table, roles)["fired"]
    assert structural and all(f["cls"] == STRUCTURAL for f in structural)
    assert {f["id"] for f in structural} <= {f["id"] for f in both}


def test_fallback_message_is_filled():
    fired = _computed(_grouped({k: np.array(v, float) for k, v in BEETLES.items()}),
                      value="y", group="g")
    assert "{" not in fired["B4.2"]["message"]["message"]


# --- Чистые данные: молчат --------------------------------------------------------------

def test_book_designs_without_problems_stay_silent():
    calcium = {"ca": [g for g, v in CALCIUM.items() for _ in v],
               "root": [x for v in CALCIUM.values() for x in v]}
    assert _computed(calcium, value="root", group="ca") == {}
    assert _computed(_potato(), value="yield", group="treatment", block="block") == {}


# --- B4. Допущения -----------------------------------------------------------------------

def test_beetle_eggs_variance_grows_with_mean_table_6_1():
    """Книга, Example 6.1: дисперсия растёт со средним → log; не тупик «взвешенный»."""
    fired = _computed(_grouped({k: np.array(v, float) for k, v in BEETLES.items()}),
                      value="y", group="g")
    assert _params(fired, "B4.2")["x2"] > _params(fired, "B4.2")["crit"] == 7.81
    assert _params(fired, "B4.3")["r"] >= 0.8
    assert "B4.5" not in fired


def test_heterogeneous_without_mean_trend_is_dead_end():
    groups = {"a": 10 + NORMAL10, "b": 20 + 5 * NORMAL10, "c": 30 + NORMAL10}
    fired = _computed(_grouped(groups), value="y", group="g")
    assert {"B4.2", "B4.5"} <= set(fired) and "B4.3" not in fired


def test_heterogeneous_without_mean_trend_points_to_welch_anova():
    """Welham, стр. 107: «взвешенный ANOVA или Уэлч» → лист B4b.welch, а не тупик."""
    groups = {"a": 10 + NORMAL10, "b": 20 + 5 * NORMAL10, "c": 30 + NORMAL10}
    fired = _computed(_grouped(groups), value="y", group="g")
    assert fired["B4.5"]["suggests"] == "welch_anova"


def test_nonnormal_homogeneous_suggests_permutation():
    groups = {k: off + SKEWED10 for k, off in (("a", 0), ("b", 1), ("c", 2))}
    fired = _computed(_grouped(groups), value="y", group="g")
    assert {"B4.1", "B4.4", "B4.6"} <= set(fired)
    assert "B4.2" not in fired and "B4.5" not in fired


def test_two_groups_point_to_mann_whitney_and_welch():
    skewed = _computed(_grouped({"a": SKEWED10, "b": 1 + SKEWED10}), value="y", group="g")
    assert _params(skewed, "B4.1")["method"] == "mann_whitney"
    assert "B4.6" not in skewed  # перестановочный ANOVA — только для 3+ групп
    spread = _computed(_grouped({"a": NORMAL10, "b": 6 * NORMAL10}), value="y", group="g")
    assert _params(spread, "B4.2")["variant"] == "welch"
    assert _params(spread, "B4.2")["method"] == "two_sample_t"
    assert _params(spread, "B4.2")["sd_ratio"] == pytest.approx(6.0, rel=0.01)


def test_two_groups_variance_rule_uses_sd_ratio_and_balance():
    """Две группы: правило welch_rule вместо Бартлетта (общее с ядром)."""
    mild = _computed(_grouped({"a": NORMAL10, "b": 1.8 * NORMAL10}), value="y", group="g")
    assert "B4.2" not in mild  # n равны, SD < 2 раз — Стьюдент уместен
    small_wide = _computed(_grouped({"a": 1.8 * NORMAL10,
                                     "b": np.concatenate([NORMAL10, NORMAL10])}),
                           value="y", group="g")
    assert _params(small_wide, "B4.2")["variant"] == "welch_liberal"
    assert "{" not in small_wide["B4.2"]["message"]["message"]


NORMAL6 = stats.norm.ppf(np.linspace(0.1, 0.9, 6))
_CELLS = [(a, b) for a in ("a1", "a2") for b in ("b1", "b2", "b3")]


def _factorial(values: dict[tuple[str, str], np.ndarray]) -> dict[str, list]:
    """Колонки A, B, y из значений по ячейкам факториала 2 × 3."""
    return {"A": [a for (a, _), v in values.items() for _ in v],
            "B": [b for (_, b), v in values.items() for _ in v],
            "y": [round(float(x), 4) for v in values.values() for x in v]}


def test_factorial_cells_not_pooled_over_second_factor():
    """Группа — ячейка A × B, а не уровень A.

    Синтетика: эффект B есть только при a2 (взаимодействие), шум в ячейках
    одинаков. Раньше группы брались по A: у a2 разброс включал эффект B, SD
    различались в 8 раз, и B4.2 советовало t-тест Уэлча для двух «групп».
    """
    shift = {"b1": 0.0, "b2": 10.0, "b3": 20.0}
    values = {(a, b): 50 + (shift[b] if a == "a2" else 0.0) + np.roll(NORMAL6, i)
              for i, (a, b) in enumerate(_CELLS)}
    fired = _computed(_factorial(values), value="y", factor_a="A", factor_b="B")
    assert not {"B4.2", "B4.3", "B4.5"} & set(fired)


def test_factorial_cells_with_sd_growing_with_mean():
    """SD ячеек пропорционально среднему (5 … 160): Бартлетт по 6 ячейкам и B4.3 → log.

    Раньше при двух уровнях A групп было две: Бартлетт заменялся правилом
    Уэлча, а B4.3 (нужно ≥ 3 групп) молчало.
    """
    means = dict(zip(_CELLS, (5, 10, 20, 40, 80, 160)))
    values = {cell: m * (1 + 0.2 * np.roll(NORMAL6, i))
              for i, (cell, m) in enumerate(means.items())}
    fired = _computed(_factorial(values), value="y", factor_a="A", factor_b="B")
    assert _params(fired, "B4.2")["crit"] == 11.07  # χ²₀.₉₅(5): шесть ячеек
    assert "variant" not in _params(fired, "B4.2")
    assert _params(fired, "B4.3")["r"] >= 0.8 and "B4.5" not in fired


def _blocked(noise_scale: dict[str, float]) -> dict[str, list]:
    """RCBD 4 варианта × 6 блоков с сильным эффектом блока (шаг 10) [синтетика]."""
    data: dict[str, list] = {"blk": [], "trt": [], "y": []}
    for i, trt in enumerate(("t1", "t2", "t3", "t4")):
        noise = np.roll(NORMAL6, i) * noise_scale.get(trt, 1.0)
        for b in range(6):
            for key, value in zip(data, (f"k{b + 1}", trt, round(30 + 3 * i + 10 * b + noise[b], 4))):
                data[key].append(value)
    return data


def test_blocked_design_uses_residual_variances():
    """С блоком однородность проверяется по остаткам модели «вариант + блок» [предл.].

    Сырые SD вариантов здесь ≈ 19 у всех (различия блоков), и вариант с шумом
    в 8 раз больше остальных был не виден; по остаткам Бартлетт его находит.
    """
    roles = dict(value="y", group="trt", block="blk")
    assert "B4.2" not in _computed(_blocked({}), **roles)
    fired = _computed(_blocked({"t3": 8.0}), **roles)
    assert _params(fired, "B4.2")["x2"] > _params(fired, "B4.2")["crit"] == 7.81
    assert "B4.5" in fired and "B4.3" not in fired


def test_block_without_factor_is_not_an_error():
    """Регрессия в блоках (роли value, x, block): фактора нет — групп для Бартлетта нет.

    После патча 103 B4.2 и B4.3 падали здесь с ValueError (запись в errors):
    дисперсии по остаткам считались без проверки, что фактор задан.
    """
    from tests.book_ch15_data import FORAGE

    data = {"block": [f"B{r[0]}" for r in FORAGE], "nitrogen": [r[2] for r in FORAGE],
            "yield": [r[3] for r in FORAGE]}
    out = recognize(_table(data), {"value": "yield", "x": "nitrogen", "block": "block"},
                    cls=COMPUTED)
    assert out["errors"] == []
    assert not {"B4.2", "B4.3", "B4.5"} & {f["id"] for f in out["fired"]}


def test_bartlett_statistic_equals_scipy_without_blocks():
    """Без блоков — тест книги для CRD (стр. 105–107): X² как у scipy.stats.bartlett."""
    groups = {k: np.array(v, float) for k, v in BEETLES.items()}
    fired = _computed(_grouped(groups), value="y", group="g")
    expected = stats.bartlett(*groups.values()).statistic
    assert _params(fired, "B4.2")["x2"] == pytest.approx(expected, abs=0.005)


# --- A. Регрессия -------------------------------------------------------------------------

def test_lack_of_fit_matches_table_12_7():
    """Книга, стр. 312: F неадекватности 4.321 на (4, 18) df, p = 0.013 (log10 счётов)."""
    x, counts = np.array(TRANSECT, float).T
    fired = _computed({"dist": list(x), "lc": list(np.log10(counts))}, x="dist", y="lc")
    assert _params(fired, "A2.2")["f"] == pytest.approx(4.32, abs=0.01)
    assert _params(fired, "A2.2")["p"] == pytest.approx(0.013, abs=5e-4)


def test_straight_line_has_no_curve_signals():
    x = np.repeat(np.arange(1.0, 7.0), 3)
    y = 2 + 0.5 * x + np.tile([-0.1, 0.0, 0.1], 6)
    fired = _computed({"x": list(x), "y": list(y)}, x="x", y="y")
    assert not {"A2.2", "A2.3", "A2.4"} & set(fired)


def test_monthly_temperature_is_periodic():
    """Гл. 17: температура по месяцам — волна, не прямая."""
    fired = _computed({"month": list(range(1, 13)), "t": TEMPERATURE}, x="month", y="t")
    assert {"A2.3", "A2.5", "DIAG.2"} <= set(fired)
    renamed = _computed({"dose": list(range(1, 13)), "t": TEMPERATURE}, x="dose", y="t")
    assert "A2.5" not in renamed  # без признака времени периодичность не предлагаем


def test_spread_growing_with_x():
    x = np.repeat(np.arange(1.0, 11.0), 2)
    y = 5 * x + x * np.tile([-1.0, 1.0], 10)
    assert "A2.4" in _computed({"x": list(x), "y": list(y)}, x="x", y="y")


def test_collinear_predictors():
    x1 = np.arange(1.0, 13.0)
    data = {"y": list(3 * x1 + np.r_[NORMAL10, 0, 0]), "x1": list(x1),
            "x2": list(2 * x1 + 0.01 * np.tile([1, -1], 6)), "x3": list(np.tile(NORMAL10[:6], 2))}
    fired = _computed(data, y="y")
    assert _params(fired, "A1.4")["v"] > 10


# --- G, DIAG ---------------------------------------------------------------------------------

def test_aphid_counts_overdispersed():
    data = {"field": [f"F{r[0]}" for r in APHIDS], "y": [r[2] for r in APHIDS]}
    fired = _computed(data, value="y", group="field")
    p = _params(fired, "G.1")
    assert p["d"] > p["crit"]


def test_overdispersion_not_judged_when_trials_are_only_guessed():
    """Успехи из разного m (Table 18.1, стр. 481) без слота «число испытаний».

    Пуассон по числу успехов принял бы разницу m (60 и 100) за сверхдисперсию:
    девианс 68.3 при χ²₀.₉₅(18) = 28.9. Биномиальная модель с m — 12.5, её нет.
    Пока m только угадано (R2.12), G.1 не считается; со слотом — биномиальная.
    """
    from tests.book_ch18_data import DEMETHYLATION

    data = {"dose": [f"d{r[0]}" for r in DEMETHYLATION], "y": [r[1] for r in DEMETHYLATION],
            "m": [r[2] for r in DEMETHYLATION]}
    assert "G.1" not in _computed(data, value="y", group="dose")
    assert "G.1" not in _computed(data, value="y", group="dose", trials="m")
    del data["m"]  # без колонки m это просто счёты: Пуассон и сверхдисперсия
    assert "G.1" in _computed(data, value="y", group="dose")


def test_non_count_response_has_no_glm_signal():
    assert "G.1" not in _computed(_potato(), value="yield", group="treatment")


def test_planted_outlier_and_influence():
    x = np.arange(1.0, 21.0)
    y = 1 + 2 * x + np.tile(MIXED10, 2) * 0.5
    assert _computed({"x": list(x), "y": list(y)}, x="x", y="y") == {}
    y[9] += 8
    assert "10" in _params(_computed({"x": list(x), "y": list(y)}, x="x", y="y"),
                           "DIAG.1")["rows"]


def test_elisa_far_dose_is_influential():
    fired = _computed({"conc": ELISA_CONC, "abs": ELISA_ABS}, x="conc", y="abs")
    assert _params(fired, "DIAG.4")["rows"] == "8"
