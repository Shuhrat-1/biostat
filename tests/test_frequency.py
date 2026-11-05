"""frequency_table — Witte & Witte, Statistics (11th ed., 2017), гл. 2–4.

Book-тесты: числа — со страниц PDF (страница PDF = печатная + 18), ответы
Progress Checks — Appendix B, стр. 420–422.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.report.schema import Severity
from core.stats.frequency import convenient_width, frequency_table, infer_unit
from tests import witte_data as W


def _col(res, key):
    return [r[key] for r in res.parameters["table"]]


# --- Book-тесты ---------------------------------------------------------------------


def test_weights_grouped_table_2_2_2_5_2_6():
    """Witte 2.1–2.5, стр. 24, 29, 30: Tables 2.2, 2.5, 2.6 из 53 весов Table 1.1.

    Интервал 112/10 = 11.2 → 10 (стр. 27), классы 130–139 … 240–249; f, доли, накопленные
    f и проценты совпадают с книгой (книга печатает их сверху вниз, здесь — снизу вверх).
    """
    res = frequency_table([str(v) for v in W.WEIGHTS])
    p = res.parameters
    assert p["kind"] == "grouped" and p["class_width"] == 10 and p["unit"] == 1
    assert _col(res, "class")[0] == "130–139" and _col(res, "class")[-1] == "240–249"
    assert _col(res, "f") == [3, 1, 17, 12, 7, 3, 4, 2, 0, 3, 0, 1]
    assert [round(v, 2) for v in _col(res, "proportion")] == [
        .06, .02, .32, .23, .13, .06, .08, .04, .00, .06, .00, .02]
    assert _col(res, "cumulative_f") == [3, 4, 21, 33, 40, 43, 47, 49, 49, 52, 52, 53]
    assert [round(v) for v in _col(res, "cumulative_percent")] == [
        6, 8, 40, 62, 75, 81, 89, 92, 92, 98, 98, 100]


def test_real_limits_and_midpoint():
    """Witte 2.2, стр. 26: реальные границы 140–149 — 139.5 и 149.5; стр. 34: середина 160–169 = 164.5."""
    rows = frequency_table(W.WEIGHTS).parameters["table"]
    assert (rows[1]["real_lower"], rows[1]["real_upper"]) == (139.5, 149.5)
    assert rows[3]["midpoint"] == 164.5


def test_dropout_iq_progress_check_2_2():
    """Witte Progress Check 2.2, стр. 26; ответ стр. 420–421: интервал 54/10 = 5.4 → 5, классы
    65–69 … 120–124 с f = 1, 1, 3, 3, 4, 7, 6, 4, 3, 2, 0, 1; реальные границы нижнего 64.5–69.5."""
    res = frequency_table(W.DROPOUT_IQ)
    assert res.parameters["class_width"] == 5
    assert _col(res, "class")[0] == "65–69" and _col(res, "class")[-1] == "120–124"
    assert _col(res, "f") == [1, 1, 3, 3, 4, 7, 6, 4, 3, 2, 0, 1]
    first = res.parameters["table"][0]
    assert (first["real_lower"], first["real_upper"]) == (64.5, 69.5)


def test_ratings_ungrouped_progress_check_2_1():
    """Witte Progress Check 2.1, стр. 23; ответ стр. 420: 10 возможных значений → без группировки;
    f для 1…10 = 1, 2, 6, 1, 2, 2, 5, 3, 2, 1."""
    res = frequency_table(W.RATINGS)
    assert res.parameters["kind"] == "ungrouped"
    assert _col(res, "class") == [str(k) for k in range(1, 11)]
    assert _col(res, "f") == [1, 2, 6, 1, 2, 2, 5, 3, 2, 1]


def test_presidential_terms_mode_table_3_1():
    """Witte 3.1, стр. 48: мода сроков — 4 года (частота 7), а не 7; несгруппированная таблица
    со всеми значениями 2…12, включая пустые (правило 2, стр. 25)."""
    res = frequency_table(W.TERMS)
    assert res.parameters["mode"] == ["4"] and res.parameters["mode_f"] == 7
    assert _col(res, "class") == [str(k) for k in range(2, 13)]
    assert _col(res, "f")[5] == 0  # 7 лет — пустой класс


def test_infant_deaths_mode_table_3_4():
    """Witte 3.4, стр. 53: модальная младенческая смертность — 4 (пять раз)."""
    res = frequency_table(W.INFANT_DEATHS, kind="categorical")
    assert res.parameters["mode"] == ["4"] and res.parameters["mode_f"] == 5


def test_facebook_nominal_table_2_7():
    """Witte 2.6, стр. 31: Table 2.7 из Table 1.2 — Yes 56, No 27; номинальные данные — без
    накопленных частот (стр. 32), мода Yes (стр. 55)."""
    res = frequency_table(W.FACEBOOK)
    assert dict(zip(_col(res, "class"), _col(res, "f"))) == {"Y": 56, "N": 27}
    assert "cumulative_f" not in res.parameters["table"][0]
    assert res.parameters["mode"] == ["Y"] and "median_class" not in res.parameters
    assert "freq_nominal" in {w.code for w in res.warnings}


def test_officer_ranks_table_2_8_and_median_table_3_5():
    """Witte 2.6, стр. 32: Table 2.8 — доли .004, .167, .204, .370, .255; накопленные проценты
    25.5, 62.5, 82.9, 99.6, 100.0 (снизу вверх); 3.5, стр. 55: медианный класс — Captain."""
    res = frequency_table(W.RANKS, count=W.RANK_COUNTS, level_order=W.RANK_ORDER)
    assert _col(res, "class") == ["Lieutenant", "Captain", "Major", "Colonel", "General"]
    assert [round(v, 3) for v in _col(res, "proportion")] == [.255, .370, .204, .167, .004]
    assert [round(v, 1) for v in _col(res, "cumulative_percent")] == [25.5, 62.5, 82.9, 99.6, 100]
    assert res.parameters["median_class"] == "Captain"
    assert res.statistics["n"] == 78827


def test_movie_ratings_progress_check_2_8():
    """Witte Progress Check 2.8, стр. 32; ответ стр. 422: f = G 3, PG 8, PG-13 3, R 4, NC-17 2;
    проценты 15, 40, 15, 20, 10; накопленные f 3, 11, 14, 18, 20; процентильный ранг PG = 55."""
    res = frequency_table(W.MOVIES, level_order=W.MOVIE_ORDER)
    assert _col(res, "f") == [3, 8, 3, 4, 2]
    assert _col(res, "percent") == pytest.approx([15, 40, 15, 20, 10])
    assert _col(res, "cumulative_f") == [3, 11, 14, 18, 20]
    assert res.parameters["table"][1]["cumulative_percent"] == pytest.approx(55)
    assert res.parameters["median_class"] == "PG"
    assert (res.parameters["lowest_class"], res.parameters["highest_class"]) == ("G", "NC-17")


def test_tenth_of_unit_classes():
    """Witte 2.2, стр. 24–26: при весах до 0.1 фунта классы 130.0–139.9, реальные границы
    139.95–149.95, ширина по-прежнему 10."""
    values = [f"{v:.1f}" for v in np.linspace(130.0, 245.0, 60)]
    res = frequency_table(values)
    assert res.parameters["unit"] == pytest.approx(0.1)
    rows = res.parameters["table"]
    assert rows[0]["class"] == "130.0–139.9"
    assert (rows[1]["real_lower"], rows[1]["real_upper"]) == (139.95, 149.95)


# --- Инварианты и эталоны -------------------------------------------------------------


def test_every_observation_in_exactly_one_class():
    """Правило 1 (стр. 25): сумма f = n, сумма долей = 1, накопленный процент кончается на 100."""
    rng = np.random.default_rng(1)
    x = np.round(rng.lognormal(3, 0.6, 400), 1)
    res = frequency_table(x.tolist())
    assert sum(_col(res, "f")) == 400
    assert sum(_col(res, "proportion")) == pytest.approx(1)
    assert _col(res, "cumulative_percent")[-1] == pytest.approx(100)
    widths = np.diff([r["real_lower"] for r in res.parameters["table"]])
    assert np.allclose(widths, res.parameters["class_width"])


def test_wilson_interval_matches_formula():
    """ДИ доли — Уилсон [предл.: Wilson 1927]: сверка с формулой (p + z²/2n ± z√(p(1−p)/n + z²/4n²))/(1 + z²/n)."""
    res = frequency_table(W.FACEBOOK)
    row = res.parameters["table"][0]
    n, p, z = 83, 56 / 83, 1.959963984540054
    centre, half = p + z ** 2 / (2 * n), z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    assert row["ci_lower"] == pytest.approx((centre - half) / (1 + z ** 2 / n))
    assert row["ci_upper"] == pytest.approx((centre + half) / (1 + z ** 2 / n))


def test_count_column_equals_expanded_rows():
    """Колонка частот эквивалентна повторению строк."""
    expanded = [r for r, c in zip(W.RANKS, [3, 5, 2, 7, 4]) for _ in range(c)]
    a = frequency_table(W.RANKS, count=[3, 5, 2, 7, 4], level_order=W.RANK_ORDER)
    b = frequency_table(expanded, level_order=W.RANK_ORDER)
    assert a.parameters["table"] == b.parameters["table"]


def test_bimodal_mode_lists_all_peaks():
    """Witte 3.1, стр. 48: мод может быть несколько — выдаются все."""
    res = frequency_table(["a", "a", "b", "b", "c"])
    assert res.parameters["mode"] == ["a", "b"]


@pytest.mark.parametrize("raw, nice", [(11.2, 10), (5.4, 5), (0.73, 0.5), (1.6, 2), (34, 20), (40, 50)])
def test_convenient_width(raw, nice):
    """Удобный интервал 1, 2, 5 × 10ᵏ, ближайший к размах/10 (стр. 25, 27)."""
    assert convenient_width(raw) == pytest.approx(nice)


def test_infer_unit_from_strings_keeps_trailing_zeros():
    """Единица измерения по записи: «150.0» — до десятых (стр. 24)."""
    assert infer_unit(["150.0", "151.5"]) == pytest.approx(0.1)
    assert infer_unit([150, 151]) == 1


def test_errors_and_missing():
    """Неизвестный уровень вне порядка, нечисловая колонка при kind=numeric, пропуски."""
    bad = frequency_table(["low", "high", "mid"], level_order="low; high")
    assert any(w.code == "freq_unknown_levels" and w.severity == Severity.ERROR
               for w in bad.warnings)
    assert frequency_table(["a", "1"], kind="numeric").warnings[-1].code == "freq_not_numeric"
    res = frequency_table(["1", "", "2", "2"])
    assert res.n == 3 and "missing_dropped" in {w.code for w in res.warnings}
