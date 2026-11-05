"""describe_by_group — Witte & Witte, Statistics (11th ed., 2017), гл. 3–4.

Book-тесты: числа со страниц PDF (страница PDF = печатная + 18). Книга даёт σ
(делитель N) для Fig. 4.1 и s (делитель n − 1) для Table 4.3; метод, как и
describe_all и Welham (стр. 15), считает s; σ проверяется через SS.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.stats.descriptive import describe_all
from core.stats.group_summary import describe_by_group
from tests import witte_data as W


def _long(groups: dict[str, list]) -> tuple[list[float], list[str]]:
    values = [float(v) for vals in groups.values() for v in vals]
    labels = [k for k, vals in groups.items() for _ in vals]
    return values, labels


def test_fig_4_1_spread_and_ss():
    """Witte 4.2–4.5, стр. 61–63, 69 (Fig. 4.1, Table 4.1): размахи A, B, C = 0, 2, 6;
    дисперсии σ² = SS/N = 0.00, 0.29, 3.14; для C: SS = 22, σ = 1.77; IQR C = 2 (Table 4.6, стр. 77)."""
    res = describe_by_group(*_long({"A": W.DIST_A, "B": W.DIST_B, "C": W.DIST_C}))
    s = res.parameters["summary"]
    assert [s[g]["range"] for g in "ABC"] == [0, 2, 6]
    assert [round(s[g]["ss"] / s[g]["n"], 2) for g in "ABC"] == [0.00, 0.29, 3.14]
    assert s["C"]["ss"] == 22 and round(math.sqrt(22 / 7), 2) == 1.77
    assert s["C"]["iqr"] == 2 and s["C"]["mean"] == 10


def test_table_4_3_sample_sd():
    """Witte 4.5, стр. 72 (Table 4.3): n = 5, ΣX = 15, X̄ = 3, SS = 30, s² = 7.50, s = 2.74."""
    s = describe_by_group(*_long({"T": W.TABLE_4_3, "C": W.DIST_C})).parameters["summary"]["T"]
    assert (s["n"], s["mean"], s["ss"]) == (5, 3, 30)
    assert round(s["variance"], 2) == 7.50 and round(s["std"], 2) == 2.74


def test_terms_and_infant_deaths_averages():
    """Witte 3.1–3.4, стр. 48–53: сроки президентов — мода 4, медиана 4.5, среднее 5.60
    (Table 3.1); младенческая смертность — мода 4, медиана 7, среднее 30.00 (Table 3.4)."""
    res = describe_by_group(*_long({"terms": W.TERMS, "infant": W.INFANT_DEATHS}))
    s, modes = res.parameters["summary"], res.parameters["modes"]
    assert (s["terms"]["median"], round(s["terms"]["mean"], 2)) == (4.5, 5.60)
    assert (s["infant"]["median"], round(s["infant"]["mean"], 2)) == (7, 30.00)
    assert modes["terms"] == {"mode": [4.0], "mode_f": 7.0}
    assert modes["infant"] == {"mode": [4.0], "mode_f": 5.0}


def test_fig_4_2_shift_keeps_spread():
    """Witte 4.1, стр. 61–62 (Fig. 4.2): B* и C* — B и C, сдвинутые на 2: средние 12,
    разброс тот же; разность средних в обоих опытах = 2."""
    groups = {"B": W.DIST_B, "B*": [v + 2 for v in W.DIST_B],
              "C": W.DIST_C, "C*": [v + 2 for v in W.DIST_C]}
    s = describe_by_group(*_long(groups)).parameters["summary"]
    assert s["B*"]["mean"] - s["B"]["mean"] == s["C*"]["mean"] - s["C"]["mean"] == 2
    assert s["B*"]["std"] == pytest.approx(s["B"]["std"])
    assert s["C*"]["std"] == pytest.approx(s["C"]["std"]) and s["C"]["std"] > s["B"]["std"]


def test_weights_mean_and_sd():
    """Witte 4.4, стр. 64 (Table 1.1): X̄ = 169.51, s = 23.33. Книга печатает s² = 544.29,
    точно 544.10 (похоже, 23.33², возведённое из округлённого s) — расхождение 0.19 (0.03 %)."""
    s = describe_by_group(*_long({"male": W.WEIGHTS, "x": [1, 2]})).parameters["summary"]["male"]
    assert round(s["mean"], 2) == 169.51 and round(s["std"], 2) == 23.33
    assert s["variance"] == pytest.approx(544.29, abs=0.2)


def test_groups_and_total_match_describe_all():
    """Метрики группы и строки «Итого» совпадают с describe_all на тех же значениях."""
    rng = np.random.default_rng(4)
    groups = {"a": rng.normal(10, 2, 15).tolist(), "b": rng.normal(12, 3, 9).tolist()}
    res = describe_by_group(*_long(groups))
    ref = describe_all({"a": groups["a"], "b": groups["b"],
                        "all": groups["a"] + groups["b"]}).parameters["summary"]
    s = res.parameters["summary"]
    for key, other in (("a", "a"), ("b", "b"), ("Итого", "all")):
        for metric, value in ref[other].items():
            if metric != "missing":
                assert s[key][metric] == pytest.approx(value, nan_ok=True), (key, metric)
    assert res.parameters["columns"] == ["a", "b", "Итого"]


def test_mode_only_for_discrete_or_repeating_values():
    """Мода — только для дискретных или повторяющихся значений; все различны — моды нет."""
    rng = np.random.default_rng(2)
    res = describe_by_group(*_long({"cont": rng.normal(5, 1, 20).round(3).tolist(),
                                    "uniq": [1, 2, 3, 4], "rep": [1.5, 1.5, 2.5, 2.5, 2.5, 3.5]}))
    modes = res.parameters["modes"]
    assert "cont" not in modes and "uniq" not in modes
    assert modes["rep"] == {"mode": [2.5], "mode_f": 3.0}


def test_natural_group_order_missing_and_small_groups():
    """Числовые метки групп — по значению (5 < 10); пропуски удаляются; n < 2 — предупреждение."""
    res = describe_by_group([1, 2, 3, float("nan"), 5], ["10", "10", "5", "5", ""])
    assert res.parameters["groups"] == ["5", "10"]
    codes = {w.code for w in res.warnings}
    assert {"missing_dropped", "group_n_too_small"} <= codes
    assert res.n == 3
