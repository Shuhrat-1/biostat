"""Book-тесты Witte & Witte (11th ed., 2017), гл. 18: двухфакторный ANOVA.

factorial_anova — из Welham (гл. 8); Witte добавляет η²_p новым ключом (пакет W2).
Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest

from core.stats.factorial import factorial_anova
from tests import witte_data as W

_NAMES = {"factor_a": "A", "factor_b": "B"}


def _fit(data: tuple[list, list, list], **kwargs):
    values, a, b = data
    return factorial_anova(values, factor_a=a, factor_b=b, column_names=_NAMES, **kwargs)


def _rows(res) -> dict[str, dict]:
    return {r["term"]: r for r in res.parameters["anova_table"]}


def test_smoke_tables_18_3_and_18_6():
    """Witte 18.4–18.5, стр. 350 (Table 18.3): SS_total 352, column 72, row 192, interaction 56,
    within 32; стр. 352 (Table 18.6): MS_within = 5.33, F = 6.75, 36.02, 5.25. Книга делит
    на округлённое 5.33: точно F_row = 36.00 — допуск approx к точному."""
    rows = _rows(_fit(W.SMOKE))
    assert [rows[t]["ss"] for t in ("A", "B", "A.B", "Residual")] == pytest.approx([72, 192, 56, 32])
    assert [rows[t]["df"] for t in ("A", "B", "A.B", "Residual")] == [2, 1, 2, 6]
    assert rows["A"]["F"] == pytest.approx(6.75)
    assert rows["B"]["F"] == pytest.approx(36.0)
    assert rows["A.B"]["F"] == pytest.approx(5.25)
    assert all(rows[t]["p"] < 0.05 for t in ("A", "B", "A.B"))


def test_smoke_partial_eta_formula_18_3():
    """Witte 18.7, стр. 353–354 (формула 18.3): η²_p = 72/104 = .69, 192/224 = .86, 56/88 = .64.
    Сумма η²_p > 1 — это не доли одного целого."""
    res = _fit(W.SMOKE)
    rows = _rows(res)
    assert rows["A"]["eta_squared_partial"] == pytest.approx(72 / 104)
    assert rows["B"]["eta_squared_partial"] == pytest.approx(192 / 224)
    assert rows["A.B"]["eta_squared_partial"] == pytest.approx(56 / 88)
    assert res.statistics["eta_p_A.B"] == rows["A.B"]["eta_squared_partial"]
    assert "eta_squared_partial" not in rows["Residual"]
    assert sum(rows[t]["eta_squared_partial"] for t in ("A", "B", "A.B")) > 1


def test_tv_progress_checks_18_3_18_4():
    """Witte, стр. 353 (Progress Check 18.3), ответы стр. 448–449: SS 144.19 / 0.07 / 0.68 / 23.50,
    F = 16.35, 0.02, 0.08; 18.4: η²_p(время) = 144.19 / (144.19 + 23.50) = .86.
    Точно SS_row = 0.0625, F_column = 16.36 (книга делит округлённые MS)."""
    res = _fit(W.TV)
    rows = _rows(res)
    assert rows["A"]["ss"] == pytest.approx(144.1875)
    assert rows["B"]["ss"] == pytest.approx(0.0625)
    assert rows["Residual"]["ss"] == pytest.approx(23.5)
    assert rows["A"]["F"] == pytest.approx(16.36, abs=5e-3)
    assert round(rows["A"]["eta_squared_partial"], 2) == 0.86


def test_sauna_review_18_10():
    """Witte, стр. 363 (Review 18.10), ответ стр. 451: SS 33.64 / 5.05 / 0.95 / 15.33, df 3, 2, 6, 24,
    F = 17.52, 3.95, 0.25; η²_p(витамин C) = 33.64 / (33.64 + 15.33) = .69.
    Точно F = 17.55 и 3.96 (книга делит округлённые MS)."""
    values = W.SAUNA[0]
    assert sum(v * v for v in values) == 385 and sum(values) == 109
    rows = _rows(_fit(W.SAUNA))
    assert rows["A"]["ss"] == pytest.approx(33.639, abs=1e-3)
    assert rows["B"]["ss"] == pytest.approx(5.056, abs=1e-3)
    assert rows["A.B"]["F"] == pytest.approx(0.246, abs=1e-3)
    assert rows["A"]["F"] == pytest.approx(17.55, abs=5e-3)
    assert round(rows["A"]["eta_squared_partial"], 2) == 0.69


def test_block_row_has_no_partial_eta():
    """С блоком: η²_p только у членов вариантов (strata «unit»), у блока — нет."""
    values, a, b = W.SMOKE
    block = ["I", "II"] * 6
    res = factorial_anova(values, factor_a=a, factor_b=b, block=block, column_names=_NAMES)
    rows = _rows(res)
    assert "eta_squared_partial" not in rows["Block"]
    ss_res = rows["Residual"]["ss"]
    assert rows["A"]["eta_squared_partial"] == pytest.approx(rows["A"]["ss"] / (rows["A"]["ss"] + ss_res))
