"""Глава 19: планирование вложенной выборки по компонентам дисперсии (Case Study 19.3)."""

from __future__ import annotations

import pytest

from core.stats.power import power_t
from core.stats.sampling import sampling_scheme_power

_COMPONENTS = "Rep=0.0050; Unit=0.0046; Branch=0.0031; Pod=0.0017"


def _run(schemes, **kw):
    return sampling_scheme_power(_COMPONENTS, schemes, delta=0.16, **kw).parameters["schemes"]


def test_scheme_one_table_19_4():
    """Книга: Table 19.4b и стр. 534–535: 60 линий × 3 × 2 ветви × 2 стручка — s²_U 0.0263
    (118 df), SE 0.0468, SED 0.0662, мощность 66.9 %; страты ветвей 0.0079 (179), стручков
    0.0017 (359)."""
    s = _run("60 3 2 2")[0]
    assert (s["s2_unit"], s["df_unit"]) == (pytest.approx(0.0263), 118)
    assert (s["se_mean"], s["sed"]) == pytest.approx((0.0468, 0.0662), abs=5e-5)
    assert s["power"] == pytest.approx(0.669, abs=5e-4)
    strata = {r["stratum"]: r for r in s["strata"]}
    assert (strata["Branch"]["variance"], strata["Branch"]["df"]) == (pytest.approx(0.0079), 179)
    assert (strata["Pod"]["variance"], strata["Pod"]["df"]) == (pytest.approx(0.0017), 359)
    assert sum(r["df"] + r.get("treatment_df", 0) + r.get("position_df", 0)
               for r in s["strata"]) == 719


def test_block_stratum_follows_table_formula():
    """Table 19.4a: дисперсия страты блоков n_L n_B n_P σ²_R + s²_U = 1.2263 (в 19.4b
    напечатано 0.6263 — расхождение книги)."""
    assert _run("60 3 2 2")[0]["strata"][0]["variance"] == pytest.approx(1.2263)


def test_alternative_schemes_by_book_formula():
    """Стр. 534–535: по формуле книги схема 2 (n_B = 2, n_P = 1) даёт s²_U = 0.0140, схема 3
    (n_B = 1, n_P = 2) — 0.0171 (в книге 0.0186 и 0.0355 — не по своей формуле)."""
    two, three = _run("110 3 2 1; 110 3 1 2")
    assert (two["s2_unit"], two["df_unit"]) == (pytest.approx(0.0140), 218)
    assert three["s2_unit"] == pytest.approx(0.0171)
    assert two["power"] > three["power"]


def test_book_powers_from_book_variances():
    """Мощности книги 52.5 % и 31.0 % воспроизводятся из её s²_U 0.0186 и 0.0355 (218 df)."""
    for s2, expected in ((0.0186, 0.525), (0.0355, 0.310)):
        sed = (2 * s2 / 6) ** 0.5
        assert power_t(0.16, sed, 218) == pytest.approx(expected, abs=5e-4)


def test_guards_and_low_df_warning():
    """Мало блоков и вариантов — предупреждение о df; неверный ввод — ошибка."""
    res = sampling_scheme_power(_COMPONENTS, "4 2 2 2", delta=0.16)
    assert any(w.code == "sampling_low_df" for w in res.warnings)
    with pytest.raises(ValueError):
        sampling_scheme_power(_COMPONENTS, "60 3 2", delta=0.16)
    with pytest.raises(ValueError):
        sampling_scheme_power("Rep=0.005", "60 3", delta=0.16)
