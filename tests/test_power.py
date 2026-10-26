"""Тесты гл. 10: повторность, мощность, эквивалентность.

Эталоны — Tables 10.1, 10.3, 10.4, 10.6, 10.7 и Examples 10.1A–C, 10.2, 10.3
(сверены с картинками страниц PDF 265, 271, 274, 277, 278).
"""

from __future__ import annotations

import json

import pytest

from core.stats.power import (
    equivalence_test,
    power_t,
    replication_power,
    replication_table,
    sigma_from_range,
    split_plot_power,
)
from tests.test_treatment_means import CALCIUM

# Table 10.5 (стр. 255): (сито, навеска, время встряхивания) → C, мг/кг.
BIOMASS = {
    ("L", "20", "30"): [971, 858, 984, 900], ("L", "20", "60"): [1062, 1028, 1020, 1106],
    ("L", "50", "30"): [956, 1083, 764, 836], ("L", "50", "60"): [1030, 1014, 981, 1065],
    ("L", "200", "30"): [951, 878, 882, 918], ("L", "200", "60"): [974, 1097, 996, 1048],
    ("S", "20", "30"): [965, 1068, 922, 968], ("S", "20", "60"): [1115, 1123, 1167, 1181],
    ("S", "50", "30"): [995, 1177, 951, 1118], ("S", "50", "60"): [1050, 1196, 1116, 1102],
    ("S", "200", "30"): [904, 983, 959, 926], ("S", "200", "60"): [1050, 1016, 1144, 1172],
}

# Table 10.4 (стр. 252): (t_B, m) → (SED, df, мощность) внутри сорта и для главного эффекта.
TABLE_10_4 = {
    (4, 6): (0.30, 48.89, 0.736, 0.17, 10, 0.987), (4, 5): (0.33, 39.11, 0.653, 0.19, 8, 0.961),
    (4, 4): (0.37, 29.33, 0.550, 0.21, 6, 0.887), (4, 3): (0.43, 19.56, 0.427, 0.24, 4, 0.701),
    (4, 2): (0.52, 9.78, 0.281, 0.30, 2, 0.335), (5, 4): (0.37, 36.92, 0.556, 0.19, 6, 0.927),
    (5, 3): (0.43, 24.61, 0.435, 0.22, 4, 0.762), (6, 4): (0.37, 44.35, 0.560, 0.18, 6, 0.951),
    (6, 3): (0.43, 29.57, 0.440, 0.21, 4, 0.807), (6, 2): (0.52, 14.78, 0.297, 0.26, 2, 0.405),
    (8, 3): (0.43, 39.18, 0.445, 0.19, 4, 0.864), (10, 2): (0.52, 24.20, 0.311, 0.22, 2, 0.491),
}


def _calcium() -> tuple[list[float], list[str]]:
    y, g = [], []
    for level, values in CALCIUM.items():
        y += list(values)
        g += [level] * len(values)
    return y, g


def _biomass() -> tuple[list[float], list[str], list[str], list[str]]:
    y, size, weight, time = [], [], [], []
    for (s, w, t), values in BIOMASS.items():
        for v in values:
            y.append(float(v))
            size.append(s)
            weight.append(w)
            time.append(t)
    return y, size, weight, time


# --- Повторность и мощность: Tables 10.1, 10.3 --------------------------------


def test_replication_table_10_1_and_10_3() -> None:
    rows = {r["n"]: r for r in replication_table(75, 4, 10, n_max=13)}
    assert (rows[2]["df"], rows[2]["t_crit"]) == (4, pytest.approx(2.776, abs=5e-4))
    assert (rows[2]["sed"], rows[2]["lsd"]) == pytest.approx((8.66, 24.04), abs=5e-3)
    assert (rows[6]["lsd"], rows[7]["lsd"]) == pytest.approx((10.43, 9.55), abs=5e-3)
    expected = {7: 0.545, 8: 0.606, 9: 0.661, 10: 0.710, 11: 0.753, 12: 0.790, 13: 0.822}
    for n, power in expected.items():
        assert rows[n]["power"] == pytest.approx(power, abs=5e-4)
    assert rows[7]["noncentrality"] == pytest.approx(2.16, abs=5e-3)


def test_lsd_range_example_10_1a() -> None:
    lo = replication_table(50, 4, 10, n_max=7)[-1]["lsd"]
    hi = replication_table(100, 4, 10, n_max=7)[-1]["lsd"]
    assert (lo, hi) == pytest.approx((7.80, 11.03), abs=1e-2)  # книга: 7.79 и 11.02 (SED округлён)


def test_power_one_sided_and_designs() -> None:
    assert power_t(10, 4.63, 24, alternative="greater") > power_t(10, 4.63, 24)
    assert power_t(-10, 4.63, 24, alternative="less") == pytest.approx(
        power_t(10, 4.63, 24, alternative="greater"))
    rcbd = {r["n"]: r["df"] for r in replication_table(75, 4, 10, design="rcbd", n_max=5)}
    assert rcbd == {2: 3, 3: 6, 4: 9, 5: 12}
    ls = replication_table(75, 5, 10, design="latin_square")
    assert [(r["n"], r["df"]) for r in ls] == [(5, 12)]


def test_sigma_from_range() -> None:
    assert sigma_from_range(10, 30) == 5


def test_replication_power_from_pilot_example_10_1() -> None:
    """Пилот — calcium (Table 4.1); s² = 75 как в Example 10.1A: n = 7 по LSD, 13 по мощности."""
    y, g = _calcium()
    result = replication_power(y, g, delta=10, s2_override=75, n_max=15)
    assert result.statistics["n_required_lsd"] == 7
    assert result.statistics["n_required_power"] == 13
    assert result.parameters["design"] == "crd"
    pilot = replication_power(y, g, delta=10)
    assert pilot.statistics["s2_planning"] == pytest.approx(76.35, abs=5e-3)
    assert pilot.statistics["cv_pct"] == pytest.approx(13.8, abs=0.05)  # книга: «14%»


def test_lsd_fraction_of_mean_example_10_1b() -> None:
    """%CV 14 и 20 при n = 7: LSD = 0.15ȳ и 0.22ȳ."""
    for cv, frac in ((14, 0.15), (20, 0.22)):
        s2 = (cv / 100) ** 2  # при ȳ = 1
        row = replication_table(s2, 4, 0.1, n_max=7)[-1]
        assert row["lsd"] == pytest.approx(frac, abs=5e-3)


def test_replication_power_invalid_delta() -> None:
    y, g = _calcium()
    assert replication_power(y, g, delta=0).warnings[-1].code == "power_invalid"


# --- Split-plot: Example 10.2, Table 10.4 -------------------------------------


@pytest.mark.parametrize(("key", "row"), list(TABLE_10_4.items()))
def test_split_plot_power_table_10_4(key, row) -> None:
    t_b, m = key
    res = split_plot_power(0.25, (0.1 * t_b + 1) * 0.25, 3, t_b, m, -0.8)
    within, main = res["within_b"], res["main_a"]
    assert (within["sed"], within["df"]) == pytest.approx(row[:2], abs=5e-3)
    assert within["power"] == pytest.approx(row[2], abs=5e-4)
    assert (main["sed"], main["df"]) == pytest.approx(row[3:5], abs=5e-3)
    assert main["power"] == pytest.approx(row[5], abs=5e-4)


def test_split_plot_power_book_rounding_rows() -> None:
    """Строки (5, 2) и (7, 3) Table 10.4 расходятся с точным расчётом в 3-м знаке."""
    a = split_plot_power(0.25, 1.5 * 0.25, 3, 5, 2, -0.8)["main_a"]["power"]
    b = split_plot_power(0.25, 1.7 * 0.25, 3, 7, 3, -0.8)["within_b"]["power"]
    assert (a, b) == pytest.approx((0.372, 0.440), abs=4e-3)


# --- Эквивалентность: Tables 10.6, 10.7 ---------------------------------------


@pytest.fixture(scope="module")
def biomass():
    y, size, weight, time = _biomass()
    return equivalence_test(y, size, factor_b=weight, factor_c=time, control="S 200 60",
                            margin=110)


def test_biomass_residual_and_sed_table_10_6(biomass) -> None:
    assert biomass.statistics["ms_resid"] == pytest.approx(4705.15, abs=5e-3)
    assert biomass.statistics["df_resid"] == 36
    assert biomass.parameters["equivalence"][0]["sed"] == pytest.approx(48.5, abs=5e-3)


def test_biomass_tost_intervals_table_10_7(biomass) -> None:
    rows = {r["level"]: r for r in biomass.parameters["equivalence"]}
    expected = {"L × 20 × 30": (-167.25, -249.1, -85.4), "S × 20 × 60": (51.0, -30.9, 132.9),
                "S × 50 × 60": (20.5, -61.4, 102.4), "L × 200 × 60": (-66.75, -148.6, 15.1)}
    for level, (d, lo, hi) in expected.items():
        row = rows[level]
        assert row["difference"] == pytest.approx(d)
        assert (row["ci_lower"], row["ci_upper"]) == pytest.approx((lo, hi), abs=0.06)
    assert [lv for lv, r in rows.items() if r["equivalent"]] == ["S × 50 × 60"]
    assert biomass.statistics["n_inequivalent"] == 0
    assert biomass.parameters["ci_level"] == pytest.approx(0.90)


def test_biomass_tost_p_and_power(biomass) -> None:
    row = next(r for r in biomass.parameters["equivalence"] if r["level"] == "S × 50 × 60")
    assert row["p_tost"] < 0.05
    other = next(r for r in biomass.parameters["equivalence"] if r["level"] == "S × 20 × 60")
    assert other["p_tost"] > 0.05
    assert biomass.statistics["power_difference_c"] == pytest.approx(0.60, abs=5e-3)


def test_equivalence_percent_margin_and_bonferroni() -> None:
    y, size, weight, time = _biomass()
    pct = equivalence_test(y, size, factor_b=weight, factor_c=time, control="S × 200 × 60",
                           margin=10, margin_kind="percent")
    assert pct.parameters["margin"] == pytest.approx(109.55)
    bonf = equivalence_test(y, size, factor_b=weight, factor_c=time, control="S 200 60",
                            margin=110, adjustment="bonferroni")
    assert bonf.statistics["n_equivalent"] == 0  # интервалы шире (McBride, стр. 255)


def test_equivalence_unknown_control() -> None:
    y, size, weight, time = _biomass()
    result = equivalence_test(y, size, factor_b=weight, factor_c=time, control="X", margin=110)
    assert result.warnings[-1].code == "equivalence_invalid"


def test_power_results_serialisable(biomass) -> None:
    y, g = _calcium()
    for result in (biomass, replication_power(y, g, delta=10)):
        json.dumps(result.to_dict(), allow_nan=False)
