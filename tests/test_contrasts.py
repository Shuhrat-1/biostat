"""Тесты контрастов (гл. 8.6–8.7): Tables 8.17, 8.19, 8.21, 8.22, 8.24, 8.25.

Числа сверены с картинками страниц 185–189, 194 (PDF 207, 208, 211, 216).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.report.schema import Severity
from core.stats.contrasts import (
    orthogonal_polynomials,
    parse_contrasts,
    polynomial_contrasts,
    treatment_contrasts,
)
from core.stats.factorial import factorial_anova
from tests.test_factorial import _potato

# Table 8.20 (стр. 188): Population, Herbicide, сырая масса по блокам 1–5.
HERBICIDE = """P1 A 5.94 3.63 5.56 4.09 3.65
P2 A 3.88 2.17 0.63 2.82 1.73
P3 A 3.55 5.16 5.17 1.07 2.61
P4 A 6.45 5.56 1.99 5.21 2.51
P5 A 0.10 0.31 3.69 4.56 0.16
P6 A 4.94 5.21 2.51 3.76 1.90
P7 A 4.07 3.74 4.67 3.41 5.73
P8 A 2.13 6.46 5.02 2.36 2.88
P9 A 2.24 2.85 0.63 1.39 3.14
P1 B 1.25 1.01 0.92 0.98 0.26
P2 B 1.55 1.44 0.90 1.12 1.74
P3 B 1.53 4.21 3.39 4.13 1.85
P4 B 2.56 1.49 1.09 2.37 0.66
P5 B 4.96 5.11 4.84 4.64 4.96
P6 B 1.89 3.00 3.05 1.22 1.58
P7 B 0.67 0.39 0.25 0.44 0.40
P8 B 0.47 0.51 0.37 0.27 0.40
P9 B 0.53 0.66 1.70 0.70 0.17
P1 C 5.37 3.96 4.05 3.37 4.11
P2 C 3.60 1.81 1.82 5.21 2.13
P3 C 4.77 3.46 5.58 4.45 2.80
P4 C 4.10 4.48 7.92 5.46 3.97
P5 C 2.96 5.14 3.16 4.46 2.45
P6 C 2.59 5.33 5.38 5.13 2.61
P7 C 5.17 5.66 4.84 4.47 3.44
P8 C 2.63 5.93 4.74 4.71 4.63
P9 C 4.48 2.90 2.91 5.73 3.71"""

# Table 8.23 (стр. 193): напряжение (мВ) → Km в повторностях 1 и 2.
VOLTAGE = {-160: (0.234, 0.219), -140: (0.320, 0.227), -120: (0.326, 0.282),
           -100: (0.327, 0.277), -80: (0.331, 0.343), -60: (0.489, 0.386),
           -40: (0.437, 0.421), -20: (0.786, 0.476), 0: (0.842, 0.611)}

_FOUR = "Control vs F1+F2+F3+F4; F1+F4 vs F2+F3; F1 vs F4; F2 vs F3"


def _rows(result) -> dict[str, dict]:
    return {r["term"]: r for r in result.parameters["anova_table"]}


def _herbicide() -> tuple[list[float], list[str], list[str], list[str]]:
    y, herb, pop, rep = [], [], [], []
    for line in HERBICIDE.splitlines():
        p, h, *cells = line.split()
        for r, v in enumerate(cells, 1):
            y.append(float(v))
            herb.append(h)
            pop.append(p)
            rep.append(str(r))
    return y, herb, pop, rep


def _voltage() -> tuple[list[float], list[str], list[str]]:
    y, volt, rep = [], [], []
    for v, km in VOLTAGE.items():
        for r, value in enumerate(km, 1):
            y.append(value)
            volt.append(str(v))
            rep.append(str(r))
    return y, volt, rep


# --- Разбор контрастов --------------------------------------------------------


def test_parse_set_and_explicit_syntax() -> None:
    levels = ["Control", "F1", "F2", "F3", "F4"]
    (name, a), (_, b) = parse_contrasts("Modes: F1+F4 vs F2, F3; f1=1, F4=-1", levels)
    assert name == "Modes"
    assert a.tolist() == [0, 0.5, -0.5, -0.5, 0.5]
    assert b.tolist() == [0, 1, 0, 0, -1]


@pytest.mark.parametrize("spec", ["", "F1=1, F2=1", "F9 vs F1", "F1 F2"])
def test_parse_rejects_bad_contrasts(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_contrasts(spec, ["F1", "F2", "F3"])


# --- Potato: Tables 8.17, 8.19 ------------------------------------------------


def test_potato_single_contrast_with_remainder_table_8_17() -> None:
    """Книга: Table 8.17 (стр. 185)."""
    y, _, trt, blk = _potato()
    result = treatment_contrasts(y, trt, block=blk, contrasts="AvB: F1+F4 vs F2+F3")
    contrast = result.parameters["contrasts"][0]
    assert contrast["estimate"] == pytest.approx(-36.75)
    assert contrast["se"] == pytest.approx(29.51, abs=5e-3)
    assert contrast["t"] == pytest.approx(-1.245, abs=5e-4)
    assert contrast["p"] == pytest.approx(0.237, abs=5e-4)
    rows = _rows(result)
    assert rows["AvB"]["ss"] == pytest.approx(5402.25)
    assert rows["AvB"]["F"] == pytest.approx(1.551, abs=5e-4)
    rem = rows["Remainder"]
    assert (rem["df"], rem["ss"]) == (3, pytest.approx(128016.95))
    assert rem["F"] == pytest.approx(12.251, abs=5e-4)


def test_potato_four_orthogonal_contrasts_table_8_19() -> None:
    """Книга: Table 8.19 (стр. 186)."""
    y, _, trt, blk = _potato()
    result = treatment_contrasts(y, trt, block=blk, contrasts=_FOUR)
    ss = [c["ss"] for c in result.parameters["contrasts"]]
    assert ss == pytest.approx([125294.45, 5402.25, 2178.00, 544.50])
    f = [c["F"] for c in result.parameters["contrasts"]]
    assert f == pytest.approx([35.972, 1.551, 0.625, 0.156], abs=5e-4)
    assert sum(ss) == pytest.approx(_rows(result)["Treatment"]["ss"])
    assert result.parameters["remainder"] is None
    assert result.parameters["non_orthogonal_pairs"] == []


def test_potato_psi1_standardized_matches_book() -> None:
    """Книга: Example 8.4E (стр. 187).

    Книга печатает ψ1 = −177.0 (SE 29.51): это стандартизованный контраст (Σl² = 1).

    Разность контроль − среднее фунгицидов равна −197.9 (SE 32.99); t одинаковый.
    """
    y, _, trt, blk = _potato()
    psi1 = treatment_contrasts(y, trt, block=blk, contrasts=_FOUR).parameters["contrasts"][0]
    assert psi1["estimate"] == pytest.approx(-197.875)
    assert psi1["se"] == pytest.approx(32.99, abs=5e-3)
    assert psi1["estimate_standardized"] == pytest.approx(-177.0, abs=0.05)
    assert psi1["se_standardized"] == pytest.approx(29.51, abs=5e-3)
    lo = psi1["estimate_standardized"] - 2.179 * psi1["se_standardized"]
    assert lo == pytest.approx(-241.3, abs=0.1)


def test_non_orthogonal_contrasts_flagged() -> None:
    y, _, trt, blk = _potato()
    result = treatment_contrasts(y, trt, block=blk, contrasts="Control vs F1; Control vs F2")
    assert any(w.code == "contrasts_not_orthogonal" for w in result.warnings)
    rem = result.parameters["remainder"]
    joint = sum(c["ss"] for c in result.parameters["contrasts"])
    assert rem["df"] == 2
    assert rem["ss"] != pytest.approx(_rows(result)["Treatment"]["ss"] - joint)


def test_bonferroni_widens_contrast_intervals() -> None:
    y, _, trt, blk = _potato()
    plain = treatment_contrasts(y, trt, block=blk, contrasts=_FOUR)
    bonf = treatment_contrasts(y, trt, block=blk, contrasts=_FOUR, adjustment="bonferroni")
    a, b = plain.parameters["contrasts"][2], bonf.parameters["contrasts"][2]
    assert b["ci_upper"] - b["ci_lower"] > a["ci_upper"] - a["ci_lower"]
    assert b["p_adjusted"] == pytest.approx(min(4 * a["p"], 1.0))


def test_invalid_contrast_is_reported() -> None:
    y, _, trt, blk = _potato()
    result = treatment_contrasts(y, trt, block=blk, contrasts="F1 vs F9")
    assert ("contrast_invalid", Severity.ERROR) in [(w.code, w.severity) for w in result.warnings]
    assert "contrasts" not in result.parameters


def test_contrast_unequal_replication_crd() -> None:
    """CRD с неравной повторностью: SE = √(s² Σ l²/n_j)."""
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 6.0]
    g = ["a", "a", "a", "b", "b", "c", "c", "c", "c"]
    c = treatment_contrasts(y, g, contrasts="a vs c").parameters["contrasts"][0]
    s2 = (2 + 0.5 + 5) / 6
    assert c["estimate"] == pytest.approx(2.0 - 7.5)
    assert c["se"] == pytest.approx(np.sqrt(s2 * (1 / 3 + 1 / 4)))


# --- Herbicide: Tables 8.21, 8.22 ---------------------------------------------


def test_herbicide_factorial_contrasts_table_8_22() -> None:
    """Книга: Table 8.22 (стр. 189)."""
    y, herb, pop, rep = _herbicide()
    result = factorial_anova(
        y, herb, pop, block=rep, transform_kind="sqrt",
        contrasts="Type 1 vs 2: A+C vs B; A vs C: A vs C",
        column_names={"factor_a": "Herbicide", "factor_b": "Population"})
    rows = _rows(result)
    expected = {"Type 1 vs 2": (1, 14.0774, 114.714), "A vs C": (1, 1.6434, 13.392),
                "(Type 1 vs 2).Population": (8, 12.4676, 12.700),
                "(A vs C).Population": (8, 1.4761, 1.504)}
    for term, (df, ss, f) in expected.items():
        assert rows[term]["df"] == df
        assert rows[term]["ss"] == pytest.approx(ss, abs=5e-5)
        assert rows[term]["F"] == pytest.approx(f, abs=5e-4)
    assert rows["(A vs C).Population"]["p"] == pytest.approx(0.165, abs=5e-4)
    assert not any(t.startswith("Remainder") for t in rows)
    names = [r["term"] for r in result.parameters["anova_table"]]
    assert names.index("Type 1 vs 2") == names.index("Herbicide") + 1


def test_herbicide_contrast_estimates_table_8_21() -> None:
    """Книга: Table 8.21 (стр. 189)."""
    y, herb, pop, rep = _herbicide()
    result = factorial_anova(y, herb, pop, block=rep, transform_kind="sqrt",
                             contrasts="A+C vs B; A vs C")
    type_c, avc = result.parameters["contrasts"]
    assert (type_c["estimate"], type_c["se"]) == pytest.approx((0.685, 0.0640), abs=5e-4)
    assert (avc["estimate"], avc["se"]) == pytest.approx((-0.270, 0.0739), abs=5e-4)
    assert type_c["by_cell"][4] == pytest.approx(-1.422, abs=5e-4)  # P5
    assert avc["by_cell"][8] == pytest.approx(-0.313, abs=5e-4)  # P9


def test_factorial_single_contrast_leaves_remainders() -> None:
    """Книга: Example 8.5 (стр. 189): остаток Remainder[A] = 1.6434."""
    y, herb, pop, rep = _herbicide()
    result = factorial_anova(y, herb, pop, block=rep, transform_kind="sqrt",
                             contrasts="A+C vs B")
    rows = _rows(result)
    assert rows["Remainder[A]"]["ss"] == pytest.approx(1.6434, abs=5e-5)
    assert rows["Remainder[A.B]"]["df"] == 8


def test_factorial_non_orthogonal_contrasts_rejected() -> None:
    y, herb, pop, rep = _herbicide()
    result = factorial_anova(y, herb, pop, block=rep, contrasts="A vs B; A vs C")
    assert any(w.code == "contrast_invalid" for w in result.warnings)


# --- Voltage: Tables 8.24, 8.25 -----------------------------------------------


@pytest.fixture(scope="module")
def voltage():
    y, volt, rep = _voltage()
    return polynomial_contrasts(y, volt, block=rep, degree=2, transform_kind="ln",
                                column_names={"group": "Voltage"})


def test_orthogonal_polynomial_coefficients_table_8_24() -> None:
    """Книга: Table 8.24 (стр. 194)."""
    x = np.arange(-160, 1, 20, dtype=float)
    lin, quad = orthogonal_polynomials(x, np.full(9, 2.0), 2)
    assert lin.tolist() == pytest.approx(list(range(-80, 81, 20)))
    assert np.round(quad).tolist() == [3733, 933, -1067, -2267, -2667, -2267, -1067, 933, 3733]


def test_orthogonal_polynomials_respect_replication() -> None:
    x, n = np.array([1.0, 2.0, 4.0, 8.0]), np.array([2.0, 3.0, 3.0, 5.0])
    poly = orthogonal_polynomials(x, n, 3)
    gram = (poly * n) @ poly.T
    assert np.allclose(gram - np.diag(np.diag(gram)), 0)
    assert np.allclose(poly @ n, 0)


def test_voltage_polynomial_anova_table_8_25(voltage) -> None:
    """Книга: Table 8.25 (стр. 194)."""
    rows = _rows(voltage)
    assert rows["Block"]["F"] == pytest.approx(12.185, abs=5e-4)
    assert rows["Voltage"]["ss"] == pytest.approx(2.33653, abs=5e-6)
    assert rows["Linear"]["ss"] == pytest.approx(2.20458, abs=5e-6)
    assert rows["Linear"]["F"] == pytest.approx(152.435, abs=5e-3)
    assert rows["Quadratic"]["F"] == pytest.approx(4.169, abs=5e-4)
    assert rows["Quadratic"]["p"] == pytest.approx(0.075, abs=5e-4)
    rem = rows["Remainder"]
    assert (rem["df"], rem["ss"]) == (6, pytest.approx(0.07166, abs=5e-6))
    assert (rem["F"], rem["p"]) == pytest.approx((0.826, 0.581), abs=5e-4)
    assert rows["Residual"]["ms"] == pytest.approx(0.01446, abs=5e-6)


def test_voltage_raw_contrasts_and_linear_fit(voltage) -> None:
    """Книга: Example 8.6 (стр. 193, 195).

    Σ l·τ̂ = 162.6 и 1218.8 (стр. 193); линейный тренд −0.434 + 0.0068·V (стр. 195).
    """
    est = [c["estimate"] for c in voltage.parameters["contrasts"]]
    assert est == pytest.approx([162.6, 1218.8], abs=0.05)
    linear = voltage.parameters["trend_fits"][0]["coefficients"]
    assert linear == pytest.approx([-0.434, 0.0068], abs=5e-4)


def test_polynomial_needs_numeric_levels() -> None:
    y, _, trt, blk = _potato()
    result = polynomial_contrasts(y, trt, block=blk)
    assert result.warnings[-1].code == "levels_not_numeric"


def test_polynomial_degree_capped_by_levels() -> None:
    y = [1.0, 2.0, 2.5, 3.0, 5.0, 6.5]
    result = polynomial_contrasts(y, ["1", "1", "2", "2", "3", "3"], degree=4)
    assert result.parameters["degree"] == 2
    assert result.parameters["remainder"] is None


def test_contrast_results_serialisable(voltage) -> None:
    json.dumps(voltage.to_dict(), allow_nan=False)
