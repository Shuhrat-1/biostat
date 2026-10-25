"""Тесты сравнений по таблице средних (гл. 8.8): Example 8.7, Tables 8.28, 8.29.

Числа сверены с картинками страниц 196, 199, 203–205 (PDF 218, 221, 225–227).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from core.stats.multcomp import dunnett_cdf, dunnett_quantile, mean_comparisons

# Table 8.27 (стр. 202–203): по делянкам — линия и масличность в блоках 1–3.
LUPIN = """DTN84 0.36 DTN84 0.34 DTN31 0.21
DTN108 0.58 DTN12 0.36 DTN78 0.36
DTN78 0.39 DTN04 0.33 DTN10 0.32
DTN19B 0.38 DTN11 0.38 CH304-70 0.24
CH304-73 0.37 DTN19B 0.33 DTN84 0.58
DTN10 0.30 DTN10 0.34 DTN108 0.56
DTN11 0.29 DTN108 0.54 DTN20 0.41
DTN19A 0.17 DTN20 0.35 DTN19B 0.32
DTN04 0.26 DTN31 0.18 DTN19A 0.30
DTN31 0.19 DTN01 0.37 DTN01 0.38
CH304-70 0.22 CH304-73 0.24 DTN04 0.23
DTN20 0.32 DTN19A 0.23 DTN11 0.28
DTN12 0.31 DTN78 0.59 DTN12 0.35
DTN01 0.32 CH304-70 0.24 CH304-73 0.29"""


def _lupin() -> tuple[list[float], list[str], list[str]]:
    y, line, block = [], [], []
    for row in LUPIN.splitlines():
        cells = row.split()
        for b in range(3):
            line.append(cells[2 * b])
            y.append(float(cells[2 * b + 1]))
            block.append(str(b + 1))
    return y, line, block


def _run(procedure: str, control: str = "DTN20", **kwargs):
    y, line, block = _lupin()
    return mean_comparisons(y, line, block=block, procedure=procedure, control=control,
                            **kwargs)


def _different(result) -> set[str]:
    return {c["group2"] for c in result.parameters["comparisons"] if c["significant"]}


def _range(result) -> tuple[float, float]:
    """Интервал средних, не отличимых от DTN20 (Fig. 8.10)."""
    half = result.parameters["comparisons"][0]["half_width"]
    return 0.360 - half, 0.360 + half


# --- Распределение Даннета ----------------------------------------------------


def test_dunnett_quantile_book_value() -> None:
    """d[0.025] для 13 вариантов и 26 df = 3.004 (стр. 203; в формуле опечатка 16 df)."""
    assert dunnett_quantile(0.05, np.ones(13), 26) == pytest.approx(3.004, abs=5e-4)


def test_dunnett_single_comparison_is_t() -> None:
    assert dunnett_quantile(0.05, np.ones(1), 10) == pytest.approx(
        stats.t.ppf(0.975, 10), abs=1e-4)


def test_dunnett_cdf_matches_scipy_dunnett() -> None:
    rng = np.random.default_rng(3)
    control = rng.normal(size=6)
    groups = [rng.normal(loc=0.4 * i, size=6) for i in range(1, 4)]
    ref = stats.dunnett(*groups, control=control, random_state=1).pvalue
    s2 = np.mean([np.var(g, ddof=1) for g in (control, *groups)])
    t_stat = [abs(g.mean() - control.mean()) / np.sqrt(2 * s2 / 6) for g in groups]
    ours = [1 - dunnett_cdf(t, np.ones(3), 20) for t in t_stat]
    assert ours == pytest.approx(ref, abs=2e-3)


def test_dunnett_one_sided_smaller_than_two_sided() -> None:
    one = dunnett_quantile(0.05, np.ones(4), 20, two_sided=False)
    two = dunnett_quantile(0.05, np.ones(4), 20)
    assert one < two
    assert two == pytest.approx(2.65, abs=5e-3)  # таблица Даннета, k = 4, df = 20


# --- Example 8.7 --------------------------------------------------------------


def test_lupin_means_and_sed_table_8_28() -> None:
    result = _run("lsd")
    means = {e.name: e.value for e in result.estimates}
    assert means["DTN108"] == pytest.approx(0.560, abs=5e-4)
    assert means["DTN31"] == pytest.approx(0.193, abs=5e-4)
    assert result.statistics["ms_resid"] == pytest.approx(0.0039, abs=5e-5)
    assert result.statistics["df_resid"] == 26
    assert result.parameters["comparisons"][0]["sed"] == pytest.approx(0.0509, abs=5e-5)


def test_lupin_dunnett_range() -> None:
    result = _run("dunnett")
    assert _range(result) == pytest.approx((0.207, 0.513), abs=5e-4)
    assert _different(result) == {"DTN108", "DTN31"}


def test_lupin_bonferroni_range() -> None:
    result = _run("bonferroni")
    assert _range(result) == pytest.approx((0.199, 0.521), abs=5e-4)
    assert _different(result) == {"DTN108", "DTN31"}


def test_lupin_tukey_and_alsd() -> None:
    """Тьюки учитывает все 91 пару: отличается только DTN108; aLSD близок к Тьюки."""
    tukey, alsd = _run("tukey"), _run("alsd")
    assert _different(tukey) == {"DTN108"}
    assert _different(alsd) == {"DTN108"}
    half_t = tukey.parameters["comparisons"][0]["half_width"]
    half_a = alsd.parameters["comparisons"][0]["half_width"]
    sed = tukey.parameters["comparisons"][0]["sed"]
    assert half_t == pytest.approx(stats.studentized_range.ppf(0.95, 14, 26) / np.sqrt(2) * sed)
    assert half_a == pytest.approx(stats.studentized_range.ppf(0.95, 13, 26) / np.sqrt(2) * sed)


def test_lupin_lsd_and_observed_fdr() -> None:
    result = _run("lsd")
    assert _different(result) == {"DTN31", "DTN19A", "CH304-70", "DTN108"}
    assert result.statistics["observed_fdr"] == pytest.approx(0.1625)
    assert any(w.code == "unadjusted_comparisons" for w in result.warnings)


def test_lupin_fdr_table_8_29() -> None:
    result = _run("fdr")
    rows = {c["group2"]: c for c in result.parameters["comparisons"]}
    assert rows["DTN108"]["t"] == pytest.approx(3.933, abs=5e-3)
    assert rows["DTN31"]["p_unadjusted"] == pytest.approx(0.003, abs=5e-4)
    assert rows["DTN01"]["t"] == pytest.approx(-0.066, abs=5e-3)
    assert _different(result) == {"DTN108", "DTN31"}


def test_all_pairs_count() -> None:
    result = _run("tukey", control="")
    assert result.statistics["n_comparisons"] == 91


def test_one_sided_dunnett_has_single_bound() -> None:
    result = _run("dunnett", alternative="greater")
    row = result.parameters["comparisons"][0]
    assert row["ci_upper"] is None
    assert _different(result) == {"DTN108"}


@pytest.mark.parametrize(("procedure", "control", "alternative"), [
    ("dunnett", "", "two-sided"), ("tukey", "DTN20", "greater"),
    ("lsd", "", "less"), ("lsd", "DTN99", "two-sided")])
def test_invalid_settings_reported(procedure: str, control: str, alternative: str) -> None:
    result = _run(procedure, control=control, alternative=alternative)
    assert result.warnings[-1].code == "comparison_invalid"


def test_dunnett_unequal_replication_crd() -> None:
    """Неравная повторность: квантиль по отношениям n_i / n_0, P согласован с ДИ."""
    rng = np.random.default_rng(7)
    sizes = {"ctrl": 8, "a": 4, "b": 5, "c": 4}
    y, g = [], []
    for k, n in sizes.items():
        y += list(rng.normal(loc=1.5 if k == "b" else 0.0, size=n))
        g += [k] * n
    result = mean_comparisons(y, g, procedure="dunnett", control="ctrl")
    for row in result.parameters["comparisons"]:
        excludes_zero = row["ci_lower"] > 0 or row["ci_upper"] < 0
        assert excludes_zero == row["significant"]


def test_comparisons_serialisable() -> None:
    json.dumps(_run("dunnett", alternative="less").to_dict(), allow_nan=False)
