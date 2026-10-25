"""Тесты RCBD и multi-stratum ANOVA (гл. 7): Tables 7.4, 7.7, Example 7.1D."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from core.report.schema import Severity
from core.stats.blocking import rcbd_anova, rcbd_subsampling_anova

# Welham et al. (2015), Tables 7.1 и 7.5 (стр. 131, 142): potato.dat / potatorow.dat.
LAYOUT = {
    1: ["F3", "Control", "F2", "F1", "F4"],
    2: ["F2", "Control", "F3", "F4", "F1"],
    3: ["Control", "F2", "F3", "F4", "F1"],
    4: ["F3", "F2", "F1", "Control", "F4"],
}
PLOT_YIELD = {
    1: [642, 377, 633, 527, 623],
    2: [600, 408, 708, 550, 604],
    3: [500, 650, 662, 562, 606],
    4: [504, 567, 533, 333, 667],
}
ROWS = {
    1: [[720, 528, 678, 642], [348, 405, 364, 391], [652, 658, 569, 653],
        [635, 512, 536, 425], [642, 639, 642, 569]],
    2: [[554, 618, 621, 607], [411, 374, 396, 451], [682, 741, 712, 697],
        [639, 544, 521, 496], [583, 530, 629, 674]],
    3: [[561, 491, 429, 519], [555, 633, 715, 697], [638, 712, 633, 665],
        [505, 597, 607, 539], [598, 620, 596, 610]],
    4: [[451, 493, 535, 537], [513, 626, 574, 555], [441, 467, 701, 523],
        [367, 319, 361, 285], [631, 618, 689, 730]],
}


def _plot_data():
    y, trt, blk = [], [], []
    for b, cells in LAYOUT.items():
        for p, t in enumerate(cells):
            y.append(PLOT_YIELD[b][p]), trt.append(t), blk.append(str(b))
    return y, trt, blk


def _row_data():
    y, trt, blk, unit = [], [], [], []
    for b, cells in LAYOUT.items():
        for p, t in enumerate(cells):
            for value in ROWS[b][p]:
                y.append(value), trt.append(t), blk.append(str(b)), unit.append(str(p + 1))
    return y, trt, blk, unit


def _codes(result) -> set[str]:
    return {w.code for w in result.warnings}


def test_book_table_7_4_rcbd_anova():
    """Table 7.4 (стр. 137): Block 3/14 987.20, Fungicide 4/133 419.20, Residual 12/41 796.80."""
    r = rcbd_anova(*_plot_data())
    s = r.statistics
    assert (s["df_block"], s["df_treatment"], s["df_resid"]) == (3, 4, 12)
    assert s["ss_block"] == pytest.approx(14987.20)
    assert s["ss_treatment"] == pytest.approx(133419.20)
    assert s["ss_resid"] == pytest.approx(41796.80)
    assert s["ss_total"] == pytest.approx(190203.20)
    assert s["ms_resid"] == pytest.approx(3483.07, abs=5e-3)
    assert s["F"] == pytest.approx(9.576, abs=5e-4)
    assert s["F_block"] == pytest.approx(1.434, abs=5e-4)
    assert r.p_value == pytest.approx(0.001, abs=5e-4)
    # В книге P для блоков 0.283; при F = 1.434 на (3, 12) df точное P = 0.2814.
    assert s["p_block"] == pytest.approx(0.2814, abs=5e-4)


def test_book_example_7_1d_means_sem_sed_lsd():
    """Example 7.1B, D (стр. 133, 141): контроль 404.5, SEM 29.51, SED 41.73, LSD 90.93."""
    r = rcbd_anova(*_plot_data())
    means = {e.name: e for e in r.estimates}
    assert means["mean_Control"].value == pytest.approx(404.5)
    assert means["mean_F3"].value == pytest.approx(629.0)
    assert means["mean_F1"].std_error == pytest.approx(29.51, abs=5e-3)
    assert r.statistics["sed"] == pytest.approx(41.73, abs=5e-3)
    assert r.statistics["lsd"] == pytest.approx(90.93, abs=5e-3)
    assert r.statistics["t_crit"] == pytest.approx(2.179, abs=5e-4)


def test_book_table_7_7_multistratum_with_rows():
    """Table 7.7 (стр. 145): три страты, SS ×4 к анализу средних делянок."""
    r = rcbd_subsampling_anova(*_row_data())
    s = r.statistics
    assert s["ss_block"] == pytest.approx(59948.80)
    assert s["ms_block"] == pytest.approx(19982.93, abs=5e-3)
    assert s["ss_treatment"] == pytest.approx(533676.80)
    assert s["ss_resid"] == pytest.approx(167187.20)
    assert s["ms_resid"] == pytest.approx(13932.27, abs=5e-3)
    assert (s["ss_within"], s["df_within"]) == (pytest.approx(186848.00), 60)
    assert s["ms_within"] == pytest.approx(3114.13, abs=5e-3)
    assert s["F"] == pytest.approx(9.576, abs=5e-4)
    assert s["F_unit_residual"] == pytest.approx(4.474, abs=5e-4)
    assert s["p_unit_residual"] < 0.001
    assert s["ss_total"] == pytest.approx(947660.80)
    assert r.parameters["subsamples_per_unit"] == 4


def test_subsampling_gives_same_treatment_inference_as_plot_means():
    """Выводы о вариантах те же, что по средним делянок (стр. 145)."""
    plots, rows = rcbd_anova(*_plot_data()), rcbd_subsampling_anova(*_row_data())
    assert rows.statistics["F"] == pytest.approx(plots.statistics["F"])
    assert rows.statistics["sed"] == pytest.approx(plots.statistics["sed"])
    assert [e.value for e in rows.estimates] == pytest.approx([e.value for e in plots.estimates])


def test_rows_without_plot_column_rejected_as_pseudo_replication():
    """Table 7.8 — неверный анализ на 72 df; без колонки делянки метод его не делает."""
    y, trt, blk, _ = _row_data()
    r = rcbd_anova(y, trt, blk)
    assert any(w.code == "rcbd_duplicate_cells" and w.severity == Severity.ERROR for w in r.warnings)
    assert r.p_value is None


def test_matches_statsmodels_on_random_rcbd():
    rng = np.random.default_rng(41)
    rows = [(f"B{b}", f"T{t}", 10 + b + 2 * t + rng.normal()) for b in range(6) for t in range(3)]
    frame = pd.DataFrame(rows, columns=["block", "trt", "y"])
    ref = anova_lm(smf.ols("y ~ C(block) + C(trt)", frame).fit())
    r = rcbd_anova(frame["y"].tolist(), frame["trt"].tolist(), frame["block"].tolist())
    assert r.statistics["ss_treatment"] == pytest.approx(ref.loc["C(trt)", "sum_sq"])
    assert r.p_value == pytest.approx(ref.loc["C(trt)", "PR(>F)"])


def test_incomplete_design_is_error():
    y, trt, blk = _plot_data()
    r = rcbd_anova(y[:-1], trt[:-1], blk[:-1])
    assert "rcbd_incomplete" in _codes(r)


def test_unequal_subsamples_is_error():
    y, trt, blk, unit = _row_data()
    r = rcbd_subsampling_anova(y[:-1], trt[:-1], blk[:-1], unit[:-1])
    assert "rcbd_unequal_subsamples" in _codes(r)


def test_plot_with_two_treatments_is_error():
    y, trt, blk, unit = _row_data()
    trt = list(trt)
    trt[1] = "F4" if trt[1] != "F4" else "F1"
    assert "rcbd_unit_mixed" in _codes(rcbd_subsampling_anova(y, trt, blk, unit))


def test_missing_values_dropped_then_design_checked():
    y, trt, blk = _plot_data()
    y = list(y) + [np.nan]
    trt, blk = list(trt) + ["F1"], list(blk) + ["1"]
    assert rcbd_anova(y, trt, blk).n == 20


def test_residuals_are_internally_studentized():
    y, trt, blk = _plot_data()
    frame = pd.DataFrame({"y": y, "trt": trt, "block": blk})
    fit = smf.ols("y ~ C(block) + C(trt)", frame.sort_values(["block", "trt"])).fit()
    ref = sorted(fit.get_influence().resid_studentized_internal)
    got = rcbd_anova(y, trt, blk).parameters["plots"]["qq"]["sample_quantiles"]
    assert got == pytest.approx(ref)


def test_unequal_column_lengths_raise():
    with pytest.raises(ValueError):
        rcbd_anova([1.0, 2.0], ["a"], ["1", "2"])


def test_result_json_safe():
    json.dumps(rcbd_subsampling_anova(*_row_data()).to_dict(), allow_nan=False)
