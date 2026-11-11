"""Рандомизированный полный блочный дизайн (RCBD) и multi-stratum ANOVA (гл. 7).

Две точки входа:
- ``rcbd_anova`` — одно наблюдение на делянку (вариант × блок): страты
  Block и Block.Unit (Table 7.6);
- ``rcbd_subsampling_anova`` — несколько измерений на делянку
  (субвыборки, псевдоповторности): страты Block, Block.Plot и
  Block.Plot.Sample (Table 7.7). Варианты проверяются против остатка
  страты делянок, а не против разброса внутри делянок (стр. 145).

Таблица ANOVA по средним делянок считается через statsmodels; для
субвыборок суммы квадратов страт делянок и блоков умножаются на число
измерений k на делянку (стр. 145). Требуется полный сбалансированный
дизайн: каждый вариант ровно один раз в каждом блоке, одинаковое k.
"""

from __future__ import annotations

import copy
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.anova import anova_lm

from core.report.schema import Severity, StatResult
from core.stats.anova import _add_mean_estimates, _lsd_comparisons
from core.stats.transforms import (
    _LOG_BASE, TRANSFORMS, _add_back_estimates, _add_ratios, _check_back_range,
    _transformed_summary, back_transform, transform,
)

_NONE = "none"

_OUTLIER_CUTOFF = 2.0


def rcbd_anova(
    values: list[float],
    treatments: list[str],
    blocks: list[str],
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
) -> StatResult:
    """Блочный ANOVA RCBD: одно наблюдение на вариант в каждом блоке (гл. 7.1–7.4).

    ``transform_kind`` — анализ на трансформированной шкале (гл. 6): средние и
    ДИ возвращаются на исходной, для log — ещё и отношения средних.
    """
    frame = _frame(values, treatments, blocks, None)
    return _analyse(frame, "rcbd_anova", alpha, transform_kind, offset)


def rcbd_subsampling_anova(
    values: list[float],
    treatments: list[str],
    blocks: list[str],
    units: list[str],
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
) -> StatResult:
    """Multi-stratum ANOVA RCBD с субвыборками внутри делянок (гл. 7.5).

    При ``transform_kind`` преобразуются отдельные измерения, затем берутся
    средние делянок на трансформированной шкале (гл. 6).
    """
    frame = _frame(values, treatments, blocks, units)
    return _analyse(frame, "rcbd_subsampling_anova", alpha, transform_kind, offset)


def _frame(
    values: list[float], treatments: list[str], blocks: list[str], units: list[str] | None
) -> pd.DataFrame:
    """Выровнять колонки и удалить строки с пропуском в любой из них."""
    data = {"y": values, "trt": treatments, "block": blocks}
    if units is not None:
        data["unit"] = units
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError("Колонки должны быть одной длины")
    frame = pd.DataFrame(data)
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    for col in frame.columns.drop("y"):
        frame[col] = frame[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    return frame.dropna().reset_index(drop=True)


def _analyse(frame: pd.DataFrame, method: str, alpha: float,
             kind: str = _NONE, offset: float = 0.0) -> StatResult:
    """Проверить дизайн, построить страты, средние вариантов и остатки."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должен быть в (0, 1)")
    if kind != _NONE and kind not in TRANSFORMS:
        raise ValueError(f"transform_kind ∈ {(_NONE, *TRANSFORMS)}")
    result = StatResult(method=method, n=len(frame), parameters={"alpha": alpha})
    if not _transform_response(frame, kind, offset, result):
        return result
    plots, k = _plot_level(frame, result)
    if plots is None:
        return result
    fit = smf.ols("y ~ C(block) + C(trt)", data=plots).fit()
    table = anova_lm(fit)
    if table.loc["Residual", "sum_sq"] <= 0:
        result.add_warning(
            "no_residual_variation", "Остаточной вариации нет", Severity.ERROR
        )
        return result
    _fill_strata(table, frame, k, result)
    _fill_means(plots, table, alpha, result)
    _fill_residuals(fit, plots, result)
    if kind != _NONE:
        _to_original_scale(result, kind, offset)
    return result


def _transform_response(frame: pd.DataFrame, kind: str, offset: float,
                        result: StatResult) -> bool:
    """Трансформировать отклик (гл. 6); значения вне области — ошибка с подсказкой."""
    if kind == _NONE:
        return True
    try:
        frame["y"] = transform(frame["y"].to_numpy(float), kind, offset)
    except ValueError as exc:
        result.add_warning("transform_domain", str(exc), Severity.ERROR)
        return False
    return True


def _to_original_scale(result: StatResult, kind: str, offset: float) -> None:
    """Средние и ДИ — на исходной шкале, для log — отношения средних (стр. 122–126).

    Таблица ANOVA, SED, LSD и попарные сравнения остаются на шкале анализа;
    сводка на ней же — в ``parameters["transformed"]``, как у transformed_means.
    """
    base = copy.copy(result)
    base.estimates, result.estimates = list(result.estimates), []
    _add_back_estimates(base, partial(back_transform, kind=kind, offset=offset), result)
    result.parameters.update({"transform": kind, "offset": offset,
                              "transformed": _transformed_summary(base)})
    if kind in _LOG_BASE:
        _add_ratios(base, _LOG_BASE[kind], result)
    _check_back_range(base, result, kind)


def _plot_level(frame: pd.DataFrame, result: StatResult) -> tuple[pd.DataFrame | None, int]:
    """Средние делянок и число измерений k на делянку; ошибка при плохом дизайне."""
    key = ["block", "unit"] if "unit" in frame else ["block", "trt"]
    per_unit = frame.groupby(key)
    if "unit" in frame and (per_unit["trt"].nunique() > 1).any():
        return _design_error(result, "rcbd_unit_mixed", "В одной делянке разные варианты"), 0
    sizes = per_unit.size()
    if "unit" not in frame and (sizes > 1).any():
        return _design_error(
            result, "rcbd_duplicate_cells",
            "Несколько наблюдений на вариант в блоке: если это измерения одной "
            "делянки — используй анализ с субвыборками и колонкой делянки",
        ), 0
    if sizes.nunique() != 1:
        return _design_error(result, "rcbd_unequal_subsamples",
                             "Разное число измерений на делянку"), 0
    if "unit" in frame:
        plots = per_unit.agg(y=("y", "mean"), trt=("trt", "first")).reset_index()
    else:
        plots = per_unit["y"].mean().reset_index()
    plots["block"] = plots["block"].astype(str)
    if not _is_complete(plots, result):
        return None, 0
    return plots, int(sizes.iloc[0])


def _design_error(result: StatResult, code: str, message: str) -> None:
    """Записать ошибку дизайна и вернуть None."""
    result.add_warning(code, message, Severity.ERROR)
    return None


def _is_complete(plots: pd.DataFrame, result: StatResult) -> bool:
    """Каждый вариант ровно один раз в каждом блоке, ≥ 2 блоков и вариантов (стр. 58)."""
    counts = pd.crosstab(plots["block"], plots["trt"])
    n_blocks, n_trts = counts.shape
    if n_blocks < 2 or n_trts < 2:
        _design_error(result, "rcbd_too_small", "Нужно ≥ 2 блоков и ≥ 2 вариантов")
        return False
    if not (counts == 1).all().all():
        _design_error(
            result, "rcbd_incomplete",
            "Не каждый вариант встречается ровно один раз в каждом блоке: дизайн "
            "неполный или несбалансированный (гл. 9, 11)",
        )
        return False
    result.parameters["labels"] = list(counts.columns)
    result.parameters["blocks"] = list(counts.index)
    return True


def _fill_strata(table: pd.DataFrame, frame: pd.DataFrame, k: int, result: StatResult) -> None:
    """Таблица страт; SS по средним делянок × k, остаток внутри делянок (стр. 145)."""
    ss_block, ss_trt, ss_res = (k * float(table.loc[r, "sum_sq"]) for r in
                                ("C(block)", "C(trt)", "Residual"))
    df_block, df_trt, df_res = (int(table.loc[r, "df"]) for r in ("C(block)", "C(trt)", "Residual"))
    ms_block, ms_trt, ms_res = ss_block / df_block, ss_trt / df_trt, ss_res / df_res
    f_trt, f_block = ms_trt / ms_res, ms_block / ms_res
    result.p_value = float(stats.f.sf(f_trt, df_trt, df_res))
    result.statistics = {
        "ss_block": ss_block, "df_block": df_block, "ms_block": ms_block,
        "F_block": f_block, "p_block": float(stats.f.sf(f_block, df_block, df_res)),
        "ss_treatment": ss_trt, "df_treatment": df_trt, "ms_treatment": ms_trt,
        "F": f_trt, "ss_resid": ss_res, "df_resid": df_res, "ms_resid": ms_res,
    }
    result.parameters["subsamples_per_unit"] = k
    if k > 1:
        _fill_within_stratum(frame, ms_res, df_res, result)
    result.statistics["ss_total"] = float(((frame["y"] - frame["y"].mean()) ** 2).sum())


def _fill_within_stratum(
    frame: pd.DataFrame, ms_unit: float, df_unit: int, result: StatResult
) -> None:
    """Страта измерений внутри делянок и тест вариации между делянками (стр. 146)."""
    within = frame["y"] - frame.groupby(["block", "unit"])["y"].transform("mean")
    ss_w = float((within**2).sum())
    df_w = int(len(frame) - frame.groupby(["block", "unit"]).ngroups)
    stats_ = result.statistics
    stats_.update({"ss_within": ss_w, "df_within": df_w})
    if df_w > 0 and ss_w > 0:
        ms_w = ss_w / df_w
        f_unit = ms_unit / ms_w
        stats_.update({
            "ms_within": ms_w, "F_unit_residual": f_unit,
            "p_unit_residual": float(stats.f.sf(f_unit, df_unit, df_w)),
        })


def _fill_means(plots: pd.DataFrame, table: pd.DataFrame, alpha: float, result: StatResult) -> None:
    """Средние вариантов, SEM, ДИ, SED, LSD по остатку страты делянок (стр. 140–141)."""
    s2 = float(table.loc["Residual", "mean_sq"])  # на шкале средних делянок
    df = int(table.loc["Residual", "df"])
    labels = result.parameters["labels"]
    groups = [plots.loc[plots["trt"] == lab, "y"].to_numpy(float) for lab in labels]
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    _add_mean_estimates(groups, labels, s2, t_crit, 1 - alpha, result)
    result.parameters["comparisons"] = _lsd_comparisons(groups, labels, s2, df, t_crit)
    sed = float(np.sqrt(2 * s2 / groups[0].size))
    result.statistics.update({"sed": sed, "lsd": t_crit * sed, "t_crit": t_crit})
    if len(labels) > 2:
        result.add_warning(
            "unadjusted_comparisons",
            "Попарные p и ДИ разностей без поправки на множественность; "
            "для всех попарных сравнений используй Tukey HSD.",
            Severity.INFO,
        )


def _fill_residuals(
    fit: RegressionResultsWrapper, plots: pd.DataFrame, result: StatResult
) -> None:
    """Стандартизованные остатки модели делянок и ряды для графиков (гл. 5)."""
    std = np.asarray(fit.get_influence().resid_studentized_internal, dtype=float)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    n = std.size
    blom = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    result.parameters["plots"] = {
        "residual_vs_fitted": {"fitted": fitted.tolist(), "residuals": std.tolist()},
        "qq": {"theoretical_quantiles": stats.norm.ppf(blom).tolist(),
               "sample_quantiles": np.sort(std).tolist()},
        "std_residuals": std.tolist(),
    }
    flagged = int((np.abs(std) > _OUTLIER_CUTOFF).sum())
    result.statistics["n_abs_std_residual_gt_2"] = flagged
    if flagged:
        result.add_warning(
            "potential_outliers",
            f"Делянок с |стандартизованным остатком| > 2: {flagged}. Проверь их "
            f"по первичным записям; удалять без найденной причины нельзя.",
            Severity.INFO,
            params={"count": flagged, "expected": round(0.05 * n, 1)},
        )
