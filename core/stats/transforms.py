"""Трансформации отклика и анализ на трансформированной шкале (гл. 6 книги).

Пары «трансформация — обратное преобразование» по Table 6.4 (стр. 123):
log10, ln (со сдвигом c), квадратный корень (со сдвигом), logit для
долей и процентов (со сдвигом). ANOVA и сравнения считаются на
трансформированной шкале; на исходную переводятся только средние и
границы ДИ (стр. 122): SE, SED и LSD обратно не переводятся.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from core.report.schema import Estimate, Severity, StatResult
from core.stats.anova import _clean_group, _resolve_labels, one_way_anova, treatment_means

TRANSFORMS = ("log10", "ln", "sqrt", "logit", "logit_pct")
_LOG_BASE = {"log10": 10.0, "ln": np.e}
# Верхняя граница шкалы для logit: доли — 1, проценты — 100.
_LOGIT_MAX = {"logit": 1.0, "logit_pct": 100.0}


def transform(values: np.ndarray, kind: str, offset: float = 0.0) -> np.ndarray:
    """z = f(y + c) по Table 6.4; ValueError, если значения вне области."""
    y = np.asarray(values, dtype=float)
    if kind in _LOG_BASE:
        if np.any(y + offset <= 0):
            raise ValueError("log определён только для y + c > 0")
        return np.log(y + offset) / np.log(_LOG_BASE[kind])
    if kind == "sqrt":
        if np.any(y + offset < 0):
            raise ValueError("√ определён только для y + c ≥ 0")
        return np.sqrt(y + offset)
    if kind in _LOGIT_MAX:
        top = _LOGIT_MAX[kind]
        if np.any(y + offset <= 0) or np.any(top - y + offset <= 0):
            raise ValueError(f"logit определён для 0 < y < {top:g} (с учётом сдвига)")
        return np.log((y + offset) / (top - y + offset))
    raise ValueError(f"kind ∈ {TRANSFORMS}")


def back_transform(z: np.ndarray | float, kind: str, offset: float = 0.0) -> np.ndarray:
    """Обратное преобразование по Table 6.4 (стр. 123).

    Для √ отрицательные z (нижняя граница ДИ) прижимаются к 0: иначе z²
    «перевернул» бы интервал.
    """
    z = np.asarray(z, dtype=float)
    if kind in _LOG_BASE:
        return _LOG_BASE[kind] ** z - offset
    if kind == "sqrt":
        return np.maximum(z, 0.0) ** 2 - offset
    if kind in _LOGIT_MAX:
        top = _LOGIT_MAX[kind]
        return ((top + offset) * np.exp(z) - offset) / (1 + np.exp(z))
    raise ValueError(f"kind ∈ {TRANSFORMS}")


def suggest_log_offset(values: np.ndarray) -> float:
    """Сдвиг для log при нулях (стр. 116): 1 для целых счётов, иначе ½ мин. положительного."""
    y = np.asarray(values, dtype=float)
    y = y[~np.isnan(y)]
    if not np.any(y <= 0):
        return 0.0
    if np.all(y == np.round(y)):
        return 1.0
    positive = y[y > 0]
    return float(positive.min() / 2) if positive.size else 1.0


def transformed_means(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
    transform_kind: str = "log10",
    offset: float = 0.0,
    alpha: float = 0.05,
) -> StatResult:
    """ANOVA на трансформированной шкале со средними и ДИ на исходной (гл. 6.3–6.4).

    Для log разности средних переводятся в отношения геометрических
    средних с ДИ, а LSD — в мультипликативный множитель (стр. 124–126).
    """
    if transform_kind not in TRANSFORMS:
        raise ValueError(f"transform_kind ∈ {TRANSFORMS}")
    clean = [_clean_group(g) for g in groups]
    labels = _resolve_labels(labels, len(clean))
    result = StatResult(
        method="transformed_means",
        n=int(sum(g.size for g in clean)),
        parameters={"labels": labels, "transform": transform_kind, "offset": offset, "alpha": alpha},
    )
    tgroups = _transform_groups(clean, transform_kind, offset, result)
    if tgroups is None:
        return result
    base = treatment_means(*tgroups, labels=labels, alpha=alpha)
    if any(w.severity == Severity.ERROR for w in base.warnings):
        result.warnings.extend(base.warnings)
        return result
    _fill_anova(tgroups, result)
    back = partial(back_transform, kind=transform_kind, offset=offset)
    _add_back_estimates(base, back, result)
    result.parameters["transformed"] = _transformed_summary(base)
    if transform_kind in _LOG_BASE:
        _add_ratios(base, _LOG_BASE[transform_kind], result)
    _check_back_range(base, result, transform_kind)
    result.warnings.extend(w for w in base.warnings if w.severity != Severity.ERROR)
    return result


def _transform_groups(
    groups: list[np.ndarray], kind: str, offset: float, result: StatResult
) -> list[np.ndarray] | None:
    """Трансформировать группы; при значениях вне области — ошибка с подсказкой сдвига."""
    try:
        return [transform(g, kind, offset) for g in groups]
    except ValueError as exc:
        if kind in _LOG_BASE:
            offset_hint = suggest_log_offset(np.concatenate(groups))
            result.add_warning(
                "transform_domain_log", str(exc), Severity.ERROR,
                params={"suggested_offset": offset_hint},
            )
        else:
            result.add_warning("transform_domain", str(exc), Severity.ERROR)
        return None


def _fill_anova(tgroups: list[np.ndarray], result: StatResult) -> None:
    """Таблица ANOVA на трансформированной шкале (как one_way_anova)."""
    anova = one_way_anova(*tgroups)
    result.p_value = anova.p_value
    keys = ("F", "df_between", "df_within", "ss_between", "ss_within", "ss_total",
            "ms_between", "ms_within")
    result.statistics = {k: anova.statistics[k] for k in keys}


def _add_back_estimates(
    base: StatResult, back: Callable[[float], np.ndarray], result: StatResult
) -> None:
    """Средние и границы ДИ на исходной шкале; SE не переводится (стр. 122)."""
    for est in base.estimates:
        result.estimates.append(
            Estimate(
                name=est.name,
                value=float(back(est.value)),
                ci_lower=float(back(est.ci_lower)),
                ci_upper=float(back(est.ci_upper)),
                ci_level=est.ci_level,
            )
        )


def _transformed_summary(base: StatResult) -> dict:
    """Средние, SEM, SED, LSD и попарные сравнения на трансформированной шкале."""
    return {
        "means": {e.name: e.value for e in base.estimates},
        "sem": {e.name: e.std_error for e in base.estimates},
        "statistics": dict(base.statistics),
        "comparisons": base.parameters["comparisons"],
    }


def _add_ratios(base: StatResult, log_base: float, result: StatResult) -> None:
    """Разность на log-шкале → отношение геометрических средних с ДИ (стр. 124–126)."""
    ratios = []
    for c in base.parameters["comparisons"]:
        ratios.append(
            {
                "group1": c["group1"],
                "group2": c["group2"],
                "ratio": float(log_base ** c["mean_diff"]),
                "ci_lower": float(log_base ** c["ci_lower"]),
                "ci_upper": float(log_base ** c["ci_upper"]),
                "lsd_factor": float(log_base ** c["lsd"]),
            }
        )
    result.parameters["ratios"] = ratios
    if "lsd" in base.statistics:
        result.statistics["lsd_factor"] = float(log_base ** base.statistics["lsd"])


def _check_back_range(base: StatResult, result: StatResult, kind: str) -> None:
    """Предупредить, если обратное преобразование вышло за допустимую шкалу (стр. 122)."""
    out = any(e.ci_lower < 0 for e in result.estimates)
    if kind == "sqrt":  # нижняя граница ниже 0 на шкале √ прижата к 0
        out = out or any(e.ci_lower < 0 for e in base.estimates)
    if kind in _LOGIT_MAX:
        out = out or any(e.ci_upper > _LOGIT_MAX[kind] for e in result.estimates)
    if out:
        result.add_warning(
            "back_transform_out_of_range",
            "После обратного преобразования со сдвигом границы вышли за допустимую "
            "шкалу (отрицательные значения или больше максимума). Рассмотри GLM (гл. 18).",
            Severity.WARNING,
        )
