"""Кандидат в выбросы по остаткам линейной модели.

Проверяется только самое крайнее наблюдение: удалённый (внешне
стьюдентизированный) остаток против t с поправкой на число наблюдений
[предл.: тест выброса, Cook & Weisberg 1982]. Для одного, самого крайнего
значения поправки Бонферрони, Холма и Бенджамини–Хохберга совпадают (порог
α / n). Тест работает и на малых опытах, где стандартизованный остаток не
может превысить √(n − p), так что порог «|r| > 3» недостижим.

Книга велит кандидата сначала проверить по записям, а не удалять и не менять
из-за него метод (стр. 109–110); поэтому здесь только отметка.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

ALPHA = 0.05
# Остатки этих методов — отклонения или условные остатки, не стандартизованные МНК.
_NOT_LINEAR = frozenset({"glm_analysis", "glmm_analysis", "mixed_model", "count_regression",
                         "ordinal_regression"})


def deleted_residuals(std_residuals: np.ndarray, df_resid: float) -> np.ndarray:
    """Удалённые остатки t_i из стандартизованных r_i: t = r·√((ν − 1) / (ν − r²))."""
    r = np.asarray(std_residuals, dtype=float)
    room = np.clip(df_resid - r**2, 1e-12, None)
    return np.where(r**2 < df_resid, r * np.sqrt((df_resid - 1) / room), np.copysign(np.inf, r))


def outlier_candidate(std_residuals: Any, df_resid: float, alpha: float = ALPHA) -> int | None:
    """Позиция самого крайнего наблюдения, если оно — кандидат в выбросы, иначе None."""
    r = np.asarray(std_residuals, dtype=float)
    if r.size < 3 or df_resid < 2 or not np.all(np.isfinite(r)):
        return None
    worst = int(np.argmax(np.abs(r)))
    t = deleted_residuals(r, df_resid)[worst]
    p = 2 * stats.t.sf(abs(t), df_resid - 1) * r.size
    return worst if p < alpha else None


def mark_residual_plot(bundle: dict[str, Any], method: str, statistics: dict[str, Any]) -> None:
    """Отметить кандидата на графике «остатки — подгонка» (ключ ``outlier`` — позиция точки).

    Отметка ставится всегда, когда тест её даёт, — и тогда, когда карта сигнала
    о выбросе не показывает (скошенные данные): смотреть на точку полезно в любом случае.
    """
    plot = bundle.get("residual_vs_fitted")
    df = statistics.get("df_resid")
    if not plot or method in _NOT_LINEAR or not isinstance(df, (int, float)):
        return
    hit = outlier_candidate(plot.get("residuals", []), float(df))
    if hit is not None:
        plot["outlier"] = hit
