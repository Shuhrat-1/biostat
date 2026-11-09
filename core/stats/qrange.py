"""Критическое q стьюдентизированного размаха — быстрее scipy ppf.

scipy.stats.studentized_range.ppf обращает CDF общим поиском: ~15 вызовов CDF,
каждый — двойной численный интеграл (≈ 0.14 с на вызов и в CPython, и в
Pyodide). Здесь корень sf(q) = α ищется brentq в узкой скобке Бонферрони:
√2·t(1 − α/2) ≤ q ≤ √2·t(1 − α/(k(k − 1))) [предл.: Hochberg & Tamhane 1987,
гл. 2 — Тьюки не консервативнее Бонферрони и не либеральнее одиночного t]. При
k = 2 скобка вырождается в точное q = √2·t. Результат кэшируется по (α, k, df).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import optimize, stats


@lru_cache(maxsize=256)
def q_critical(alpha: float, k: int, df: float) -> float:
    """q_{1−α}(k, df): верхняя α-квантиль стьюдентизированного размаха k средних."""
    if k < 2 or not 0 < alpha < 1 or not df > 0:
        raise ValueError("k ≥ 2, alpha ∈ (0, 1), df > 0")
    lo = np.sqrt(2) * stats.t.ppf(1 - alpha / 2, df)
    if k == 2:
        return float(lo)
    hi = np.sqrt(2) * stats.t.ppf(1 - alpha / (k * (k - 1)), df)

    def gap(q: float) -> float:
        return float(stats.studentized_range.sf(q, k, df)) - alpha

    return float(optimize.brentq(gap, lo * (1 - 1e-9), hi * (1 + 1e-9), xtol=1e-10, rtol=1e-12))
