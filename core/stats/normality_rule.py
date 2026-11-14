"""Одно правило «нормальность нарушена» для методов ядра и карты (аудит 02-06).

Шапиро–Уилк при больших выборках отвергает и пустяковые отклонения, при
которых F и t устойчивы по центральной предельной теореме. Поэтому при
n > 30 в каждой группе нарушением считается только заметная асимметрия
(|g₁| > 0.5) [BioStat]; при меньших выборках решает сам тест [кн. стр. 101].
Правило общее с распознавателями B4.1 и B0.6 — как `welch_rule.py` для
дисперсий: предупреждение метода и сигнал карты не расходятся.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ALPHA = 0.05
LARGE_GROUP = 30   # больше — работает ЦПТ, важна только асимметрия [BioStat]
MIN_SKEW = 0.5     # |g₁| остатков, заметная при больших группах [BioStat]
SHAPIRO_MIN = 3
SHAPIRO_MAX = 5000  # предел scipy.stats.shapiro


@dataclass(frozen=True)
class NormalityCheck:
    """P Шапиро–Уилка и вывод правила."""

    p: float
    violated: bool


def shapiro_p(values: np.ndarray, truncate: bool = False) -> float | None:
    """P Шапиро–Уилка; None — значений мало, слишком много или они не варьируют."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if truncate:
        v = v[:SHAPIRO_MAX]
    if not SHAPIRO_MIN <= v.size <= SHAPIRO_MAX or float(np.ptp(v)) == 0.0:
        return None
    return float(stats.shapiro(v).pvalue)


def departure_matters(values: np.ndarray, min_group_size: int) -> bool:
    """Важно ли отклонение: в малых группах — всегда, в больших — при заметной асимметрии."""
    if min_group_size <= LARGE_GROUP:
        return True
    return abs(float(stats.skew(np.asarray(values, dtype=float)))) > MIN_SKEW


def check_normality(values: np.ndarray, min_group_size: int,
                    truncate: bool = False) -> NormalityCheck | None:
    """Правило для остатков (или разностей); None — тест неприменим.

    min_group_size — наименьшее число наблюдений в группе (для одной выборки
    и регрессии — всё n): от него зависит, работает ли ЦПТ.
    """
    p = shapiro_p(values, truncate)
    if p is None:
        return None
    return NormalityCheck(p, p < ALPHA and departure_matters(values, min_group_size))
