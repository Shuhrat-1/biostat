"""Однородность дисперсий трёх и более групп: одно правило для ядра и карты.

Для двух групп правило — ``welch_rule`` (отношение SD и баланс n). Для трёх и
более групп книга даёт тест Бартлетта: X² против 95-го перцентиля χ²(t − 1)
[кн. стр. 105–107]. Если SD растут вместе со средними, лечат трансформацией
отклика [кн. стр. 96–97, 107, 115]; если дисперсии различаются без такой
связи — взвешенный анализ или ANOVA Уэлча [кн. стр. 107; предл.: Welch 1951].

Связь SD со средним — критерий BioStat: SD различаются хотя бы вдвое и
r(log SD, log среднего) ≥ 0.8. По трём группам такое r часто выходит случайно
(13 % при 3 × 3, аудит 01-07), поэтому при трёх группах совет — справка.

Предупреждения методов (`one_way_anova`, `anova_permutation_test`) и сигналы
карты B4.2, B4.3, B4.5, B4.6 вызывают одни и те же функции и не расходятся.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ALPHA = 0.05
SD_RATIO_FOR_TREND = 2.0      # SD различаются хотя бы вдвое [BioStat]
R_FOR_TREND = 0.8             # r(log SD, log среднего) [BioStat]
MIN_GROUPS_FOR_TREND = 4      # с 4 групп связь — уверенный сигнал [BioStat]


@dataclass(frozen=True)
class Spread:
    """Разброс внутри группы: среднее, оценка дисперсии и её df."""

    mean: float
    var: float
    df: float


@dataclass(frozen=True)
class VarianceAdvice:
    """Вывод правила.

    case: "transform" — SD растут со средними (при трёх группах ``confident``
    ложно: справка); "welch" — дисперсии неоднородны без такой связи.
    x2, crit — Бартлетт (None, если тест не определён); r — связь SD и среднего.
    """

    case: str
    x2: float | None
    crit: float | None
    r: float | None = None
    confident: bool = True


def spreads_of(groups: list[np.ndarray]) -> list[Spread]:
    """Разбросы групп с n ≥ 2 (выборочная дисперсия, df = n − 1)."""
    return [Spread(float(g.mean()), float(g.var(ddof=1)), g.size - 1.0)
            for g in groups if g.size >= 2]


def bartlett_statistic(spreads: list[Spread]) -> float:
    """X² Бартлетта по дисперсиям групп и их df [кн. стр. 105–106]."""
    df = np.array([s.df for s in spreads])
    var = np.array([s.var for s in spreads])
    total = df.sum()
    pooled = float(df @ var / total)
    correction = 1 + (np.sum(1 / df) - 1 / total) / (3 * (df.size - 1))
    return float((total * np.log(pooled) - df @ np.log(var)) / correction)


def bartlett(spreads: list[Spread]) -> tuple[float, float] | None:
    """(X², критическое χ²₀.₉₅(t − 1)); None — меньше двух групп или нулевая дисперсия."""
    if len(spreads) < 2 or any(s.var <= 0 for s in spreads):
        return None
    crit = float(stats.chi2.ppf(1 - ALPHA, len(spreads) - 1))
    return bartlett_statistic(spreads), crit


def sd_tracks_mean(spreads: list[Spread]) -> float | None:
    """r между log SD и log среднего (≥ 3 групп, SD ×2, r ≥ 0.8); None — связи нет."""
    means = np.array([s.mean for s in spreads])
    if len(spreads) < 3 or means.min() <= 0 or np.ptp(means) == 0:
        return None
    sds = np.sqrt([s.var for s in spreads])
    if sds.min() <= 0 or sds.max() / sds.min() < SD_RATIO_FOR_TREND:
        return None
    r = float(np.corrcoef(np.log(means), np.log(sds))[0, 1])
    return r if r >= R_FOR_TREND else None


def variance_advice(spreads: list[Spread]) -> VarianceAdvice | None:
    """Совет по дисперсиям трёх и более групп или None (дисперсии однородны)."""
    if len(spreads) < 3:
        return None
    test = bartlett(spreads)
    x2, crit = test if test is not None else (None, None)
    r = sd_tracks_mean(spreads)
    if r is not None:
        return VarianceAdvice("transform", x2, crit, r,
                              confident=len(spreads) >= MIN_GROUPS_FOR_TREND)
    if test is not None and x2 > crit:
        return VarianceAdvice("welch", x2, crit)
    return None
