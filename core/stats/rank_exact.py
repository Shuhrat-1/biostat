"""Точное перестановочное P ранговых тестов — с рангами при совпадениях (аудит 02-02).

При совпадениях и нулевых разностях scipy переходит на нормальное приближение
при любом n, и на малых выборках P выходит ниже достижимого: 6 пар с одной
нулевой разностью дают 0.043 при точном 0.0625. Здесь P считается по
перестановочному распределению самих рангов (средние ранги при совпадениях):

* знаково-ранговый тест — все 2ⁿ расстановок знаков равновероятны;
* Манн–Уитни — все C(N, n₁) разбиений на группы равновероятны.

Распределение суммы рангов строится свёрткой (динамическое программирование),
а не перебором: результат тот же, время — миллисекунды. Удвоенные средние
ранги — целые, поэтому суммы сравниваются точно.

[BioStat] Источник: перестановочное распределение при совпадениях (мнение
аудитора; Witte, гл. 20, решает малые выборки по таблицам, то есть точно).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# [BioStat] Предел точного расчёта — по замеру в Pyodide 0.26.4 (≤ 0.5 с):
# знаково-ранговый тест — до 200 ненулевых разностей (0.02 с; при 400 — 0.15 с),
# Манн–Уитни — до 100 наблюдений в сумме (50 + 50: 0.06 с; 75 + 75: 0.29 с;
# 100 + 100: 0.91 с). Запас — на медленные машины. Перебор не годится:
# scipy.stats.permutation_test на C(20, 10) = 184 756 разбиениях — 19 с.
MAX_SIGNED_RANK_N = 200
MAX_RANK_SUM_N = 100

_TOLERANCE = 1e-9


def _doubled_ranks(values: np.ndarray) -> np.ndarray:
    """Удвоенные средние ранги — целые числа (ранг 2.5 → 5)."""
    return np.rint(2 * stats.rankdata(values)).astype(np.int64)


def _signed_rank_distribution(ranks2: np.ndarray) -> np.ndarray:
    """P(2R⁺ = s), s = 0…Σ: каждый ранг входит в R⁺ с вероятностью ½."""
    dist = np.zeros(int(ranks2.sum()) + 1)
    dist[0] = 1.0
    for r in ranks2:
        shifted = np.zeros_like(dist)
        shifted[r:] = dist[:-r]
        dist = 0.5 * (dist + shifted)
    return dist


def _rank_sum_distribution(ranks2: np.ndarray, size: int) -> np.ndarray:
    """P(2W = s) для суммы рангов группы из ``size`` наблюдений.

    table[j, s] — число способов выбрать j наблюдений с удвоенной суммой s.
    """
    table = np.zeros((size + 1, int(ranks2.sum()) + 1))
    table[0, 0] = 1.0
    for r in ranks2:
        table[1:, r:] += table[:-1, :-r].copy()
    return table[size] / table[size].sum()


def _tail(dist: np.ndarray, observed: int, mean: float, alternative: str) -> float:
    """Доля распределения не менее крайняя, чем наблюдённая сумма."""
    sums = np.arange(dist.size)
    if alternative == "greater":
        keep = sums >= observed
    elif alternative == "less":
        keep = sums <= observed
    else:
        keep = np.abs(sums - mean) >= abs(observed - mean) - _TOLERANCE
    return float(min(1.0, dist[keep].sum()))


def signed_rank_p(diff: np.ndarray, alternative: str = "two-sided") -> float | None:
    """Точное P знаково-рангового теста; нулевые разности отброшены.

    ``greater`` — разности смещены вверх (R⁺ велика). None — выборка больше
    предела, нужно нормальное приближение.
    """
    nonzero = np.asarray(diff, dtype=float)
    nonzero = nonzero[nonzero != 0]
    if not 0 < nonzero.size <= MAX_SIGNED_RANK_N:
        return None
    ranks2 = _doubled_ranks(np.abs(nonzero))
    observed = int(ranks2[nonzero > 0].sum())
    return _tail(_signed_rank_distribution(ranks2), observed, ranks2.sum() / 2, alternative)


def rank_sum_p(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> float | None:
    """Точное P Манна–Уитни по сумме рангов группы a.

    ``greater`` — группа a стохастически больше. None — выборка больше предела.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 1 or n_b < 1 or n_a + n_b > MAX_RANK_SUM_N:
        return None
    ranks2 = _doubled_ranks(np.concatenate([a, b]))
    if n_a > n_b:  # таблица по меньшей группе: W_b = Σ − W_a
        flipped = {"greater": "less", "less": "greater"}.get(alternative, alternative)
        return _rank_sum_tail(ranks2, ranks2[n_a:], flipped)
    return _rank_sum_tail(ranks2, ranks2[:n_a], alternative)


def _rank_sum_tail(ranks2: np.ndarray, group: np.ndarray, alternative: str) -> float:
    """Хвост распределения суммы рангов группы ``group`` среди всех ``ranks2``."""
    mean = group.size * ranks2.sum() / ranks2.size
    dist = _rank_sum_distribution(ranks2, group.size)
    return _tail(dist, int(group.sum()), mean, alternative)
