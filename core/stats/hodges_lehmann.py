"""Оценки сдвига Ходжеса–Лемана с ДИ [предл.: Hodges & Lehmann 1963].

В scipy/statsmodels нет (инструкция W2) — реализовано вручную:
- две независимые выборки: медиана всех n₁·n₂ разностей aᵢ − bⱼ; ДИ — порядковые
  статистики разностей, номер берётся из распределения U Манна–Уитни;
- одна выборка разностей (парный случай): медиана средних Уолша (dᵢ + dⱼ)/2, i ≤ j;
  ДИ — из распределения T⁺ знаково-рангового критерия.

Точное нулевое распределение (динамика по числу способов) — если нет совпадений
и выборка невелика; иначе нормальное приближение без поправки на совпадения
[предл.: Hollander & Wolfe 1999, гл. 3–4]. Уровень покрытия точного ДИ ≥ 1 − α
(дискретность) — его фактическое значение возвращается. Если выборка так мала, что
даже крайняя конфигурация не значима на уровне α, границы ДИ — None.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_EXACT_MAX_PAIRS = 2500  # n₁·n₂ для точного распределения U
_EXACT_MAX_N = 50  # n для точного распределения T⁺


def mw_null_counts(m: int, n: int) -> np.ndarray:
    """Число способов получить U = 0…mn (U — число пар aᵢ > bⱼ), без совпадений."""
    # f[i][j] — распределение для (i, j); f(i, j, u) = f(i − 1, j, u − j) + f(i, j − 1, u).
    prev = [np.ones(1, dtype=float) for _ in range(n + 1)]  # i = 0: U = 0 единственно
    for i in range(1, m + 1):
        cur = [np.ones(1, dtype=float)]  # j = 0
        for j in range(1, n + 1):
            out = np.zeros(i * j + 1)
            a = prev[j]
            out[j:j + a.size] += a
            b = cur[j - 1]
            out[:b.size] += b
            cur.append(out)
        prev = cur
    return prev[n]


def signrank_null_counts(n: int) -> np.ndarray:
    """Число подмножеств {1…n} с суммой T⁺ = 0…n(n+1)/2."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for k in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[k:] = counts[:-k]
        counts = counts + shifted
    return counts


def _exact_rank(counts: np.ndarray, alpha: float) -> tuple[int, float]:
    """Наибольшее k ≥ 1 с P(S ≤ k − 1) ≤ α/2 и фактическое покрытие 1 − 2·P(S ≤ k − 1)."""
    cdf = np.cumsum(counts) / counts.sum()
    k = int(np.searchsorted(cdf, alpha / 2, side="right"))  # число значений с cdf ≤ α/2
    if k == 0:  # даже P(S = 0) > α/2: выборка слишком мала, ДИ не ограничен
        return 0, 1.0
    return k, 1 - 2 * float(cdf[k - 1])


def _normal_rank(mean: float, sd: float, alpha: float) -> int:
    """k ≈ среднее − z·SD (округление вниз); 0 — ДИ не ограничен."""
    return max(int(np.floor(mean - stats.norm.ppf(1 - alpha / 2) * sd)), 0)


def _interval(sorted_values: np.ndarray, k: int) -> tuple[float | None, float | None]:
    """k-я снизу и k-я сверху порядковые статистики (нумерация с 1); k = 0 — None, None."""
    if k < 1:
        return None, None
    k = min(k, (sorted_values.size + 1) // 2)
    return float(sorted_values[k - 1]), float(sorted_values[-k])


def hl_two_sample(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict[str, float | str]:
    """Сдвиг a − b: оценка, ДИ, метод ДИ и (для точного) фактическое покрытие."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m, n = a.size, b.size
    diffs = np.sort((a[:, None] - b[None, :]).ravel())
    out: dict[str, float | str] = {"hl_shift": float(np.median(diffs))}
    pooled = np.concatenate([a, b])
    if np.unique(pooled).size == pooled.size and m * n <= _EXACT_MAX_PAIRS:
        k, cover = _exact_rank(mw_null_counts(m, n), alpha)
        out.update({"hl_ci_method": "exact", "hl_ci_coverage": cover})
    else:
        k = _normal_rank(m * n / 2, np.sqrt(m * n * (m + n + 1) / 12), alpha)
        out.update({"hl_ci_method": "normal", "hl_ci_coverage": 1 - alpha})
    out["hl_ci_lower"], out["hl_ci_upper"] = _interval(diffs, k)
    return out


def walsh_averages(d: np.ndarray) -> np.ndarray:
    """Средние Уолша (dᵢ + dⱼ)/2 для i ≤ j, по возрастанию."""
    d = np.asarray(d, float)
    i, j = np.triu_indices(d.size)
    return np.sort((d[i] + d[j]) / 2)


def hl_one_sample(d: np.ndarray, alpha: float = 0.05) -> dict[str, float | str]:
    """Центр симметрии разностей: медиана средних Уолша, ДИ по T⁺.

    Используются все разности (и нулевые) — так определена оценка; тест Уилкоксона
    нули отбрасывает.
    """
    d = np.asarray(d, float)
    n = d.size
    walsh = walsh_averages(d)
    out: dict[str, float | str] = {"hl_shift": float(np.median(walsh))}
    absd = np.abs(d)
    if np.unique(absd).size == n and np.all(absd > 0) and n <= _EXACT_MAX_N:
        k, cover = _exact_rank(signrank_null_counts(n), alpha)
        out.update({"hl_ci_method": "exact", "hl_ci_coverage": cover})
    else:
        mean = n * (n + 1) / 4
        k = _normal_rank(mean, np.sqrt(n * (n + 1) * (2 * n + 1) / 24), alpha)
        out.update({"hl_ci_method": "normal", "hl_ci_coverage": 1 - alpha})
    out["hl_ci_lower"], out["hl_ci_upper"] = _interval(walsh, k)
    return out
