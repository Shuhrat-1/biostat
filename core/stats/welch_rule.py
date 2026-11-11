"""Стьюдент или Уэлч для двух групп: правило по отношению SD и балансу n.

Тест на равенство дисперсий как «переключатель» искажает итоговый α и
слаб при малых n [предл.: Zimmerman 2004; Ruxton 2006; Delacre et al. 2017].
Важен не P Левене, а сочетание неравных SD с неравными n: при большей
дисперсии в меньшей группе P Стьюдента занижен (тест либерален), при
большей дисперсии в большей группе — завышен (консервативен).

Пороги — BioStat [предл.: Ruxton 2006; Zimmerman 2004], в книгах их нет:
n уравновешены (отношение ≤ 1.5) — совет Уэлча при отношении SD > 2;
n неуравновешены — при отношении SD > 1.5. Общее для ядра и карты (B4.2).

При n < 8 в группе отношение выборочных SD ненадёжно: при равных дисперсиях
оно превышает 2 в 40 % случаев при n = 3 и в 22 % при n = 5 (имитация,
аудит 01-06), поэтому совет помечается ``reliable = False`` и подаётся как
справка, а не как предупреждение. Порог 8 — BioStat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SD_RATIO_BALANCED = 2.0
SD_RATIO_UNBALANCED = 1.5
N_RATIO_UNBALANCED = 1.5
MIN_N_RELIABLE = 8


@dataclass(frozen=True)
class WelchAdvice:
    """Совет Уэлча: отношения SD и n (большее к меньшему) и случай.

    case: "welch" — n уравновешены; "welch_unbalanced" — большая SD в
    большей группе (Стьюдент консервативен); "welch_liberal" — большая SD
    в меньшей группе (P Стьюдента занижен).
    """

    sd_ratio: float
    n_ratio: float
    case: str
    n_min: int = MIN_N_RELIABLE

    @property
    def reliable(self) -> bool:
        """В меньшей группе хватает наблюдений, чтобы верить отношению SD."""
        return self.n_min >= MIN_N_RELIABLE


def welch_advice(a: np.ndarray, b: np.ndarray) -> WelchAdvice | None:
    """Совет Уэлча или None (Стьюдент уместен; вырожденные группы — молчим)."""
    if a.size < 2 or b.size < 2:
        return None
    sd_a, sd_b = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    if min(sd_a, sd_b) <= 0:
        return None
    sd_ratio = max(sd_a, sd_b) / min(sd_a, sd_b)
    n_min = int(min(a.size, b.size))
    n_ratio = max(a.size, b.size) / n_min
    if n_ratio <= N_RATIO_UNBALANCED:
        if sd_ratio <= SD_RATIO_BALANCED:
            return None
        return WelchAdvice(sd_ratio, n_ratio, "welch", n_min)
    if sd_ratio <= SD_RATIO_UNBALANCED:
        return None
    larger_sd_in_smaller = (sd_a > sd_b) == (a.size < b.size)
    case = "welch_liberal" if larger_sd_in_smaller else "welch_unbalanced"
    return WelchAdvice(sd_ratio, n_ratio, case, n_min)
