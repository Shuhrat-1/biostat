"""Общие проверки вырожденных данных: нет вариации, нет остаточной вариации.

Отказ — запись в результате с кодом (его переводит фронтенд), а не
исключение и не NaN в таблице. Пороги относительные: результат анализа
не зависит от единиц измерения (моль/л против мкмоль/л).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from core.report.schema import Severity, StatResult

# Доля общей SS, ниже которой остаточная SS — ошибка округления, а не вариация.
RELATIVE_ZERO = 1e-12


def is_constant(values: Iterable[float]) -> bool:
    """Все значения (без пропусков) одинаковы или значений нет."""
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    return v.size == 0 or float(np.ptp(v)) == 0.0


def no_residual(rss: float, total: float) -> bool:
    """Остаточная SS — ноль с точностью до округления (относительно общей SS)."""
    return total <= 0 or rss <= RELATIVE_ZERO * total


def refuse_constant(result: StatResult, values: Iterable[float]) -> bool:
    """Записать отказ, если отклик не варьирует; True — метод должен остановиться."""
    if not is_constant(values):
        return False
    result.add_warning("no_variation", "Значения не варьируют (все одинаковы или все "
                       "разности равны нулю) — анализировать нечего.", Severity.ERROR)
    return True


def refuse_degenerate_line(result: StatResult, x: Iterable[float], y: Iterable[float]) -> bool:
    """Отказ, если y постоянен или лежит точно на прямой от x; True — остановиться.

    На таких данных остаточная SS — ошибка округления, и F, t, P из неё —
    шум (F = −22, t = 3e15 в находке 02-04), а не результат.
    """
    xa, ya = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    keep = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[keep], ya[keep]
    if refuse_constant(result, ya):
        return True
    xc, yc = xa - xa.mean(), ya - ya.mean()
    sxx, syy = float(xc @ xc), float(yc @ yc)
    if sxx == 0:
        return False
    return refuse_no_residual(result, syy - float(xc @ yc) ** 2 / sxx, syy)


def refuse_no_residual(result: StatResult, rss: float, total: float) -> bool:
    """Записать отказ, если модель описывает данные точно; True — остановиться."""
    if not no_residual(rss, total):
        return False
    result.add_warning("no_residual_variation", "Остаточной вариации нет: модель описывает "
                       "данные точно, ошибку оценить нельзя.", Severity.ERROR)
    return True


def note_single_groups(result: StatResult, sizes: Sequence[int],
                       labels: Sequence[str]) -> None:
    """Предупредить о группах из одного наблюдения (аудит 02-11).

    Такая группа входит в расчёт своей средней, но в оценку остаточной
    дисперсии ничего не вносит: остаточные df = N − t.
    """
    single = [str(lab) for lab, n in zip(labels, sizes) if n == 1]
    if single:
        result.add_warning(
            "single_observation_group",
            f"В группе одно наблюдение ({', '.join(single)}): её среднее не оценивает "
            "разброс, остаточная дисперсия — по остальным группам.",
            Severity.WARNING, params={"groups": ", ".join(single)})


def check_group_sizes(result: StatResult, sizes: Sequence[int], labels: Sequence[str]) -> bool:
    """Хватает ли повторностей для остаточной дисперсии (N − t ≥ 1); False — отказ.

    Группа из одного наблюдения расчёту не мешает — о ней предупреждение.
    """
    if any(n < 1 for n in sizes):
        result.add_warning("empty_group", "Все группы должны быть непусты", Severity.ERROR)
        return False
    if sum(sizes) - len(sizes) < 1:
        result.add_warning("n_too_small", "Нет степеней свободы остатка: нужны повторности",
                           Severity.ERROR)
        return False
    note_single_groups(result, sizes, labels)
    return True
