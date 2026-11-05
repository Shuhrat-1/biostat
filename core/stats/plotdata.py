"""Подготовка числовых рядов для графиков (гл. 2, 5 книги).

Ядро готовит данные, фронтенд (ECharts) рисует. Такое разделение держит
статистические модули чистыми и позволяет считать бины гистограммы в
numpy, а не в JS.

Каждая функция возвращает словарь, готовый к JSON-сериализации и передаче
в конкретный тип графика.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


MIN_CLASSES, MAX_CLASSES = 5, 20


def nice_edges(arr: np.ndarray) -> np.ndarray:
    """Границы классов «удобные» (Witte, стр. 25, 27): ширина 1 / 2 / 5 × 10^k,
    границы кратны ширине. Число классов — по правилу numpy «auto», в пределах
    5–20 (у Witte ~10 для ручных таблиц; на больших выборках классов больше).
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.array([lo - 0.5, lo + 0.5])
    k = np.clip(np.histogram_bin_edges(arr, bins="auto").size - 1, MIN_CLASSES, MAX_CLASSES)
    raw = (hi - lo) / k
    mag = 10.0 ** np.floor(np.log10(raw))
    width = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw - 1e-12 * mag)
    start = np.floor(lo / width) * width
    n = int(np.floor((hi - start) / width + 1e-9)) + 1  # класс, куда попадает max
    return np.round(start + width * np.arange(n + 1), 12)


def histogram_data(values: list[float], bins: str | int = "nice") -> dict[str, Any]:
    """Данные гистограммы: границы классов, частоты, нормальная кривая.

    По умолчанию классы «удобные» (:func:`nice_edges`); ``bins`` — число или
    правило numpy, если нужна прежняя разбивка.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"bin_edges": [], "bin_centers": [], "counts": [], "normal_curve": None}

    counts, edges = np.histogram(arr, bins=nice_edges(arr) if bins == "nice" else bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "counts": counts.tolist(),
        "normal_curve": _normal_curve(arr, float(edges[1] - edges[0])),
    }


def _normal_curve(arr: np.ndarray, bin_width: float) -> dict[str, list] | None:
    """Плотность нормального распределения в масштабе частот (× n × ширина класса)."""
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return None
    mean, std = float(arr.mean()), float(arr.std(ddof=1))
    xs = np.linspace(float(arr.min()), float(arr.max()), 100)
    ys = stats.norm.pdf(xs, mean, std) * arr.size * bin_width
    return {"x": xs.tolist(), "y": ys.tolist()}


def boxplot_data(groups: dict[str, list[float]]) -> dict[str, Any]:
    """Данные боксплота по группам: квартили, усы, выбросы.

    Границы усов — по правилу 1.5*IQR (Тьюки), точки за ними помечаются
    как выбросы. Формат совпадает с тем, что ждёт боксплот ECharts.
    """
    labels = []
    boxes = []
    outliers = []

    for i, (label, values) in enumerate(groups.items()):
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue

        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        inside = arr[(arr >= low_fence) & (arr <= high_fence)]
        whisker_low = float(inside.min()) if inside.size else float(arr.min())
        whisker_high = float(inside.max()) if inside.size else float(arr.max())

        labels.append(label)
        # Порядок ECharts: [min, Q1, median, Q3, max].
        boxes.append([whisker_low, float(q1), float(median), float(q3), whisker_high])
        for value in arr[(arr < low_fence) | (arr > high_fence)]:
            outliers.append([i, float(value)])

    return {"labels": labels, "boxes": boxes, "outliers": outliers}


def scatter_regression_data(
    x: list[float], y: list[float], slope: float, intercept: float
) -> dict[str, Any]:
    """Данные scatter-графика с линией регрессии.

    Точки плюс две крайние точки линии (по min и max x) — этого хватает
    ECharts, чтобы провести прямую.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    mask = ~(np.isnan(ax) | np.isnan(ay))
    ax, ay = ax[mask], ay[mask]
    if ax.size == 0:
        return {"points": [], "line": []}

    x_min, x_max = float(ax.min()), float(ax.max())
    line = [
        [x_min, intercept + slope * x_min],
        [x_max, intercept + slope * x_max],
    ]
    points = [[float(xv), float(yv)] for xv, yv in zip(ax, ay)]
    return {"points": points, "line": line}


def bar_means_data(
    groups: dict[str, list[float]], ci_level: float = 0.95
) -> dict[str, Any]:
    """Средние по группам со стандартной ошибкой — для графика «среднее ± SE».

    Рисуется точками с планками, не столбцами (Witte 2.11, Fig. 14.5).
    Дополняет боксплот: боксплот — про разброс данных, точки — про точность
    оценки среднего.
    """
    labels = []
    means = []
    errors = []

    for label, values in groups.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        labels.append(label)
        means.append(float(arr.mean()))
        errors.append(float(stats.sem(arr)) if arr.size >= 2 else 0.0)

    return {"labels": labels, "means": means, "errors": errors}


def qq_data(values: list[float]) -> dict[str, Any] | None:
    """Нормальный Q-Q: позиции Блома против стандартизованных значений.

    Та же шкала, что у Q-Q остатков моделей (опорная линия y = x): точки на
    линии — нормальность; дуга — асимметрия; S-образность — тяжёлые хвосты.
    Главная проверка нормальности по Welham (гл. 5.2), тест Шапиро — дополнение.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3 or arr.std(ddof=1) == 0:
        return None
    n = arr.size
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    z = np.sort((arr - arr.mean()) / arr.std(ddof=1))
    return {"theoretical_quantiles": blom.tolist(), "sample_quantiles": z.tolist()}
