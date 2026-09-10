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


def histogram_data(values: list[float], bins: str | int = "auto") -> dict[str, Any]:
    """Данные гистограммы: границы бинов и частоты.

    Число бинов по умолчанию выбирается правилом numpy (Фридмана-Дьякониса
    и др.), что адекватно для большинства биологических распределений.
    Добавляет плотность нормального распределения для наложения кривой.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"bins": [], "counts": [], "normal_curve": None}

    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    normal_curve = None
    if arr.size >= 2 and arr.std(ddof=1) > 0:
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        xs = np.linspace(float(arr.min()), float(arr.max()), 100)
        # Плотность в масштабе частот: * n * ширина бина.
        bin_width = float(edges[1] - edges[0])
        ys = stats.norm.pdf(xs, mean, std) * arr.size * bin_width
        normal_curve = {"x": xs.tolist(), "y": ys.tolist()}

    return {
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "counts": counts.tolist(),
        "normal_curve": normal_curve,
    }


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
    """Средние по группам со стандартной ошибкой — для столбчатого графика.

    Планки погрешности показывают ±SE, дополняя боксплот: боксплот про
    разброс данных, столбцы со средними — про точность оценки среднего.
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
