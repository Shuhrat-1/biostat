"""Аудит 4, часть Б: цветовые токены (`web/src/styles.ts`) — контраст WCAG 2.2 и
различимость при дальтонизме; 04-06 и 04-07 — цвета и подписи графиков."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from tests.color_math import VIEWS, contrast, delta_e, lab

WEB = Path(__file__).resolve().parents[1] / "web" / "src"
STYLES = (WEB / "styles.ts").read_text(encoding="utf-8")
WHITE = "#ffffff"


def _block(name: str) -> str:
    start = STYLES.index(f"export const {name} = {{")
    return STYLES[start:STYLES.index("} as const;", start)]


def _token(block: str, key: str) -> str:
    return re.search(rf'\b{key}: "(#[0-9a-f]{{6}})"', _block(block)).group(1)


def _list(key: str) -> list[str]:
    raw = re.search(rf"{key}: \[([^\]]*)\]", _block("chart")).group(1)
    return re.findall(r'"(#[0-9a-f]{6})"', raw)


CATEGORICAL = _list("categorical")


# --- Текст и элементы интерфейса --------------------------------------------------------

@pytest.mark.parametrize(("fg", "bg", "need"), [
    (("c", "text"), ("c", "panel"), 4.5), (("c", "muted"), ("c", "panel"), 4.5),
    (("c", "muted"), ("c", "subtle"), 4.5), (("c", "muted"), ("c", "bg"), 4.5),
    (("c", "accent"), ("c", "panel"), 4.5), (("c", "accent"), ("c", "selectedBg"), 4.5),
    (("c", "ok"), ("c", "okBg"), 4.5), (("c", "warn"), ("c", "warnBg"), 4.5),
    (("c", "bad"), ("c", "badBg"), 4.5), (("c", "info"), ("c", "infoBg"), 4.5),
    (("chart", "axis"), ("c", "panel"), 4.5), (("chart", "ref"), ("c", "panel"), 4.5),
    (("chart", "outlier"), ("c", "panel"), 4.5),
    (("c", "borderInput"), ("c", "panel"), 3.0),   # рамка поля: было 1.27 (04-06)
    (("mapEdge", "strong"), ("c", "panel"), 3.0), (("mapEdge", "hint"), ("c", "panel"), 3.0),
])
def test_token_contrast_meets_wcag(fg: tuple[str, str], bg: tuple[str, str], need: float) -> None:
    assert contrast(_token(*fg), _token(*bg)) >= need


def test_categorical_palette_is_eight_colours_starting_from_the_accent() -> None:
    assert len(CATEGORICAL) == 8 and CATEGORICAL[0] == _token("c", "accent")
    assert all(contrast(h, WHITE) >= 3.0 for h in CATEGORICAL)  # WCAG 1.4.11


@pytest.mark.parametrize("view", VIEWS)
def test_categorical_colours_stay_apart_for_colour_blind_viewers(view: str | None) -> None:
    """Было: при дейтанопии 4 из 10 цветов сливались (min ΔE 0.2). Стало ≥ 16.8."""
    worst = min(delta_e(a, b, view) for a, b in itertools.combinations(CATEGORICAL, 2))
    assert worst >= 16.0


@pytest.mark.parametrize("view", ["протан", "дейтан"])
def test_map_outlines_are_distinguishable(view: str) -> None:
    """Контур «указывают данные» / «подсказка»: было ΔE 7.5 / 11.8."""
    assert delta_e(_token("mapEdge", "strong"), _token("mapEdge", "hint"), view) >= 20


def test_sequential_scale_gets_darker_evenly_in_every_view() -> None:
    seq = _list("sequential")
    for view in VIEWS:
        lightness = [lab(h, view)[0] for h in seq]
        steps = [a - b for a, b in zip(lightness, lightness[1:])]
        assert all(s > 5 for s in steps), (view, lightness)


def test_diverging_scale_has_balanced_ends_and_a_light_middle() -> None:
    div = _list("diverging")
    ends = [lab(div[0])[0], lab(div[-1])[0]]
    assert len(div) == 5 and abs(ends[0] - ends[1]) < 6 and lab(div[2])[0] > 95
    assert all(delta_e(div[0], div[-1], v) > 30 for v in VIEWS)


# --- Где цвета применены -----------------------------------------------------------------

def test_no_colour_literals_left_in_charts_map_and_app() -> None:
    for name in ("charts.tsx", "SensoryMap.tsx", "App.tsx"):
        text = (WEB / name).read_text(encoding="utf-8")
        assert re.findall(r'"#[0-9a-fA-F]{3,8}"', text) == [], name


def test_red_marks_only_outliers() -> None:
    """Красный — только выброс: линия подгонки, полоса и границы эквивалентности — нет (04-07)."""
    charts = (WEB / "charts.tsx").read_text(encoding="utf-8")
    uses = [line for line in charts.splitlines() if "chart.outlier" in line]
    assert len(uses) == 3 and all("outlier" in u or "symbolSize: 6" in u for u in uses)


def test_group_keeps_one_colour_across_the_result() -> None:
    charts = (WEB / "charts.tsx").read_text(encoding="utf-8")
    assert "function groupPalette(bundle: PlotBundle)" in charts
    assert charts.count("colorOf(") >= 3  # боксплот, средние ± SE, средние с ДИ
    assert "seriesStyle(" in charts and "groupColor(" in charts
    assert "if (i < 0 || nGroups > chart.categorical.length) return c.accent;" in STYLES


def test_04_06_faded_map_nodes_use_text_colour_not_opacity() -> None:
    sensory = (WEB / "SensoryMap.tsx").read_text(encoding="utf-8")
    assert "opacity: dim" not in sensory and "color: onFill ? c.accent : dim ? c.muted : c.text" in sensory
    assert "opacity: 0.5" not in sensory


def test_04_07_axes_carry_column_names() -> None:
    charts = (WEB / "charts.tsx").read_text(encoding="utf-8")
    assert 'name: "X"' not in charts and 'name: "Y"' not in charts
    bridge = (WEB / "pyodide" / "bridge.ts").read_text(encoding="utf-8")
    assert '_bundle["axis_names"] = {"x": _cols.get("x") or _cols.get("covariate"),' in bridge


def test_04_07_bridge_sends_axis_names_for_a_regression() -> None:
    from tests.bridge_runner import CASES, plots, run, table

    run("simple_linear_regression", table("simple_linear_regression"))
    roles = CASES["simple_linear_regression"][1]
    bundle = plots("simple_linear_regression")
    assert bundle["axis_names"] == {"x": roles["x"], "y": roles["y"]}
    run("one_way_anova", table("one_way_anova"))
    assert plots("one_way_anova")["axis_names"]["y"] == CASES["one_way_anova"][1]["value"]
