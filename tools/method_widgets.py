"""Виджеты страниц методов: статичный SVG рисует генератор, скрипт только оживляет.

groups-split — точки групп, средние, общая средняя и отрезки «между / внутри» у
выделенной точки. Без JS виден график в исходном положении и итог расчёта ядра;
web/public/methods/assets/groups-split.js двигает те же элементы ползунками.
Геометрия (GEOMETRY, шаг точек) — одна с layout() в скрипте; тест сверяет позиции.
"""

from __future__ import annotations

import html
import json
import math

GEOMETRY = {"W": 560, "H": 330, "L": 50, "R": 20, "T": 20, "B": 40}
DOT_STEP, DOT_OFFSET, MEAN_HALF = 16, 40, 55
COLORS = ("var(--g1)", "var(--g2)", "var(--g3)", "var(--g4)", "var(--g5)", "var(--g6)")


def _r(v: float) -> str:
    """Координата SVG: 2 знака без хвостовых нулей."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s != "-0" else "0"


class Frame:
    """Шкалы графика: y → пиксели, номер группы → x."""

    def __init__(self, y_min: float, y_max: float, n_groups: int) -> None:
        g = GEOMETRY
        self.y_min, self.y_max, self.n = y_min, y_max, n_groups
        self.plot_h, self.plot_w = g["H"] - g["T"] - g["B"], g["W"] - g["L"] - g["R"]

    def sy(self, v: float) -> float:
        v = min(self.y_max, max(self.y_min, v))
        return GEOMETRY["T"] + (self.y_max - v) / (self.y_max - self.y_min) * self.plot_h

    def col_x(self, i: int) -> float:
        return GEOMETRY["L"] + (i + 0.5) * self.plot_w / self.n


def _axes(frame: Frame, cfg: dict, axis: str, names: list[str]) -> str:
    g, out = GEOMETRY, []
    v = math.ceil(cfg["y_min"] / cfg["y_step"]) * cfg["y_step"]
    while v <= cfg["y_max"] + 1e-9:
        y = _r(frame.sy(v))
        out.append(f'<line x1="{g["L"]}" x2="{g["W"] - g["R"]}" y1="{y}" y2="{y}" stroke="#e4e4e7"/>'
                   f'<text x="{g["L"] - 8}" y="{_r(frame.sy(v) + 4)}" text-anchor="end" font-size="12" '
                   f'fill="#52525b">{v:g}</text>')
        v += cfg["y_step"]
    mid = _r(g["T"] + frame.plot_h / 2)
    out.append(f'<text x="14" y="{mid}" font-size="12" fill="#52525b" transform="rotate(-90 14 {mid})" '
               f'text-anchor="middle">{html.escape(axis)}</text>')
    out += [f'<text x="{_r(frame.col_x(i))}" y="{g["H"] - 12}" text-anchor="middle" font-size="13" '
            f'fill="#18181b">{html.escape(n)}</text>' for i, n in enumerate(names)]
    return "".join(out)


def _marks(frame: Frame, groups: list[tuple[str, list[float]]], highlight: list[int]) -> str:
    """Общая средняя, средние групп, отрезки у выделенной точки, точки."""
    g = GEOMETRY
    allv = [v for _, ys in groups for v in ys]
    grand = sum(allv) / len(allv)
    means = [sum(ys) / len(ys) for _, ys in groups]
    gy = _r(frame.sy(grand))
    out = [f'<line class="gs-grand" x1="{g["L"]}" x2="{g["W"] - g["R"]}" y1="{gy}" y2="{gy}" '
           f'stroke="var(--grand)" stroke-dasharray="6 4" stroke-width="1.5"/>']
    for i, m in enumerate(means):
        x, y = frame.col_x(i), _r(frame.sy(m))
        out.append(f'<line class="gs-mean" data-g="{i}" x1="{_r(x - MEAN_HALF)}" x2="{_r(x + MEAN_HALF)}" '
                   f'y1="{y}" y2="{y}" stroke="#18181b" stroke-width="2.5"/>')
    hg, hj = highlight
    hx = frame.col_x(hg) - DOT_OFFSET + hj * DOT_STEP
    out.append(f'<line class="gs-seg-b" x1="{_r(hx + 12)}" x2="{_r(hx + 12)}" y1="{gy}" '
               f'y2="{_r(frame.sy(means[hg]))}" stroke="#c2410c" stroke-width="3"/>')
    out.append(f'<line class="gs-seg-w" x1="{_r(hx)}" x2="{_r(hx)}" y1="{_r(frame.sy(means[hg]))}" '
               f'y2="{_r(frame.sy(groups[hg][1][hj]))}" stroke="#7c3aed" stroke-width="3"/>')
    for i, (_, ys) in enumerate(groups):
        for j, v in enumerate(ys):
            hl = (i, j) == (hg, hj)
            out.append(f'<circle class="gs-dot" data-g="{i}" data-j="{j}" cx="{_r(frame.col_x(i) - DOT_OFFSET + j * DOT_STEP)}" '
                       f'cy="{_r(frame.sy(v))}" r="{7.5 if hl else 5.5}" fill="{COLORS[i]}" fill-opacity="0.85" '
                       f'stroke="{"#18181b" if hl else "#fff"}" stroke-width="1"/>')
    return "".join(out)


def _controls(t: dict, values: dict) -> str:
    """Ползунки (скрыты без JS), полосы MS и итог: исходное положение — расчёт ядра."""
    esc = html.escape
    msb, msw = values["ms_between"], values["ms_within"]
    top = max(msb, msw, 1e-9)
    p = "< 0.001" if values["p"] < 0.001 else f'{values["p"]:.3f}'
    f_text = "> 999" if values["F"] > 999 else f'{values["F"]:.1f}'
    rows = [(t["ss_between"], "gs-ssb", f'{values["ss_between"]:.2f}'), (t["ss_within"], "gs-ssw", f'{values["ss_within"]:.2f}'),
            (f'F({values["df_between"]}, {values["df_within"]})', "gs-f", f_text), ("P", "gs-p", p)]
    readout = "".join(f'<div><span>{esc(a)}</span><b class="{c}">{v}</b></div>' for a, c, v in rows)
    return (
        f'<div class="ctrl"><div class="gs-sliders" hidden>'
        f'<label for="kb">{esc(t["kb"])}</label><input id="kb" class="gs-kb" type="range" min="0" max="1.5" step="0.05" value="1">'
        f'<label for="kw">{esc(t["kw"])}</label><input id="kw" class="gs-kw" type="range" min="0.3" max="3" step="0.05" value="1"></div>'
        f'<div class="bars" aria-hidden="true"><div class="note">{esc(t["ms_between"])}</div>'
        f'<div class="bar gs-bar-b" style="width:{_r(100 * msb / top)}%;background:#c2410c"></div>'
        f'<div class="note">{esc(t["ms_within"])}</div>'
        f'<div class="bar gs-bar-w" style="width:{_r(max(1, 100 * msw / top))}%;background:#7c3aed"></div></div>'
        f'<div class="readout" aria-live="polite">{readout}</div>'
        f'<p class="note gs-note" hidden>{esc(t["note"])}</p></div>')


def groups_split(cfg: dict, t: dict, example: dict, values: dict) -> str:
    """HTML виджета «между / внутри» на данных примера."""
    esc, groups = html.escape, values["groups"]
    frame = Frame(cfg["y_min"], cfg["y_max"], len(groups))
    data = json.dumps([{"name": n, "y": ys} for n, ys in groups], ensure_ascii=False)
    attrs = (f'data-widget="groups-split" data-groups="{esc(data)}" '
             f'data-df="{esc(json.dumps([values["df_between"], values["df_within"]]))}" '
             f'data-highlight="{esc(json.dumps(cfg["highlight"]))}" '
             f'data-y="{esc(json.dumps([cfg["y_min"], cfg["y_max"]]))}"')
    g = GEOMETRY
    svg = (f'<svg viewBox="0 0 {g["W"]} {g["H"]}" role="img" aria-label="{esc(t["aria"])}">'
           f'{_axes(frame, cfg, t["axis"], [n for n, _ in groups])}{_marks(frame, groups, cfg["highlight"])}</svg>')
    legend = (f'<div class="legend"><span><i style="border-color:var(--grand);border-top-style:dashed"></i>'
              f'{esc(t["legend_grand"])}</span><span><i style="border-color:#18181b"></i>{esc(t["legend_mean"])}</span>'
              f'<span>{esc(t["legend_segments"])} <b style="color:#c2410c">{esc(t["between"])}</b> {esc(t["and"])} '
              f'<b style="color:#7c3aed">{esc(t["within"])}</b></span></div>')
    return f'    <div class="viz" {attrs}>\n      <div>{svg}{legend}</div>\n      {_controls(t, values)}\n    </div>\n'
