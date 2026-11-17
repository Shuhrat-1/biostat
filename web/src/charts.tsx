/**
 * Обёртка ECharts и построение опций под статистические графики.
 *
 * Импортируем только нужные модули ECharts (не весь пакет) — это держит
 * бандл в разумных пределах. Каждая функция строит опции для своего типа
 * графика из данных, подготовленных ядром.
 */

import * as echarts from "echarts/core";
import {
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
} from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { SVGRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";
import { escapeHtml as esc } from "./escape";
import { useT } from "./i18n";
import type { PlotBundle } from "./pyodide/bridge";
import { c, chart, groupColor, seriesStyle } from "./styles";

echarts.use([
  BarChart,
  BoxplotChart,
  ScatterChart,
  LineChart,
  CustomChart,
  HeatmapChart,
  GridComponent,
  VisualMapComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  MarkLineComponent,
  SVGRenderer,
]);

const BASE_GRID = { left: 48, right: 20, top: 40, bottom: 40 };
const AXIS_STYLE = {
  axisLine: { lineStyle: { color: chart.axisLine } },
  axisLabel: { color: chart.axis },
  // Имя оси по умолчанию берёт цвет линии оси (1.27:1 на белом) — задаём явно (аудит 04-06).
  nameTextStyle: { color: chart.axis },
  splitLine: { lineStyle: { color: chart.grid } },
};

/** Один график ECharts. Пересоздаёт опции при смене данных. */
function Chart({
  option,
  height = 300,
}: {
  option: echarts.EChartsCoreOption;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    instance.current = echarts.init(ref.current, undefined, {
      renderer: "svg",
    });
    const chart = instance.current;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
    };
  }, []);

  useEffect(() => {
    instance.current?.setOption(option, true);
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height }} />;
}

/* ------------------------- Построители опций ------------------------- */

type T = (key: string) => string;
type Option = echarts.EChartsCoreOption;
/** Подписи осей: имя колонки из формы, иначе запасная подпись. */
type AxisNames = NonNullable<PlotBundle["axis_names"]>;

/** Цвет группы по её имени — один на весь результат (аудит 04-07). */
type ColorOf = (label: string, index: number, count: number) => string;

/**
 * Порядок групп берётся из первого графика, где они стоят по оси (боксплот,
 * средние); остальные графики красят ту же группу тем же цветом.
 */
function groupPalette(bundle: PlotBundle): { colorOf: ColorOf; order: string[] } {
  const order = bundle.boxplot?.labels ?? bundle.bar_means?.labels
    ?? (bundle.means_ci && !bundle.means_ci.numeric ? bundle.means_ci.items.map((it) => it.label) : undefined)
    ?? bundle.group_curves?.curves.map((cv) => cv.name) ?? [];
  const colorOf: ColorOf = (label, index, count) => {
    const i = order.indexOf(label);
    return i >= 0 ? groupColor(i, order.length) : groupColor(index, count);
  };
  return { colorOf, order };
}

/** Заливка фигуры группы — тот же цвет, прозрачность 18 %; контур сплошной. */
const fill = (hex: string) => `${hex}2e`;

/**
 * Планка погрешности custom-серии: value = [позиция, низ, верх].
 * vertical — вдоль оси y (средние), иначе вдоль x (forest разностей).
 */
function whisker(api: any, vertical: boolean) {
  const [pos, lo, hi] = [api.value(0), api.value(1), api.value(2)];
  const a = api.coord(vertical ? [pos, lo] : [lo, pos]);
  const b = api.coord(vertical ? [pos, hi] : [hi, pos]);
  const w = 6;
  const cap = (p: number[]) => (vertical
    ? { x1: p[0] - w, y1: p[1], x2: p[0] + w, y2: p[1] }
    : { x1: p[0], y1: p[1] - w, x2: p[0], y2: p[1] + w });
  const style = { stroke: c.text, lineWidth: 1.5 };
  return {
    type: "group",
    children: [
      { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style },
      { type: "line", shape: cap(a), style },
      { type: "line", shape: cap(b), style },
    ],
  };
}

function histogramOption(h: NonNullable<PlotBundle["histogram"]>, t: T, names: AxisNames = {}): Option {
  const bars = h.bin_centers.map((x, i) => [x, h.counts[i]]);
  const series: Record<string, unknown>[] = [
    { type: "bar", data: bars, barWidth: "99%", itemStyle: { color: c.accent, opacity: 0.75 },
      name: t("axis.frequency") },
  ];
  if (h.normal_curve) {
    series.push({
      type: "line", data: h.normal_curve.x.map((x, i) => [x, h.normal_curve!.y[i]]),
      smooth: true, symbol: "none", lineStyle: { color: chart.fit, width: 2 }, name: t("axis.normal"),
    });
  }
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "axis" },
    legend: { top: 8, textStyle: { color: c.muted } },
    xAxis: { type: "value", scale: true, name: names.y ?? undefined, nameLocation: "middle",
      nameGap: 26, ...AXIS_STYLE },
    yAxis: { type: "value", name: t("axis.frequency"), ...AXIS_STYLE },
    series,
  };
}

function boxplotOption(b: NonNullable<PlotBundle["boxplot"]>, colorOf: ColorOf, names: AxisNames = {}): Option {
  const n = b.labels.length;
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "category", data: b.labels, ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: names.y ?? undefined, ...AXIS_STYLE },
    series: [
      { type: "boxplot", data: b.boxes.map((box, i) => {
        const color = colorOf(b.labels[i], i, n);
        return { value: box, itemStyle: { color: fill(color), borderColor: color, borderWidth: 1.5 } };
      }) },
      { type: "scatter", data: b.outliers, symbolSize: 6, itemStyle: { color: chart.outlier } },
    ],
  };
}

/**
 * Средние ± SE точками, а не столбцами (Witte 2.11, Fig. 14.5): столбец с
 * основанием не от нуля искажает пропорции, а от нуля — прячет различия.
 */
function barMeansOption(b: NonNullable<PlotBundle["bar_means"]>, t: T, colorOf: ColorOf): Option {
  const lo = b.means.map((m, i) => m - b.errors[i]);
  const hi = b.means.map((m, i) => m + b.errors[i]);
  const tip = (p: any) => `${esc(b.labels[p.dataIndex])}: ${fmt(b.means[p.dataIndex])}`
    + ` ± ${fmt(b.errors[p.dataIndex])}`;
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item", formatter: tip },
    xAxis: { type: "category", data: b.labels, ...AXIS_STYLE },
    yAxis: { type: "value", name: t("axis.mean"), ...AXIS_STYLE, ...bounds([...lo, ...hi]) },
    series: [
      { type: "custom", data: b.means.map((_, i) => [i, lo[i], hi[i]]),
        renderItem: (_p: unknown, api: any) => whisker(api, true), z: 1 },
      { type: "scatter", symbolSize: 10, z: 2, data: b.means.map((m, i) => ({
        value: [i, m], itemStyle: { color: colorOf(b.labels[i], i, b.labels.length) } })) },
    ],
  };
}

function scatterOption(sc: NonNullable<PlotBundle["scatter"]>, names: AxisNames = {}): Option {
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: names.x ?? "x", nameLocation: "middle", nameGap: 26,
      ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: names.y ?? "y", ...AXIS_STYLE },
    series: [
      { type: "scatter", data: sc.points, symbolSize: 7, itemStyle: { color: c.accent, opacity: 0.6 } },
      // Линия подгонки нейтральная: красный на графиках значит только «выброс».
      { type: "line", data: sc.line, symbol: "none", lineStyle: { color: chart.fit, width: 2 } },
      // Доверительная полоса кривой: две пунктирные границы.
      ...(sc.band ? [sc.band.lower, sc.band.upper] : []).map((data) => ({
        type: "line", data, symbol: "none", silent: true,
        lineStyle: { color: chart.ref, width: 1.5, type: "dashed" },
      })),
    ],
  };
}

function qqOption(qq: NonNullable<PlotBundle["qq"]>, t: T): Option {
  const points = qq.theoretical_quantiles.map((x, i) => [x, qq.sample_quantiles[i]]);
  // Опорная линия y=x в диапазоне теоретических квантилей.
  const lo = Math.min(...qq.theoretical_quantiles);
  const hi = Math.max(...qq.theoretical_quantiles);
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: t("axis.theoretical"), ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: t("axis.sample"), ...AXIS_STYLE },
    series: [
      { type: "scatter", data: points, symbolSize: 6, itemStyle: { color: c.accent, opacity: 0.6 } },
      { type: "line", data: [[lo, lo], [hi, hi]], symbol: "none",
        lineStyle: { color: chart.ref, type: "dashed" } },
    ],
  };
}

function residualOption(r: NonNullable<PlotBundle["residual_vs_fitted"]>, t: T): Option {
  const points = r.fitted.map((f, i) => [f, r.residuals[i]]);
  const series: Record<string, unknown>[] = [{
    type: "scatter",
    data: points.filter((_, i) => i !== r.outlier),
    symbolSize: 6,
    itemStyle: { color: c.accent, opacity: 0.6 },
    markLine: { silent: true, symbol: "none", lineStyle: { color: chart.ref, type: "dashed" },
      data: [{ yAxis: 0 }] },
  }];
  if (r.outlier !== undefined && points[r.outlier]) {
    // Кандидат в выбросы: крупная красная точка с подписью — сначала проверить запись.
    series.push({
      type: "scatter", name: t("plot.outlier"), data: [points[r.outlier]], symbolSize: 12,
      itemStyle: { color: chart.outlier },
      label: { show: true, position: "right", color: chart.outlier, formatter: t("plot.outlier") },
      tooltip: { formatter: t("plot.outlierHint") },
    });
  }
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: t("axis.fitted"), ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: t("axis.residuals"), ...AXIS_STYLE },
    series,
  };
}

/**
 * Границы оси по концам планок: scale: true смотрит только на точки, и концы
 * ДИ (custom-серия) вылезали бы за сетку. 5 % запаса с каждой стороны.
 */
function bounds(values: number[]): { min: number; max: number } {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.05;
  // Округляем до «круглого» шага, иначе крайние подписи вида −290.289.
  // Шаг 1 / 2 / 5 × 10^k, как у делений ECharts: крайние подписи совпадают с сеткой.
  const raw = (hi - lo + 2 * pad) / 5 || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((k) => k * mag).find((v) => v >= raw) ?? 10 * mag;
  // toPrecision убирает хвосты плавающей точки (1.4000000000000001).
  const round = (v: number) => Number(v.toPrecision(12));
  const min = round(Math.floor((lo - pad) / step) * step);
  // Неотрицательные данные (частоты, вероятности, рычаги) — ось не уходит ниже 0.
  return { min: lo >= 0 ? Math.max(0, min) : min, max: round(Math.ceil((hi + pad) / step) * step) };
}

const fmt = (v: number) =>
  (Math.abs(v) >= 100 ? v.toFixed(1) : String(Number(v.toPrecision(4))));

/** Средние с ДИ: точка и планка; у доз — числовая ось (гл. 8.4). */
function meansCiOption(m: NonNullable<PlotBundle["means_ci"]>, t: T, colorOf: ColorOf): Option {
  const pos = (i: number) => (m.numeric ? m.items[i].x! : i);
  const tip = (i: number) => {
    const it = m.items[i];
    return `${esc(it.label)}: ${fmt(it.mean)} [${fmt(it.lower)}; ${fmt(it.upper)}]`;
  };
  return {
    grid: { ...BASE_GRID, bottom: m.numeric ? 40 : 60 },
    tooltip: { trigger: "item", formatter: (p: any) => tip(p.dataIndex) },
    xAxis: m.numeric
      ? { type: "value", scale: true, ...AXIS_STYLE }
      : { type: "category", data: m.items.map((it) => it.label), ...AXIS_STYLE,
          axisLabel: { color: chart.axis, rotate: m.items.length > 8 ? 35 : 0 } },
    yAxis: { type: "value", name: t("axis.mean"), ...AXIS_STYLE,
      ...bounds(m.items.flatMap((it) => [it.lower, it.upper])) },
    series: [
      { type: "custom", data: m.items.map((it, i) => [pos(i), it.lower, it.upper]),
        renderItem: (_p: unknown, api: any) => whisker(api, true), z: 1 },
      // Дозы (числовая ось) — одна кривая, цвет акцента; группы — цветом группы.
      { type: "scatter", symbolSize: 10, z: 2, data: m.items.map((it, i) => ({
        value: [pos(i), it.mean],
        itemStyle: { color: m.numeric ? c.accent : colorOf(it.label, i, m.items.length) } })) },
    ],
  };
}

/** Взаимодействие: x — уровни A, линии — уровни B; ДИ — в подсказке. */
function interactionOption(g: NonNullable<PlotBundle["interaction"]>): Option {
  const tip = (p: any) => {
    const s = g.series[p.seriesIndex];
    const i = p.dataIndex;
    if (s.values[i] === null) return "";
    return `${esc(g.factor_a)}=${esc(g.levels_a[i])}, ${esc(g.factor_b)}=${esc(s.name)}: ${fmt(s.values[i]!)}`
      + ` [${fmt(s.lower[i]!)}; ${fmt(s.upper[i]!)}]`;
  };
  return {
    grid: { ...BASE_GRID, top: 56 },
    tooltip: { trigger: "item", formatter: tip },
    legend: { top: 8, textStyle: { color: c.muted }, data: g.series.map((s) => s.name) },
    xAxis: { type: "category", name: g.factor_a, data: g.levels_a, ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, ...AXIS_STYLE },
    series: g.series.map((s, i) => {
      const st = seriesStyle(i);
      return { type: "line", name: s.name, data: s.values, connectNulls: false, symbol: st.symbol,
        symbolSize: 8, lineStyle: { width: 2, color: st.color, type: st.lineType }, itemStyle: { color: st.color } };
    }),
  };
}

/** Forest разностей (или отношений): опорная линия и границы TOST. */
function differencesOption(d: NonNullable<PlotBundle["differences"]>, t: T): Option {
  const labels = d.items.map((it) => it.label);
  const lines: Record<string, unknown>[] = [
    { xAxis: d.reference, lineStyle: { color: chart.ref, type: "dashed" }, label: { show: false } },
  ];
  const edges = d.items.flatMap((it) => [it.lower, it.upper]).concat(d.reference);
  if (d.margin !== undefined) {
    for (const x of [d.reference - d.margin, d.reference + d.margin]) {
      edges.push(x);
      lines.push({ xAxis: x, lineStyle: { color: chart.fit, type: "dotted", width: 1.5 },
        label: { formatter: `${t("axis.margin")}${fmt(d.margin)}`, color: chart.fit, position: "start" } });
    }
  }
  const tip = (p: any) => {
    const it = d.items[p.dataIndex];
    return `${esc(it.label)}: ${fmt(it.diff)} [${fmt(it.lower)}; ${fmt(it.upper)}]`;
  };
  return {
    grid: { left: 12, right: 28, top: d.margin !== undefined ? 36 : 20, bottom: 36, containLabel: true },
    animation: false,
    tooltip: { trigger: "item", formatter: tip },
    xAxis: { type: "value", name: t(d.kind === "ratio" ? "axis.ratio" : "axis.difference"),
      nameLocation: "middle", nameGap: 24, ...AXIS_STYLE, ...bounds(edges) },
    yAxis: { type: "category", data: labels, inverse: true, ...AXIS_STYLE },
    series: [
      { type: "custom", data: d.items.map((it, i) => [i, it.lower, it.upper]),
        renderItem: (_p: unknown, api: any) => {
          // в custom-серии на горизонтальном forest позиция — индекс категории оси y
          const [i, lo, hi] = [api.value(0), api.value(1), api.value(2)];
          const a = api.coord([lo, i]);
          const b = api.coord([hi, i]);
          const style = { stroke: c.text, lineWidth: 1.5 };
          return { type: "group", children: [
            { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style },
            { type: "line", shape: { x1: a[0], y1: a[1] - 5, x2: a[0], y2: a[1] + 5 }, style },
            { type: "line", shape: { x1: b[0], y1: b[1] - 5, x2: b[0], y2: b[1] + 5 }, style },
          ] };
        }, z: 1 },
      { type: "scatter", data: d.items.map((it, i) => [it.diff, i]), symbolSize: 8,
        itemStyle: { color: c.accent }, z: 2,
        markLine: { silent: true, symbol: "none", lineStyle: { type: "dashed" }, data: lines } },
    ],
  };
}

/** Кривые по группам: точки (цветом группы, если известна) и линии. */
function groupCurvesOption(g: NonNullable<PlotBundle["group_curves"]>, t: T, order: string[],
  axes: AxisNames = {}): Option {
  const names = g.curves.map((cv) => cv.name);
  // Номер группы — по общему порядку результата; сверх 8 групп цвет повторяется, меняются линия и маркер.
  const index = (name: string) => (order.includes(name) ? order.indexOf(name) : names.indexOf(name));
  const style = (name: string) => seriesStyle(Math.max(0, index(name)));
  const color = (name: string | null) => style(name ?? "").color;
  const byGroup = new Map<string | null, number[][]>();
  for (const [x, y, grp] of g.points) {
    const key = grp !== null && names.includes(grp) ? grp : null;
    byGroup.set(key, [...(byGroup.get(key) ?? []), [x, y]]);
  }
  const points = [...byGroup].map(([grp, data]) => ({
    type: "scatter", name: grp ?? t("axis.all_groups"), data, symbolSize: 7,
    symbol: grp === null ? "circle" : style(grp).symbol,
    itemStyle: { color: grp === null ? chart.ref : color(grp), opacity: 0.7 },
  }));
  const curves = g.curves.flatMap((cv) => [
    { type: "line", name: cv.name, data: cv.line, symbol: "none",
      lineStyle: { color: color(cv.name), width: 2, type: style(cv.name).lineType },
      itemStyle: { color: color(cv.name) } },
    ...(cv.band ? [cv.band.lower, cv.band.upper] : []).map((data) => ({
      type: "line", name: cv.name, data, symbol: "none", silent: true,
      lineStyle: { color: color(cv.name), width: 1, type: "dashed", opacity: 0.7 } })),
  ]);
  return {
    grid: { ...BASE_GRID, top: 56 },
    tooltip: { trigger: "item" },
    legend: { top: 8, textStyle: { color: c.muted }, data: [...new Set([...names, ...points.map((p) => p.name)])] },
    xAxis: { type: "value", scale: true, name: axes.x ?? "x", nameLocation: "middle", nameGap: 26,
      ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: axes.y ?? "y", ...AXIS_STYLE },
    series: [...points, ...curves],
  };
}

const EXTRA_AXES: Record<string, [string, string]> = {
  residual_vs_x: ["x", "axis.residuals"],
  discrepancy_vs_x: ["x", "axis.discrepancy"],
  abs_residual_vs_fitted: ["axis.fitted", "axis.abs_residuals"],
  leverage_vs_x: ["x", "axis.leverage"],
  cook_vs_fitted: ["axis.fitted", "axis.cook"],
  observed_vs_fitted: ["axis.fitted", "axis.observed"],
  power_vs_n: ["axis.n", "axis.power"],
  cp_vs_p: ["axis.p", "axis.cp"],
  adj_r2_vs_p: ["axis.p", "axis.adj_r2"],
};

/** Точечный график против x / подгонки с опорными линиями (0, пороги, y = x). */
function extraScatterOption(e: NonNullable<PlotBundle["extra_scatter"]>[number], t: T): Option {
  const [xKey, yKey] = EXTRA_AXES[e.kind];
  const xName = e.x_name || (xKey === "x" ? "x" : t(xKey));
  const marks: Record<string, unknown>[] = e.ref.map((y) => ({ yAxis: y }));
  const series: Record<string, unknown>[] = [{
    type: e.line ? "line" : "scatter", data: e.points, symbolSize: 6,
    itemStyle: { color: c.accent, opacity: e.line ? 1 : 0.6 },
    markLine: marks.length ? { silent: true, symbol: "none",
      lineStyle: { color: chart.ref, type: "dashed" }, label: { show: false }, data: marks } : undefined,
  }];
  if (e.diagonal || e.kind === "observed_vs_fitted") {
    // y = x по диапазону оси x (Cp = p; наблюдённое = подогнанное).
    const v = e.points.map((p) => p[0]);
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    series.push({ type: "line", data: [[lo, lo], [hi, hi]], symbol: "none", silent: true,
      lineStyle: { color: chart.ref, type: "dashed" } });
  }
  const ys = e.points.map((p) => p[1]).concat(e.ref);
  const tip = (p: any) => {
    const [x, y] = e.points[p.dataIndex] ?? [];
    const label = e.labels?.[p.dataIndex];
    return `${label ? `${esc(label)}<br/>` : ""}${esc(xName)} ${fmt(x)} · ${t(yKey)} ${fmt(y)}`;
  };
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item", formatter: (p: any) => (p.seriesIndex === 0 ? tip(p) : "") },
    xAxis: { type: "value", scale: true, name: xName, ...AXIS_STYLE },
    yAxis: { type: "value", name: t(yKey), ...AXIS_STYLE,
      ...(e.kind === "power_vs_n" ? { min: 0, max: 1 } : bounds(ys)) },
    series,
  };
}

/** Вероятности категорий: составные столбцы (порядок категорий сохранён). */
function stackedOption(st: NonNullable<PlotBundle["stacked"]>, t: T): Option {
  return {
    grid: { ...BASE_GRID, top: 56 },
    color: [...chart.categorical],
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
      valueFormatter: (v: number) => `${(v * 100).toFixed(1)}%` },
    legend: { top: 8, textStyle: { color: c.muted } },
    xAxis: { type: "category", data: st.rows.map((r) => r.label), ...AXIS_STYLE },
    yAxis: { type: "value", min: 0, max: 1, name: t("axis.probability"), ...AXIS_STYLE },
    series: st.categories.map((cat, j) => ({
      type: "bar", name: cat, stack: "p", barWidth: "55%",
      data: st.rows.map((r) => r.values[j]),
    })),
  };
}

/** Теплокарта скорректированных остатков: синий — меньше ожидаемого, красный — больше. */
function heatmapOption(h: NonNullable<PlotBundle["heatmap"]>, t: T): Option {
  const limit = Math.max(3, ...h.cells.map((cell) => Math.abs(cell[2])));
  return {
    grid: { left: 12, right: 24, top: 16, bottom: 64, containLabel: true },
    tooltip: { trigger: "item", formatter: (p: any) => {
      const [col, row, res, obs, exp] = p.value as number[];
      return `${esc(h.rows[row])} × ${esc(h.columns[col])}: ${obs} (${t("axis.expected")} ${fmt(exp)}),`
        + ` ${t("axis.adj_residual")} ${fmt(res)}`;
    } },
    xAxis: { type: "category", data: h.columns, ...AXIS_STYLE },
    yAxis: { type: "category", data: h.rows, inverse: true, ...AXIS_STYLE },
    visualMap: { min: -limit, max: limit, calculable: false, orient: "horizontal", left: "center",
      bottom: 0, dimension: 2, text: [`+${fmt(limit)}`, `−${fmt(limit)}`],
      inRange: { color: [...chart.diverging] }, textStyle: { color: chart.axis } },
    // Число в тёмной ячейке — белым: тёмный текст на синем давал 2.64:1 (аудит 04-06).
    series: [{ type: "heatmap",
      data: h.cells.map((cell) => ({ value: cell,
        label: { color: Math.abs(cell[2]) > 0.55 * limit ? c.panel : c.text } })),
      label: { show: true, formatter: (p: any) => String(p.value[3]) },
      itemStyle: { borderColor: c.panel, borderWidth: 2 } }],
  };
}

/** До/после: линия на объект; рост — бирюзовым, падение — ржавым (красный — только выброс). */
function pairedOption(pd: NonNullable<PlotBundle["paired"]>): Option {
  const pairs = pd.pairs ?? [];
  return {
    grid: { ...BASE_GRID, left: 56, right: 56 },
    tooltip: { trigger: "item", formatter: (p: any) => {
      const [b, a] = pairs[p.seriesIndex];
      return `${esc(pd.labels[0])} ${fmt(b)} → ${esc(pd.labels[1])} ${fmt(a)}`;
    } },
    xAxis: { type: "category", data: pd.labels, boundaryGap: false, ...AXIS_STYLE },
    yAxis: { type: "value", ...AXIS_STYLE, ...bounds(pairs.flat()) },
    series: pairs.map(([b, a]) => ({
      type: "line", data: [b, a], symbolSize: 6,
      lineStyle: { width: 1.5, opacity: 0.7, color: a >= b ? c.accent : chart.categorical[1] },
      itemStyle: { color: a >= b ? c.accent : chart.categorical[1] },
    })),
  };
}

/** Горизонтальные столбцы (MSEP моделей): лучшая модель — цветом акцента. */
function barsOption(b: NonNullable<PlotBundle["bars"]>, t: T): Option {
  return {
    grid: { left: 12, right: 40, top: 16, bottom: 36, containLabel: true },
    tooltip: { trigger: "item", formatter: (p: any) => `${esc(b.items[p.dataIndex].label)}: ${fmt(p.value)}` },
    xAxis: { type: "value", name: t("axis.msep"), nameLocation: "middle", nameGap: 24, ...AXIS_STYLE },
    yAxis: { type: "category", data: b.items.map((it) => it.label), inverse: true, ...AXIS_STYLE },
    series: [{
      type: "bar", barWidth: "55%",
      data: b.items.map((it) => ({ value: it.value,
        itemStyle: { color: it.label === b.best ? c.accent : chart.ref } })),
      label: { show: true, position: "right", color: c.muted,
        formatter: (p: any) => (b.items[p.dataIndex].label === b.best ? t("axis.best") : "") },
    }],
  };
}

/** Поверхность отклика: теплокарта подгонки на сетке x1 × x2. */
function surfaceOption(sf: NonNullable<PlotBundle["surface"]>, t: T): Option {
  const z = sf.cells.map((cell) => cell[2]);
  const label = (v: number) => fmt(v);
  return {
    grid: { left: 12, right: 24, top: 16, bottom: 72, containLabel: true },
    tooltip: { trigger: "item", formatter: (p: any) => {
      const [i, j, v] = p.value as number[];
      return `${esc(sf.names[0])} ${fmt(sf.x1[i])}, ${esc(sf.names[1])} ${fmt(sf.x2[j])}: ${fmt(v)}`;
    } },
    xAxis: { type: "category", name: sf.names[0], data: sf.x1.map(label), ...AXIS_STYLE },
    yAxis: { type: "category", name: sf.names[1], data: sf.x2.map(label), ...AXIS_STYLE },
    visualMap: { min: Math.min(...z), max: Math.max(...z), calculable: true, orient: "horizontal",
      left: "center", bottom: 0, dimension: 2, text: [t("axis.fitted_value"), ""],
      inRange: { color: [...chart.sequential] }, textStyle: { color: chart.axis } },
    series: [{ type: "heatmap", data: sf.cells }],
  };
}

/** Профили повторных измерений: тонкая линия на субъект, жирная — средние. */
function profilesOption(pr: NonNullable<PlotBundle["profiles"]>, t: T): Option {
  const all = pr.lines.flat().concat(pr.means);
  return {
    grid: { ...BASE_GRID, left: 56, right: 40 },
    tooltip: { trigger: "item" },
    xAxis: { type: "category", data: pr.conditions, boundaryGap: false, ...AXIS_STYLE },
    yAxis: { type: "value", ...AXIS_STYLE, ...bounds(all) },
    series: [
      ...pr.lines.map((row) => ({ type: "line", data: row, symbolSize: 4, silent: true,
        lineStyle: { width: 1, opacity: 0.45, color: chart.ref }, itemStyle: { color: chart.ref } })),
      { type: "line", name: t("axis.mean_line"), data: pr.means, symbolSize: 9,
        lineStyle: { width: 3, color: c.accent }, itemStyle: { color: c.accent } },
    ],
  };
}

/** Наблюдаемые и ожидаемые частоты рядом (одномерный χ²); ось от 0 (Witte 2.11). */
function observedExpectedOption(oe: NonNullable<PlotBundle["observed_expected"]>, t: T): Option {
  return {
    grid: { ...BASE_GRID, top: 56 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { top: 8, textStyle: { color: c.muted } },
    xAxis: { type: "category", data: oe.labels, ...AXIS_STYLE },
    yAxis: { type: "value", min: 0, name: t("axis.frequency"), ...AXIS_STYLE },
    series: [
      { type: "bar", name: t("axis.observed_f"), data: oe.observed, itemStyle: { color: c.accent } },
      { type: "bar", name: t("axis.expected_f"), data: oe.expected,
        itemStyle: { color: c.panel, borderColor: c.text, borderType: "dashed", borderWidth: 1.5 } },
    ],
  };
}

/** Таблица частот: классы — смежные столбцы, категории — с зазорами (Witte 2.8–2.10). */
function frequencyOption(fr: NonNullable<PlotBundle["frequency"]>, t: T): Option {
  return {
    grid: { ...BASE_GRID, bottom: 60 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: { type: "category", data: fr.labels, ...AXIS_STYLE,
      axisLabel: { color: chart.axis, rotate: fr.labels.length > 10 ? 35 : 0 } },
    yAxis: { type: "value", min: 0, name: t("axis.frequency"), ...AXIS_STYLE },
    series: [{ type: "bar", data: fr.f, barCategoryGap: fr.adjacent ? "0%" : "40%",
      itemStyle: { color: c.accent, opacity: 0.8, borderColor: fr.adjacent ? c.panel : undefined,
        borderWidth: fr.adjacent ? 1 : 0 } }],
  };
}

/* --------------------------- Публичный вид --------------------------- */

interface Item {
  key: string;
  title: string;
  option: Option;
  height?: number;
}

/** Итоговые виды (мощность, отбор подмножеств) — к итогам; прочие — к диагностике. */
const SUMMARY_KINDS = new Set(["power_vs_n", "cp_vs_p", "adj_r2_vs_p"]);

function extraItems(bundle: PlotBundle, t: T, summary: boolean): Item[] {
  return (bundle.extra_scatter ?? [])
    .map((e, i) => ({ e, i }))
    .filter(({ e }) => SUMMARY_KINDS.has(e.kind) === summary)
    .map(({ e, i }) => ({
      key: `extra_${i}`,
      title: t(`plots.${e.kind}`) + (e.x_name ? `: ${e.x_name}` : ""),
      option: extraScatterOption(e, t),
    }));
}

/** Графики-итоги идут первыми: это вывод анализа; диагностика — следом. */
function summaryItems(bundle: PlotBundle, t: T): Item[] {
  const items: Item[] = [];
  if (bundle.interaction) {
    const g = bundle.interaction;
    items.push({ key: "interaction", title: `${t("plots.interaction")} (${g.factor_a} × ${g.factor_b})`,
      option: interactionOption(bundle.interaction) });
  }
  if (bundle.paired) {
    if (bundle.paired.pairs) {
      items.push({ key: "paired", title: t("plots.paired"), option: pairedOption(bundle.paired) });
    }
    items.push({ key: "paired_diff", title: t("plots.paired_diff"),
      option: histogramOption(bundle.paired.diff_hist, t) });
  }
  if (bundle.stacked) {
    items.push({ key: "stacked", title: t("plots.stacked"), option: stackedOption(bundle.stacked, t) });
  }
  items.push(...extraItems(bundle, t, true));
  if (bundle.frequency) {
    items.push({ key: "frequency", title: t("plots.frequency"), option: frequencyOption(bundle.frequency, t) });
  }
  if (bundle.observed_expected) {
    items.push({ key: "observed_expected", title: t("plots.observed_expected"),
      option: observedExpectedOption(bundle.observed_expected, t) });
  }
  if (bundle.profiles) {
    items.push({ key: "profiles", title: t("plots.profiles"), option: profilesOption(bundle.profiles, t) });
  }
  if (bundle.surface) {
    items.push({ key: "surface", title: t("plots.surface"), option: surfaceOption(bundle.surface, t),
      height: 420 });
  }
  if (bundle.bars) {
    items.push({ key: "bars", title: t(`plots.${bundle.bars.kind}`), option: barsOption(bundle.bars, t),
      height: Math.max(160, 34 * bundle.bars.items.length + 70) });
  }
  if (bundle.heatmap) {
    items.push({ key: "heatmap", title: t("plots.heatmap"), option: heatmapOption(bundle.heatmap, t),
      height: Math.max(200, 48 * bundle.heatmap.rows.length + 110) });
  }
  if (bundle.means_ci) {
    const m = bundle.means_ci;
    const title = m.title === "predictions"
      ? t(m.interval === "none" ? "plots.predictions_noci" : "plots.predictions")
      : t(m.interval === "se" ? "plots.means_se" : "plots.means_ci");
    items.push({ key: "means_ci", title: bundle.means_ci.term ? `${title}: ${bundle.means_ci.term}` : title,
      option: meansCiOption(bundle.means_ci, t, groupPalette(bundle).colorOf) });
  }
  if (bundle.group_curves) {
    const models = bundle.group_curves.title === "models";
    items.push({ key: "group_curves", title: t(models ? "plots.group_curves_models" : "plots.group_curves"),
      option: groupCurvesOption(bundle.group_curves, t, groupPalette(bundle).order, bundle.axis_names),
      height: 340 });
  }
  if (bundle.differences) {
    const d = bundle.differences;
    const key = d.margin !== undefined ? "plots.equivalence"
      : d.kind === "ratio" ? "plots.ratios" : "plots.differences";
    items.push({ key: "differences", title: t(key), option: differencesOption(d, t),
      height: Math.max(180, 26 * d.items.length + 70) });
  }
  return items;
}

function diagnosticItems(bundle: PlotBundle, t: T): Item[] {
  const items: Item[] = [];
  const add = (key: string, option: Option) => items.push({ key, title: t(`plots.${key}`), option });
  const { colorOf } = groupPalette(bundle);
  const names = bundle.axis_names ?? {};
  if (bundle.histogram) add("histogram", histogramOption(bundle.histogram, t, names));
  if (bundle.boxplot) add("boxplot", boxplotOption(bundle.boxplot, colorOf, names));
  if (bundle.bar_means) add("bar_means", barMeansOption(bundle.bar_means, t, colorOf));
  if (bundle.scatter) add("scatter", scatterOption(bundle.scatter, names));
  if (bundle.qq) add("qq", qqOption(bundle.qq, t));
  if (bundle.residual_vs_fitted) add("residual_vs_fitted", residualOption(bundle.residual_vs_fitted, t));
  items.push(...extraItems(bundle, t, false));
  return items;
}

/** Число для словесной сводки: три значащие цифры. */
const short = (v: number) => String(Number(v.toPrecision(3)));

/** «пшеница 5.3, ячмень 4.75, …» — не больше восьми групп. */
function listed(labels: string[], values: number[]): string {
  const items = labels.slice(0, 8).map((l, i) => `${l} ${short(values[i])}`);
  return items.join(", ") + (labels.length > 8 ? ", …" : "");
}

/**
 * Краткая сводка графика словами — текстовая альтернатива для экранного диктора
 * (аудит 04-04, WCAG 1.1.1). Нет сводки — подпись из одного заголовка.
 */
export function chartSummary(key: string, bundle: PlotBundle, t: T): string | null {
  const n = (k: string, v: number) => t(k).replace("{n}", String(v));
  switch (key) {
    case "boxplot": {
      const b = bundle.boxplot!;
      return `${n("plots.sum.groups", b.labels.length)}; ${t("plots.sum.medians")}: `
        + listed(b.labels, b.boxes.map((box) => box[2]));
    }
    case "bar_means": {
      const b = bundle.bar_means!;
      return `${n("plots.sum.groups", b.labels.length)}; ${t("plots.sum.means")}: ${listed(b.labels, b.means)}`;
    }
    case "means_ci": {
      const m = bundle.means_ci!;
      return `${n("plots.sum.groups", m.items.length)}; ${t("plots.sum.means")}: `
        + listed(m.items.map((it) => it.label), m.items.map((it) => it.mean));
    }
    case "histogram": {
      const h = bundle.histogram!;
      const total = h.counts.reduce((a, b) => a + b, 0);
      return `${n("plots.sum.obs", total)}; ${t("plots.sum.range")
        .replace("{a}", short(h.bin_edges[0])).replace("{b}", short(h.bin_edges[h.bin_edges.length - 1]))}`;
    }
    case "scatter": return n("plots.sum.points", bundle.scatter!.points.length);
    case "qq": return n("plots.sum.points", bundle.qq!.sample_quantiles.length);
    case "residual_vs_fitted": return n("plots.sum.points", bundle.residual_vs_fitted!.residuals.length);
    case "group_curves": return n("plots.sum.curves", bundle.group_curves!.curves.length);
    case "paired": return n("plots.sum.pairs", bundle.paired!.pairs?.length ?? 0);
    case "frequency": return n("plots.sum.categories", bundle.frequency!.labels.length);
    case "differences": return n("plots.sum.comparisons", bundle.differences!.items.length);
    default: return null;
  }
}

/** Отрисовать все графики из набора, каждый со своим заголовком. */
export function Plots({ bundle }: { bundle: PlotBundle }) {
  const t = useT();
  const items = [...summaryItems(bundle, t), ...diagnosticItems(bundle, t)];
  if (items.length === 0) return null;

  return (
    <div style={{ display: "grid", gap: "1.5rem", marginTop: "1rem" }}>
      {items.map(({ key, title, option, height }) => {
        const summary = chartSummary(key, bundle, t);
        return (
          <div key={key}>
            <div style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.4rem", color: c.text }}>
              {title}
            </div>
            {/* Для диктора график — одна картинка с подписью: заголовок и сводка. */}
            <div role="img" aria-label={summary ? `${title}. ${summary}` : title}>
              <Chart option={option} height={height} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
