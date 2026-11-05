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
import { useT } from "./i18n";
import type { PlotBundle } from "./pyodide/bridge";
import { c } from "./styles";

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
  axisLine: { lineStyle: { color: c.border } },
  axisLabel: { color: c.muted },
  splitLine: { lineStyle: { color: "#f4f4f5" } },
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
const RED = "#dc2626";
const MUTED_LINE = "#94a3b8";
const PALETTE = ["#0f766e", "#b45309", "#1d4ed8", "#be123c", "#4d7c0f",
  "#7c3aed", "#0369a1", "#a16207", "#475569", "#c2410c"];

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

function histogramOption(h: NonNullable<PlotBundle["histogram"]>, t: T): Option {
  const bars = h.bin_centers.map((x, i) => [x, h.counts[i]]);
  const series: Record<string, unknown>[] = [
    { type: "bar", data: bars, barWidth: "99%", itemStyle: { color: c.accent, opacity: 0.75 },
      name: t("axis.frequency") },
  ];
  if (h.normal_curve) {
    series.push({
      type: "line", data: h.normal_curve.x.map((x, i) => [x, h.normal_curve!.y[i]]),
      smooth: true, symbol: "none", lineStyle: { color: RED, width: 2 }, name: t("axis.normal"),
    });
  }
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "axis" },
    legend: { top: 8, textStyle: { color: c.muted } },
    xAxis: { type: "value", scale: true, ...AXIS_STYLE },
    yAxis: { type: "value", name: t("axis.frequency"), ...AXIS_STYLE },
    series,
  };
}

function boxplotOption(b: NonNullable<PlotBundle["boxplot"]>): Option {
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "category", data: b.labels, ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, ...AXIS_STYLE },
    series: [
      { type: "boxplot", data: b.boxes, itemStyle: { color: "#ccfbf1", borderColor: c.accent } },
      { type: "scatter", data: b.outliers, symbolSize: 6, itemStyle: { color: RED } },
    ],
  };
}

/**
 * Средние ± SE точками, а не столбцами (Witte 2.11, Fig. 14.5): столбец с
 * основанием не от нуля искажает пропорции, а от нуля — прячет различия.
 */
function barMeansOption(b: NonNullable<PlotBundle["bar_means"]>, t: T): Option {
  const lo = b.means.map((m, i) => m - b.errors[i]);
  const hi = b.means.map((m, i) => m + b.errors[i]);
  const tip = (p: any) => `${b.labels[p.dataIndex]}: ${fmt(b.means[p.dataIndex])}`
    + ` ± ${fmt(b.errors[p.dataIndex])}`;
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item", formatter: tip },
    xAxis: { type: "category", data: b.labels, ...AXIS_STYLE },
    yAxis: { type: "value", name: t("axis.mean"), ...AXIS_STYLE, ...bounds([...lo, ...hi]) },
    series: [
      { type: "custom", data: b.means.map((_, i) => [i, lo[i], hi[i]]),
        renderItem: (_p: unknown, api: any) => whisker(api, true), z: 1 },
      { type: "scatter", data: b.means.map((m, i) => [i, m]), symbolSize: 9,
        itemStyle: { color: c.accent }, z: 2 },
    ],
  };
}

function scatterOption(sc: NonNullable<PlotBundle["scatter"]>): Option {
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: "X", ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: "Y", ...AXIS_STYLE },
    series: [
      { type: "scatter", data: sc.points, symbolSize: 7, itemStyle: { color: c.accent, opacity: 0.6 } },
      { type: "line", data: sc.line, symbol: "none", lineStyle: { color: RED, width: 2 } },
      // Доверительная полоса кривой: две пунктирные границы.
      ...(sc.band ? [sc.band.lower, sc.band.upper] : []).map((data) => ({
        type: "line", data, symbol: "none", silent: true,
        lineStyle: { color: RED, width: 1, type: "dashed", opacity: 0.7 },
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
        lineStyle: { color: MUTED_LINE, type: "dashed" } },
    ],
  };
}

function residualOption(r: NonNullable<PlotBundle["residual_vs_fitted"]>, t: T): Option {
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: t("axis.fitted"), ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: t("axis.residuals"), ...AXIS_STYLE },
    series: [{
      type: "scatter",
      data: r.fitted.map((f, i) => [f, r.residuals[i]]),
      symbolSize: 6,
      itemStyle: { color: c.accent, opacity: 0.6 },
      markLine: { silent: true, symbol: "none", lineStyle: { color: MUTED_LINE, type: "dashed" },
        data: [{ yAxis: 0 }] },
    }],
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
function meansCiOption(m: NonNullable<PlotBundle["means_ci"]>, t: T): Option {
  const pos = (i: number) => (m.numeric ? m.items[i].x! : i);
  const tip = (i: number) => {
    const it = m.items[i];
    return `${it.label}: ${fmt(it.mean)} [${fmt(it.lower)}; ${fmt(it.upper)}]`;
  };
  return {
    grid: { ...BASE_GRID, bottom: m.numeric ? 40 : 60 },
    tooltip: { trigger: "item", formatter: (p: any) => tip(p.dataIndex) },
    xAxis: m.numeric
      ? { type: "value", scale: true, ...AXIS_STYLE }
      : { type: "category", data: m.items.map((it) => it.label), ...AXIS_STYLE,
          axisLabel: { color: c.muted, rotate: m.items.length > 8 ? 35 : 0 } },
    yAxis: { type: "value", name: t("axis.mean"), ...AXIS_STYLE,
      ...bounds(m.items.flatMap((it) => [it.lower, it.upper])) },
    series: [
      { type: "custom", data: m.items.map((it, i) => [pos(i), it.lower, it.upper]),
        renderItem: (_p: unknown, api: any) => whisker(api, true), z: 1 },
      { type: "scatter", data: m.items.map((it, i) => [pos(i), it.mean]), symbolSize: 9,
        itemStyle: { color: c.accent }, z: 2 },
    ],
  };
}

/** Взаимодействие: x — уровни A, линии — уровни B; ДИ — в подсказке. */
function interactionOption(g: NonNullable<PlotBundle["interaction"]>): Option {
  const tip = (p: any) => {
    const s = g.series[p.seriesIndex];
    const i = p.dataIndex;
    if (s.values[i] === null) return "";
    return `${g.factor_a}=${g.levels_a[i]}, ${g.factor_b}=${s.name}: ${fmt(s.values[i]!)}`
      + ` [${fmt(s.lower[i]!)}; ${fmt(s.upper[i]!)}]`;
  };
  return {
    grid: { ...BASE_GRID, top: 56 },
    color: PALETTE,
    tooltip: { trigger: "item", formatter: tip },
    legend: { top: 8, textStyle: { color: c.muted }, data: g.series.map((s) => s.name) },
    xAxis: { type: "category", name: g.factor_a, data: g.levels_a, ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, ...AXIS_STYLE },
    series: g.series.map((s) => ({
      type: "line", name: s.name, data: s.values, connectNulls: false,
      symbolSize: 8, lineStyle: { width: 2 },
    })),
  };
}

/** Forest разностей (или отношений): опорная линия и границы TOST. */
function differencesOption(d: NonNullable<PlotBundle["differences"]>, t: T): Option {
  const labels = d.items.map((it) => it.label);
  const lines: Record<string, unknown>[] = [
    { xAxis: d.reference, lineStyle: { color: MUTED_LINE, type: "dashed" }, label: { show: false } },
  ];
  const edges = d.items.flatMap((it) => [it.lower, it.upper]).concat(d.reference);
  if (d.margin !== undefined) {
    for (const x of [d.reference - d.margin, d.reference + d.margin]) {
      edges.push(x);
      lines.push({ xAxis: x, lineStyle: { color: RED, type: "dashed" },
        label: { formatter: `${t("axis.margin")}${fmt(d.margin)}`, color: RED, position: "start" } });
    }
  }
  const tip = (p: any) => {
    const it = d.items[p.dataIndex];
    return `${it.label}: ${fmt(it.diff)} [${fmt(it.lower)}; ${fmt(it.upper)}]`;
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
function groupCurvesOption(g: NonNullable<PlotBundle["group_curves"]>, t: T): Option {
  const names = g.curves.map((cv) => cv.name);
  const color = (name: string | null) => PALETTE[Math.max(0, names.indexOf(name ?? "")) % PALETTE.length];
  const byGroup = new Map<string | null, number[][]>();
  for (const [x, y, grp] of g.points) {
    const key = grp !== null && names.includes(grp) ? grp : null;
    byGroup.set(key, [...(byGroup.get(key) ?? []), [x, y]]);
  }
  const points = [...byGroup].map(([grp, data]) => ({
    type: "scatter", name: grp ?? t("axis.all_groups"), data, symbolSize: 7,
    itemStyle: { color: grp === null ? MUTED_LINE : color(grp), opacity: 0.7 },
  }));
  const curves = g.curves.flatMap((cv) => [
    { type: "line", name: cv.name, data: cv.line, symbol: "none",
      lineStyle: { color: color(cv.name), width: 2 }, itemStyle: { color: color(cv.name) } },
    ...(cv.band ? [cv.band.lower, cv.band.upper] : []).map((data) => ({
      type: "line", name: cv.name, data, symbol: "none", silent: true,
      lineStyle: { color: color(cv.name), width: 1, type: "dashed", opacity: 0.7 } })),
  ]);
  return {
    grid: { ...BASE_GRID, top: 56 },
    tooltip: { trigger: "item" },
    legend: { top: 8, textStyle: { color: c.muted }, data: [...new Set([...names, ...points.map((p) => p.name)])] },
    xAxis: { type: "value", scale: true, name: "X", ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: "Y", ...AXIS_STYLE },
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
      lineStyle: { color: e.kind === "residual_vs_x" || e.kind === "discrepancy_vs_x" ? MUTED_LINE : RED,
        type: "dashed" }, label: { show: false }, data: marks } : undefined,
  }];
  if (e.diagonal || e.kind === "observed_vs_fitted") {
    // y = x по диапазону оси x (Cp = p; наблюдённое = подогнанное).
    const v = e.points.map((p) => p[0]);
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    series.push({ type: "line", data: [[lo, lo], [hi, hi]], symbol: "none", silent: true,
      lineStyle: { color: MUTED_LINE, type: "dashed" } });
  }
  const ys = e.points.map((p) => p[1]).concat(e.ref);
  const tip = (p: any) => {
    const [x, y] = e.points[p.dataIndex] ?? [];
    const label = e.labels?.[p.dataIndex];
    return `${label ? `${label}<br/>` : ""}${xName} ${fmt(x)} · ${t(yKey)} ${fmt(y)}`;
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
    color: PALETTE,
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
      return `${h.rows[row]} × ${h.columns[col]}: ${obs} (${t("axis.expected")} ${fmt(exp)}),`
        + ` ${t("axis.adj_residual")} ${fmt(res)}`;
    } },
    xAxis: { type: "category", data: h.columns, ...AXIS_STYLE },
    yAxis: { type: "category", data: h.rows, inverse: true, ...AXIS_STYLE },
    visualMap: { min: -limit, max: limit, calculable: false, orient: "horizontal", left: "center",
      bottom: 0, dimension: 2, text: [`+${fmt(limit)}`, `−${fmt(limit)}`],
      inRange: { color: ["#1d4ed8", "#f8fafc", "#dc2626"] }, textStyle: { color: c.muted } },
    series: [{ type: "heatmap", data: h.cells,
      label: { show: true, color: c.text, formatter: (p: any) => String(p.value[3]) },
      itemStyle: { borderColor: "#fff", borderWidth: 2 } }],
  };
}

/** До/после: линия на объект; рост — цветом акцента, падение — красным. */
function pairedOption(pd: NonNullable<PlotBundle["paired"]>): Option {
  const pairs = pd.pairs ?? [];
  return {
    grid: { ...BASE_GRID, left: 56, right: 56 },
    tooltip: { trigger: "item", formatter: (p: any) => {
      const [b, a] = pairs[p.seriesIndex];
      return `${pd.labels[0]} ${fmt(b)} → ${pd.labels[1]} ${fmt(a)}`;
    } },
    xAxis: { type: "category", data: pd.labels, boundaryGap: false, ...AXIS_STYLE },
    yAxis: { type: "value", ...AXIS_STYLE, ...bounds(pairs.flat()) },
    series: pairs.map(([b, a]) => ({
      type: "line", data: [b, a], symbolSize: 6,
      lineStyle: { width: 1.5, opacity: 0.7, color: a >= b ? c.accent : RED },
      itemStyle: { color: a >= b ? c.accent : RED },
    })),
  };
}

/** Горизонтальные столбцы (MSEP моделей): лучшая модель — цветом акцента. */
function barsOption(b: NonNullable<PlotBundle["bars"]>, t: T): Option {
  return {
    grid: { left: 12, right: 40, top: 16, bottom: 36, containLabel: true },
    tooltip: { trigger: "item", formatter: (p: any) => `${b.items[p.dataIndex].label}: ${fmt(p.value)}` },
    xAxis: { type: "value", name: t("axis.msep"), nameLocation: "middle", nameGap: 24, ...AXIS_STYLE },
    yAxis: { type: "category", data: b.items.map((it) => it.label), inverse: true, ...AXIS_STYLE },
    series: [{
      type: "bar", barWidth: "55%",
      data: b.items.map((it) => ({ value: it.value,
        itemStyle: { color: it.label === b.best ? c.accent : MUTED_LINE } })),
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
      return `${sf.names[0]} ${fmt(sf.x1[i])}, ${sf.names[1]} ${fmt(sf.x2[j])}: ${fmt(v)}`;
    } },
    xAxis: { type: "category", name: sf.names[0], data: sf.x1.map(label), ...AXIS_STYLE },
    yAxis: { type: "category", name: sf.names[1], data: sf.x2.map(label), ...AXIS_STYLE },
    visualMap: { min: Math.min(...z), max: Math.max(...z), calculable: true, orient: "horizontal",
      left: "center", bottom: 0, dimension: 2, text: [t("axis.fitted_value"), ""],
      inRange: { color: ["#f0fdfa", "#5eead4", "#0f766e", "#134e4a"] }, textStyle: { color: c.muted } },
    series: [{ type: "heatmap", data: sf.cells }],
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
      option: meansCiOption(bundle.means_ci, t) });
  }
  if (bundle.group_curves) {
    const models = bundle.group_curves.title === "models";
    items.push({ key: "group_curves", title: t(models ? "plots.group_curves_models" : "plots.group_curves"),
      option: groupCurvesOption(bundle.group_curves, t), height: 340 });
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
  if (bundle.histogram) add("histogram", histogramOption(bundle.histogram, t));
  if (bundle.boxplot) add("boxplot", boxplotOption(bundle.boxplot));
  if (bundle.bar_means) add("bar_means", barMeansOption(bundle.bar_means, t));
  if (bundle.scatter) add("scatter", scatterOption(bundle.scatter));
  if (bundle.qq) add("qq", qqOption(bundle.qq, t));
  if (bundle.residual_vs_fitted) add("residual_vs_fitted", residualOption(bundle.residual_vs_fitted, t));
  items.push(...extraItems(bundle, t, false));
  return items;
}

/** Отрисовать все графики из набора, каждый со своим заголовком. */
export function Plots({ bundle }: { bundle: PlotBundle }) {
  const t = useT();
  const items = [...summaryItems(bundle, t), ...diagnosticItems(bundle, t)];
  if (items.length === 0) return null;

  return (
    <div style={{ display: "grid", gap: "1.5rem", marginTop: "1rem" }}>
      {items.map(({ key, title, option, height }) => (
        <div key={key}>
          <div style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.4rem", color: c.text }}>
            {title}
          </div>
          <Chart option={option} height={height} />
        </div>
      ))}
    </div>
  );
}
