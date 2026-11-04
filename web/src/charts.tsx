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
  LineChart,
  ScatterChart,
} from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
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
  GridComponent,
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

function barMeansOption(b: NonNullable<PlotBundle["bar_means"]>, t: T): Option {
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: b.labels, ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: t("axis.mean"), ...AXIS_STYLE },
    series: [
      { type: "bar", data: b.means, itemStyle: { color: c.accent, opacity: 0.8 }, barWidth: "50%" },
      {
        type: "custom",
        data: b.means.map((m, i) => [i, m - b.errors[i], m + b.errors[i]]),
        renderItem: (_p: unknown, api: any) => whisker(api, true),
      },
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
  const step = 10 ** Math.floor(Math.log10((hi - lo + 2 * pad) / 5 || 1));
  return { min: Math.floor((lo - pad) / step) * step, max: Math.ceil((hi + pad) / step) * step };
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4));

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
};

/** Точечный график против x / подгонки с опорными линиями (0, пороги, y = x). */
function extraScatterOption(e: NonNullable<PlotBundle["extra_scatter"]>[number], t: T): Option {
  const [xKey, yKey] = EXTRA_AXES[e.kind];
  const xName = e.x_name || (xKey === "x" ? "x" : t(xKey));
  const marks: Record<string, unknown>[] = e.ref.map((y) => ({ yAxis: y }));
  const series: Record<string, unknown>[] = [{
    type: "scatter", data: e.points, symbolSize: 6, itemStyle: { color: c.accent, opacity: 0.6 },
    markLine: marks.length ? { silent: true, symbol: "none",
      lineStyle: { color: e.kind === "residual_vs_x" || e.kind === "discrepancy_vs_x" ? MUTED_LINE : RED,
        type: "dashed" }, label: { show: false }, data: marks } : undefined,
  }];
  if (e.kind === "observed_vs_fitted") {
    const v = e.points.flat();
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    series.push({ type: "line", data: [[lo, lo], [hi, hi]], symbol: "none", silent: true,
      lineStyle: { color: MUTED_LINE, type: "dashed" } });
  }
  const ys = e.points.map((p) => p[1]).concat(e.ref);
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: xName, ...AXIS_STYLE },
    yAxis: { type: "value", name: t(yKey), ...AXIS_STYLE, ...bounds(ys) },
    series,
  };
}

/* --------------------------- Публичный вид --------------------------- */

interface Item {
  key: string;
  title: string;
  option: Option;
  height?: number;
}

/** Графики-итоги идут первыми: это вывод анализа; диагностика — следом. */
function summaryItems(bundle: PlotBundle, t: T): Item[] {
  const items: Item[] = [];
  if (bundle.interaction) {
    const g = bundle.interaction;
    items.push({ key: "interaction", title: `${t("plots.interaction")} (${g.factor_a} × ${g.factor_b})`,
      option: interactionOption(bundle.interaction) });
  }
  if (bundle.means_ci) {
    const title = t(bundle.means_ci.interval === "se" ? "plots.means_se" : "plots.means_ci");
    items.push({ key: "means_ci", title: bundle.means_ci.term ? `${title}: ${bundle.means_ci.term}` : title,
      option: meansCiOption(bundle.means_ci, t) });
  }
  if (bundle.group_curves) {
    items.push({ key: "group_curves", title: t("plots.group_curves"),
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
  (bundle.extra_scatter ?? []).forEach((e, i) => items.push({
    key: `extra_${i}`,
    title: t(`plots.${e.kind}`) + (e.x_name ? `: ${e.x_name}` : ""),
    option: extraScatterOption(e, t),
  }));
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
