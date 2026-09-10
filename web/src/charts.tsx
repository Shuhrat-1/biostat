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

function histogramOption(h: NonNullable<PlotBundle["histogram"]>) {
  const bars = h.bin_centers.map((x, i) => [x, h.counts[i]]);
  const series: Record<string, unknown>[] = [
    {
      type: "bar",
      data: bars,
      barWidth: "99%",
      itemStyle: { color: c.accent, opacity: 0.75 },
      name: "Частота",
    },
  ];
  if (h.normal_curve) {
    series.push({
      type: "line",
      data: h.normal_curve.x.map((x, i) => [x, h.normal_curve!.y[i]]),
      smooth: true,
      symbol: "none",
      lineStyle: { color: "#dc2626", width: 2 },
      name: "Норм. распределение",
    });
  }
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "axis" },
    legend: { top: 8, textStyle: { color: c.muted } },
    xAxis: { type: "value", scale: true, ...AXIS_STYLE },
    yAxis: { type: "value", name: "Частота", ...AXIS_STYLE },
    series,
  };
}

function boxplotOption(b: NonNullable<PlotBundle["boxplot"]>) {
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "category", data: b.labels, ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, ...AXIS_STYLE },
    series: [
      {
        type: "boxplot",
        data: b.boxes,
        itemStyle: { color: "#ccfbf1", borderColor: c.accent },
      },
      {
        type: "scatter",
        data: b.outliers,
        symbolSize: 6,
        itemStyle: { color: "#dc2626" },
      },
    ],
  };
}

function barMeansOption(b: NonNullable<PlotBundle["bar_means"]>) {
  // Планки погрешности рисуем custom-серией поверх столбцов.
  const errorBars = b.means.map((m, i) => ({
    value: [i, m - b.errors[i], m + b.errors[i]],
  }));
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: b.labels, ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: "Среднее", ...AXIS_STYLE },
    series: [
      {
        type: "bar",
        data: b.means,
        itemStyle: { color: c.accent, opacity: 0.8 },
        barWidth: "50%",
      },
      {
        type: "custom",
        data: errorBars,
        renderItem: (_params: unknown, api: any) => {
          const idx = api.value(0);
          const low = api.coord([idx, api.value(1)]);
          const high = api.coord([idx, api.value(2)]);
          const width = 8;
          return {
            type: "group",
            children: [
              {
                type: "line",
                shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] },
                style: { stroke: c.text, lineWidth: 1.5 },
              },
              {
                type: "line",
                shape: {
                  x1: high[0] - width,
                  y1: high[1],
                  x2: high[0] + width,
                  y2: high[1],
                },
                style: { stroke: c.text, lineWidth: 1.5 },
              },
              {
                type: "line",
                shape: {
                  x1: low[0] - width,
                  y1: low[1],
                  x2: low[0] + width,
                  y2: low[1],
                },
                style: { stroke: c.text, lineWidth: 1.5 },
              },
            ],
          };
        },
      },
    ],
  };
}

function scatterOption(sc: NonNullable<PlotBundle["scatter"]>) {
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: "X", ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: "Y", ...AXIS_STYLE },
    series: [
      {
        type: "scatter",
        data: sc.points,
        symbolSize: 7,
        itemStyle: { color: c.accent, opacity: 0.6 },
      },
      {
        type: "line",
        data: sc.line,
        symbol: "none",
        lineStyle: { color: "#dc2626", width: 2 },
      },
    ],
  };
}

function qqOption(qq: NonNullable<PlotBundle["qq"]>) {
  const points = qq.theoretical_quantiles.map((t, i) => [
    t,
    qq.sample_quantiles[i],
  ]);
  // Опорная линия y=x в диапазоне теоретических квантилей.
  const lo = Math.min(...qq.theoretical_quantiles);
  const hi = Math.max(...qq.theoretical_quantiles);
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: "Теор. квантили", ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: "Выб. квантили", ...AXIS_STYLE },
    series: [
      {
        type: "scatter",
        data: points,
        symbolSize: 6,
        itemStyle: { color: c.accent, opacity: 0.6 },
      },
      {
        type: "line",
        data: [
          [lo, lo],
          [hi, hi],
        ],
        symbol: "none",
        lineStyle: { color: "#94a3b8", type: "dashed" },
      },
    ],
  };
}

function residualOption(r: NonNullable<PlotBundle["residual_vs_fitted"]>) {
  const points = r.fitted.map((f, i) => [f, r.residuals[i]]);
  return {
    grid: BASE_GRID,
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: "Предсказанные", ...AXIS_STYLE },
    yAxis: { type: "value", scale: true, name: "Остатки", ...AXIS_STYLE },
    series: [
      {
        type: "scatter",
        data: points,
        symbolSize: 6,
        itemStyle: { color: c.accent, opacity: 0.6 },
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: "#94a3b8", type: "dashed" },
          data: [{ yAxis: 0 }],
        },
      },
    ],
  };
}

/* --------------------------- Публичный вид --------------------------- */

const TITLES: Record<string, string> = {
  histogram: "Гистограмма распределения",
  boxplot: "Боксплот по группам",
  bar_means: "Средние по группам (± SE)",
  scatter: "Диаграмма рассеяния с линией регрессии",
  qq: "Q-Q график (нормальность остатков)",
  residual_vs_fitted: "Остатки против предсказанных",
};

/** Отрисовать все графики из набора, каждый со своим заголовком. */
export function Plots({ bundle }: { bundle: PlotBundle }) {
  const items: Array<{ key: string; option: echarts.EChartsCoreOption }> = [];

  if (bundle.histogram)
    items.push({ key: "histogram", option: histogramOption(bundle.histogram) });
  if (bundle.boxplot)
    items.push({ key: "boxplot", option: boxplotOption(bundle.boxplot) });
  if (bundle.bar_means)
    items.push({ key: "bar_means", option: barMeansOption(bundle.bar_means) });
  if (bundle.scatter)
    items.push({ key: "scatter", option: scatterOption(bundle.scatter) });
  if (bundle.qq) items.push({ key: "qq", option: qqOption(bundle.qq) });
  if (bundle.residual_vs_fitted)
    items.push({
      key: "residual_vs_fitted",
      option: residualOption(bundle.residual_vs_fitted),
    });

  if (items.length === 0) return null;

  return (
    <div style={{ display: "grid", gap: "1.5rem", marginTop: "1rem" }}>
      {items.map(({ key, option }) => (
        <div key={key}>
          <div
            style={{
              fontSize: "0.9rem",
              fontWeight: 600,
              marginBottom: "0.4rem",
              color: c.text,
            }}
          >
            {TITLES[key] ?? key}
          </div>
          <Chart option={option} />
        </div>
      ))}
    </div>
  );
}
