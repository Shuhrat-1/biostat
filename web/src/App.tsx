/**
 * BioStat — главный экран.
 *
 * Полный путь: загрузка файла -> предпросмотр и качество -> выбор метода
 * и колонок -> результат с диагностикой.
 *
 * Все расчёты идут в браузере через Pyodide: данные не покидают машину
 * пользователя.
 */

import { lazy, Suspense, useEffect, useState } from "react";
import {
  FileDrop,
  MethodPanel,
  PreviewPanel,
  QualityPanel,
  ResultPanel,
} from "./components";
import {
  getPlotData,
  getRows,
  parseFile,
  runMethod,
  type ColumnSelection,
  type ColumnType,
  type ParsedTable,
  type PlotBundle,
  type StatResult,
} from "./pyodide/bridge";
import { initRuntime, type LoadProgress } from "./pyodide/runtime";
import { I18nContext, LANGUAGES, useT, type Lang } from "./i18n";
import { c, s } from "./styles";

// ECharts тяжёлый — грузим отдельным чанком при первом показе графиков.
const Plots = lazy(() => import("./charts").then((m) => ({ default: m.Plots })));

/** Корень: держит выбранный язык и раздаёт его через контекст. */
export default function App() {
  const [lang, setLang] = useState<Lang>("ru");
  return (
    <I18nContext.Provider value={lang}>
      <AppInner lang={lang} setLang={setLang} />
    </I18nContext.Provider>
  );
}

function AppInner({
  lang,
  setLang,
}: {
  lang: Lang;
  setLang: (l: Lang) => void;
}) {
  const t = useT();
  const [progress, setProgress] = useState<LoadProgress>({
    stage: "idle",
    message: "",
  });
  const [fileBytes, setFileBytes] = useState<Uint8Array | null>(null);
  const [table, setTable] = useState<ParsedTable | null>(null);
  const [typeOverrides, setTypeOverrides] = useState<Record<string, ColumnType>>(
    {},
  );
  const [result, setResult] = useState<StatResult | null>(null);
  const [plots, setPlots] = useState<PlotBundle | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    initRuntime((p) => !cancelled && setProgress(p)).catch((e) => {
      if (!cancelled) {
        setError(e instanceof Error ? e.message : String(e));
        setProgress({ stage: "error", message: "" });
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const runParse = async (
    bytes: Uint8Array,
    overrides: Record<string, ColumnType>,
  ) => {
    setBusy(true);
    setError(null);
    try {
      setTable(await parseFile(bytes, { type_overrides: overrides, lang }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setTable(null);
    } finally {
      setBusy(false);
    }
  };

  // При смене языка переразбираем текущий файл, чтобы метки методов
  // (они приходят из ядра) обновились на новый язык.
  useEffect(() => {
    if (fileBytes) void runParse(fileBytes, typeOverrides);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  const handleFile = async (file: File) => {
    setResult(null);
    setPlots(null);
    setTypeOverrides({});
    const bytes = new Uint8Array(await file.arrayBuffer());
    setFileBytes(bytes);
    await runParse(bytes, {});
  };

  // Смена типа колонки — переразбираем тот же файл с новым переопределением.
  const handleTypeChange = async (column: string, type: ColumnType) => {
    if (!fileBytes) return;
    const next = { ...typeOverrides, [column]: type };
    setTypeOverrides(next);
    setResult(null);
    setPlots(null);
    await runParse(fileBytes, next);
  };

  const handleRun = async (
    method: string,
    columns: ColumnSelection,
    kwargs: Record<string, unknown>,
  ) => {
    setBusy(true);
    setError(null);
    setPlots(null);
    try {
      setResult(await runMethod(method, columns, kwargs));
      // Графики строим после результата и не роняем расчёт, если они не вышли.
      try {
        setPlots(await getPlotData(method, columns));
      } catch {
        setPlots(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setFileBytes(null);
    setTable(null);
    setResult(null);
    setPlots(null);
    setTypeOverrides({});
    setError(null);
  };

  const ready = progress.stage === "ready";

  return (
    <div style={s.page}>
      <div style={s.container}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
          }}
        >
          <div>
            <h1 style={s.h1}>BioStat</h1>
            <p style={s.sub}>{t("app.subtitle")}</p>
          </div>
          <div style={{ display: "flex", gap: "0.3rem" }}>
            {LANGUAGES.map((l) => (
              <button
                key={l.code}
                onClick={() => setLang(l.code)}
                style={{
                  border: `1px solid ${lang === l.code ? c.accent : c.border}`,
                  background: lang === l.code ? "#f0fdfa" : c.panel,
                  color: lang === l.code ? c.accent : c.muted,
                  borderRadius: 6,
                  padding: "0.25rem 0.55rem",
                  fontSize: "0.8rem",
                  cursor: "pointer",
                  fontWeight: lang === l.code ? 600 : 400,
                }}
              >
                {l.code.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {!ready && <LoadingBar progress={progress} />}
        {error && <div style={s.error}>{error}</div>}

        {ready && (
          <>
            {!table && <FileDrop onFile={handleFile} busy={busy} />}

            {table && (
              <>
                <QualityPanel table={table} />
                <PreviewPanel
                  table={table}
                  overrides={typeOverrides}
                  onTypeChange={handleTypeChange}
                  onLoadRows={getRows}
                  busy={busy}
                />
                <MethodPanel table={table} onRun={handleRun} busy={busy} />
                {result && <ResultPanel result={result} />}
                {plots && (
                  <div style={s.panel}>
                    <h2 style={s.h2}>{t("plots.title")}</h2>
                    <Suspense
                      fallback={
                        <div style={{ color: c.muted }}>{t("plots.loading")}</div>
                      }
                    >
                      <Plots bundle={plots} />
                    </Suspense>
                  </div>
                )}

                <button
                  onClick={reset}
                  style={{
                    ...s.button,
                    background: "transparent",
                    color: c.muted,
                    padding: "0.4rem 0",
                  }}
                >
                  {t("reset")}
                </button>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function LoadingBar({ progress }: { progress: LoadProgress }) {
  const t = useT();
  const steps = ["loading-runtime", "loading-packages", "loading-core"];
  const index = steps.indexOf(progress.stage);
  const percent = index < 0 ? 5 : ((index + 1) / (steps.length + 1)) * 100;

  // Сообщение по стадии, а не по тексту из ядра — так оно локализуется.
  const stageKey: Record<string, string> = {
    idle: "load.runtime",
    "loading-runtime": "load.runtime",
    "loading-packages": "load.packages",
    "loading-core": "load.core",
    ready: "load.ready",
    error: "load.error",
  };
  const message = stageKey[progress.stage]
    ? t(stageKey[progress.stage])
    : progress.message;

  return (
    <div style={s.panel}>
      <div style={{ marginBottom: "0.6rem" }}>{message}</div>
      <div
        style={{
          height: 6,
          background: c.border,
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${percent}%`,
            height: "100%",
            background: c.accent,
            transition: "width 0.4s ease",
          }}
        />
      </div>
      <div style={{ marginTop: "0.6rem", fontSize: "0.82rem", color: c.muted }}>
        {t("load.note")}
      </div>
    </div>
  );
}
