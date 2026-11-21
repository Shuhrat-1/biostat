/**
 * BioStat — главный экран.
 *
 * Полный путь: загрузка файла -> предпросмотр и качество -> выбор метода
 * и колонок -> результат с диагностикой.
 *
 * Все расчёты идут в браузере через Pyodide: данные не покидают машину
 * пользователя.
 */

import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  MethodPanel,
  PreviewPanel,
  QualityPanel,
  ResultPanel,
  UnusableFile,
  WelcomeScreen,
} from "./components";
import { exampleFileBytes } from "./exampleData";
import {
  buildProvenance,
  isoWithOffset,
  sha256Hex,
  type InputFile,
  type Provenance,
} from "./provenance";
import {
  getPlotData,
  getRows,
  parseFile,
  planningSuggestions,
  runMethod,
  runtimeVersions,
  tableSuggestions,
  type ColumnSelection,
  type ColumnType,
  type MethodSuggestion,
  type ParsedTable,
  type PlotBundle,
  type StatResult,
} from "./pyodide/bridge";
import { cancelRun, initRuntime, type LoadProgress } from "./pyodide/runtime";
import { RunStatus, slowHint } from "./RunStatus";
import { I18nContext, LANGUAGES, useLang, useT, type Lang } from "./i18n";
import { c, s } from "./styles";
import { SensoryMap } from "./SensoryMap";
import { useMapSync } from "./useMapSync";
import { explainError } from "./userErrors";
import { MAX_FILE_BYTES, megabytes } from "./limits";

// ECharts тяжёлый — грузим отдельным чанком при первом показе графиков.
const Plots = lazy(() => import("./charts").then((m) => ({ default: m.Plots })));

/** Корень: держит выбранный язык и раздаёт его через контекст. */
export default function App() {
  const [lang, setLang] = useState<Lang>(() => storedLang() ?? "ru");
  // Держим <html lang> в синхроне с UI-языком — для SEO и скринридеров.
  useEffect(() => {
    document.documentElement.lang = lang;
    saveLang(lang);
  }, [lang]);
  return (
    <I18nContext.Provider value={lang}>
      <AppInner lang={lang} setLang={setLang} />
    </I18nContext.Provider>
  );
}

const LANG_KEY = "biostat.lang";

/**
 * Язык прошлого визита (аудит 05-06). localStorage может быть недоступен
 * (приватный режим, запрет cookies) — тогда язык по умолчанию, без ошибки.
 */
function storedLang(): Lang | null {
  try {
    const v = window.localStorage.getItem(LANG_KEY);
    return LANGUAGES.some((l) => l.code === v) ? (v as Lang) : null;
  } catch {
    return null;
  }
}

function saveLang(lang: Lang): void {
  try {
    window.localStorage.setItem(LANG_KEY, lang);
  } catch {
    // хранилище недоступно — язык просто не запомнится
  }
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
  // Имя, размер и SHA-256 исходного файла — для provenance выгрузки (P4).
  const [inputFile, setInputFile] = useState<InputFile | null>(null);
  // Сведения о последнем расчёте: метод, колонки, параметры, время, версии.
  const [provenance, setProvenance] = useState<Provenance | null>(null);
  const [table, setTable] = useState<ParsedTable | null>(null);
  const [typeOverrides, setTypeOverrides] = useState<Record<string, ColumnType>>(
    {},
  );
  const [result, setResult] = useState<StatResult | null>(null);
  const [plots, setPlots] = useState<PlotBundle | null>(null);
  // Режим планирования без данных: список методов, которым файл не нужен.
  const [planning, setPlanning] = useState<MethodSuggestion[] | null>(null);
  const [busy, setBusy] = useState(false);
  // Расчёт пользователя идёт (для плашки «Остановить»), подсказка о долгом методе,
  // перезапуск Worker'а после отмены, рантайм хоть раз загрузился (UI не размонтируем).
  const [running, setRunning] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const [booted, setBooted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Предпросмотр сворачивается после первого расчёта по файлу: дальше он стоит на пути
  // к карте и форме (аудит 05-03). Ручной выбор пользователя держится до нового файла.
  const [previewFolded, setPreviewFolded] = useState(false);
  // Номер таблицы: растёт с новым файлом, примером, сменой типа и сбросом,
  // но не со сменой языка — форма метода тогда сохраняет выбор.
  const [tableKey, setTableKey] = useState(0);
  const nextTable = () => setTableKey((k) => k + 1);
  const previewTouched = useRef(false);
  const unfoldPreview = () => {
    setPreviewFolded(false);
    previewTouched.current = false;
  };
  const map = useMapSync(table);
  // После расчёта результат подводится под верх окна и получает фокус: иначе он
  // остаётся за нижним краем, и о завершении расчёта ничто не говорит (аудит 04-03, 05-05).
  const resultRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!result || !resultRef.current) return;
    resultRef.current.scrollIntoView({ block: "start" });
    resultRef.current.focus({ preventScroll: true });
  }, [result]);

  useEffect(() => {
    let cancelled = false;
    initRuntime((p) => {
      if (cancelled) return;
      setProgress(p);
      if (p.stage === "ready") setBooted(true);
    }).catch((e) => {
      if (!cancelled) {
        setError(explainError(e, t));
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
      setError(explainError(e, t));
      setTable(null);
    } finally {
      setBusy(false);
    }
  };

  // При смене языка подписи методов (они приходят из ядра) перезапрашиваются для
  // уже разобранной таблицы — без повторного разбора: вкладка, метод, колонки и
  // результат остаются (задание patch-132, п. 4).
  useEffect(() => {
    if (table) void relabel();
    if (planning) void openPlanning();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  const suggestionOf = (method: string) =>
    [...(table?.suggestions ?? []), ...(planning ?? [])].find((sg) => sg.method === method);
  const methodLabel = (method: string) => suggestionOf(method)?.label;
  // Все параметры метода для provenance: заданные в форме и значения по умолчанию.
  const effectiveParams = (method: string, kwargs: Record<string, unknown>) => ({
    ...Object.fromEntries((suggestionOf(method)?.params ?? []).map((p) => [p.name, p.default])),
    ...kwargs,
  });

  const relabel = async () => {
    try {
      const suggestions = await tableSuggestions(lang);
      setTable((cur) => (cur ? { ...cur, suggestions } : cur));
    } catch (e) {
      setError(explainError(e, t));
    }
  };

  const handleFile = async (file: File) => {
    nextTable();
    unfoldPreview();
    setPlanning(null);
    setResult(null);
    setPlots(null);
    setTypeOverrides({});
    // Огромный файл не читаем вовсе: разбор упал бы с MemoryError через полминуты (аудит 03-05).
    if (file.size > MAX_FILE_BYTES) {
      setTable(null);
      setFileBytes(null);
      setError(t("error.fileOverLimit")
        .replace("{size}", megabytes(file.size))
        .replace("{limit}", megabytes(MAX_FILE_BYTES)));
      return;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    setFileBytes(bytes);
    setInputFile({ name: file.name, size_bytes: file.size, sha256: await sha256Hex(bytes) });
    await runParse(bytes, {});
  };

  // Пример данных идёт тем же путём, что и загруженный файл.
  const handleExample = async () => {
    nextTable();
    unfoldPreview();
    setPlanning(null);
    setResult(null);
    setPlots(null);
    setTypeOverrides({});
    const bytes = exampleFileBytes(lang);
    setFileBytes(bytes);
    setInputFile({ name: `biostat_example_${lang}.csv`, size_bytes: bytes.length, sha256: await sha256Hex(bytes) });
    await runParse(bytes, {});
  };

  // Смена типа колонки — переразбираем тот же файл с новым переопределением.
  const handleTypeChange = async (column: string, type: ColumnType) => {
    if (!fileBytes) return;
    const next = { ...typeOverrides, [column]: type };
    nextTable();
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
    setRunning(true);
    setHint(slowHint(method, table?.n_rows ?? 0));
    setError(null);
    setPlots(null);
    try {
      const computedAt = isoWithOffset(new Date());
      const res = await runMethod(method, columns, kwargs);
      setProvenance(buildProvenance({
        computed_at: computedAt,
        input_file: planning || !inputFile || !table ? null : {
          ...inputFile, encoding: table.dialect.encoding, delimiter: table.dialect.delimiter,
          decimal: table.dialect.decimal, has_header: table.dialect.has_header,
        },
        runtime: await runtimeVersions().catch(() => ({})),
        method: { id: method, name: methodLabel(method) ?? method },
        parameters: effectiveParams(method, kwargs),
        columns: { ...columns },
        language: lang,
        study_context: map.context,
      }));
      setResult(res);
      if (!previewTouched.current) setPreviewFolded(true);
      // Графики строим после результата и не роняем расчёт, если они не вышли.
      try {
        setPlots(await getPlotData(method, columns));
      } catch {
        setPlots(null);
      }
    } catch (e) {
      if (!isCancel(e)) setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setBusy(false);
      setRunning(false);
    }
  };

  // Отмена: Worker останавливается и запускается заново (таблица восстанавливается).
  const handleCancel = async () => {
    setRestarting(true);
    try {
      await cancelRun();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRestarting(false);
    }
  };

  // Планирование опыта (гл. 19): методы без колонок, файл не нужен.
  const openPlanning = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    setPlots(null);
    try {
      setPlanning(await planningSuggestions(lang));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    nextTable();
    unfoldPreview();
    setPlanning(null);
    setFileBytes(null);
    setTable(null);
    setResult(null);
    setPlots(null);
    setTypeOverrides({});
    setError(null);
  };

  // После первой загрузки интерфейс не прячем: перезапуск после отмены — плашка.
  const ready = booted;

  // Результат, графики и сброс — одинаковы для данных и для планирования.
  const resultsBlock = (
    <>
      <div ref={resultRef} tabIndex={-1} aria-live="polite" style={{ outline: "none", scrollMarginTop: "0.75rem" }}>
        {result && (
          <ResultPanel
            result={result}
            context={map.context}
            methodLabel={methodLabel(result.method)}
            provenance={provenance}
          />
        )}
      </div>
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
  );

  return (
    <div style={s.page}>
      <div style={s.container}>
        {(table || planning) && <SkipLink />}
        <header
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
                aria-pressed={lang === l.code}
                lang={l.code}
                style={{
                  border: `1px solid ${lang === l.code ? c.accent : c.border}`,
                  background: lang === l.code ? c.selectedBg : c.panel,
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
        </header>

        <main>

        {!ready && <LoadingBar progress={progress} />}
        <RunStatus active={running} hint={hint} restarting={restarting} onCancel={handleCancel} />
        {error && <div style={s.error}>{error}</div>}

        {ready && (
          <>
            {!table && !planning && (
              <WelcomeScreen
                onFile={handleFile}
                onExample={handleExample}
                onPlanning={openPlanning}
                busy={busy}
              />
            )}

            {!table && planning && (
              <>
                <div style={{ ...s.panel, color: c.muted, fontSize: "0.9rem" }}>
                  {t("planning.lead")}
                </div>
                <MethodPanel
                  suggestions={planning}
                  summaryValues={{}}
                  onRun={handleRun}
                  busy={busy || restarting}
                  running={running}
                  onCancel={handleCancel}
                />
                {resultsBlock}
              </>
            )}

            {table && table.usability?.ok === false && (
              <UnusableFile table={table} onReset={reset} />
            )}

            {table && table.usability?.ok !== false && (
              <>
                <QualityPanel table={table} />
                <PreviewPanel
                  table={table}
                  overrides={typeOverrides}
                  onTypeChange={handleTypeChange}
                  onLoadRows={getRows}
                  busy={busy}
                  folded={previewFolded}
                  onToggleFold={() => {
                    previewTouched.current = true;
                    setPreviewFolded(!previewFolded);
                  }}
                />
                <SensoryMap
                  suggestions={table.suggestions}
                  selection={map.selection}
                  recognition={map.recognition}
                  computing={map.computing}
                  hasRoles={map.hasRoles}
                  context={map.context}
                  onContext={map.setContext}
                  onPickLeaf={map.onPickLeaf}
                />
                <MethodPanel
                  suggestions={table.suggestions}
                  summaryValues={table.summary_values}
                  onRun={handleRun}
                  busy={busy || restarting}
                  running={running}
                  onCancel={handleCancel}
                  blockHint={map.blockHint}
                  tableKey={tableKey}
                  request={map.request}
                  onSelection={map.onSelection}
                />
                {resultsBlock}
              </>
            )}
          </>
        )}
        </main>
        <SiteFooter />
      </div>
    </div>
  );
}

/** Адрес статичной страницы на языке интерфейса: английская — без префикса. */
function pagePath(slug: string, lang: Lang): string {
  return lang === "en" ? `/${slug}/` : `/${lang}/${slug}/`;
}

/**
 * Подвал: «Конфиденциальность · Условия · Методы и проверка · Лицензия» —
 * ссылки добавляются по мере готовности страниц (задание P1–P6).
 */
function SiteFooter() {
  const t = useT();
  const lang = useLang();
  return (
    <footer style={{ borderTop: `1px solid ${c.border}`, marginTop: "2rem", padding: "0.8rem 0",
      fontSize: "0.85rem", color: c.muted }}>
      <a href={pagePath("privacy", lang)} style={{ color: c.accent }}>{t("footer.privacy")}</a>
      {" · "}
      <a href={pagePath("terms", lang)} style={{ color: c.accent }}>{t("footer.terms")}</a>
      {" · "}
      <a href={pagePath("methods", lang)} style={{ color: c.accent }}>{t("footer.methods")}</a>
      {" · "}
      <a href="/LICENSE.txt" style={{ color: c.accent }}>{t("footer.license")}</a>
    </footer>
  );
}

/**
 * Ссылка-пропуск «К расчёту» (аудит 04-04, WCAG 2.4.1): первая по Tab, видна
 * только в фокусе, ведёт к колонкам и кнопке «Рассчитать».
 */
function SkipLink() {
  const t = useT();
  const [shown, setShown] = useState(false);
  return (
    <a
      href="#calc"
      onFocus={() => setShown(true)}
      onBlur={() => setShown(false)}
      onClick={(e) => {
        const target = document.getElementById("calc");
        if (!target) return;
        e.preventDefault();
        target.focus();
        target.scrollIntoView({ block: "start" });
      }}
      style={shown ? {
        position: "absolute", top: 8, left: 8, zIndex: 10, padding: "0.5rem 0.8rem",
        background: c.panel, color: c.accent, border: `2px solid ${c.accent}`, borderRadius: 6,
        fontWeight: 600,
      } : {
        position: "absolute", width: 1, height: 1, overflow: "hidden", clipPath: "inset(50%)",
        whiteSpace: "nowrap",
      }}
    >
      {t("skip.calc")}
    </a>
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

/** Расчёт остановлен пользователем — не ошибка. */
function isCancel(e: unknown): boolean {
  return e instanceof Error && e.name === "CancelledError";
}
