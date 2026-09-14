/** Компоненты интерфейса: загрузка, качество данных, предпросмотр,
 *  выбор метода, вывод результата. */

import { useEffect, useRef, useState } from "react";
import type {
  ColumnProfile,
  ColumnSelection,
  ColumnType,
  MethodParam,
  MethodSuggestion,
  ParsedTable,
  StatResult,
} from "./pyodide/bridge";
import { useT } from "./i18n";
import { useCoreMessage } from "./coreMessages";
import { c, s, typeBadgeStyle } from "./styles";

/* ------------------------------ Загрузка ------------------------------ */

/* --------------------- Непригодный файл --------------------- */

export function UnusableFile({
  table,
  onReset,
}: {
  table: ParsedTable;
  onReset: () => void;
}) {
  const t = useT();
  const tm = useCoreMessage();
  const reason = table.usability.reason ?? "no_columns";

  return (
    <div style={s.panel}>
      <div
        style={{
          padding: "0.85rem 1rem",
          background: c.warnBg,
          color: c.warn,
          borderRadius: 6,
          fontSize: "0.95rem",
          lineHeight: 1.5,
          marginBottom: "1rem",
        }}
      >
        {tm({ code: reason, message: reason })}
      </div>

      {/* Предупреждения детекции — помогают понять, что пошло не так. */}
      {table.warnings.length > 0 && (
        <ul
          style={{
            margin: "0 0 1rem",
            paddingLeft: "1.1rem",
            fontSize: "0.85rem",
            color: c.muted,
          }}
        >
          {table.warnings.map((w, i) => (
            <li key={i}>{tm(w)}</li>
          ))}
        </ul>
      )}

      <button style={s.button} onClick={onReset}>
        {t("reset")}
      </button>
    </div>
  );
}

/* ------------------------- Стартовый экран ------------------------- */

export function WelcomeScreen({
  onFile,
  onExample,
  busy,
}: {
  onFile: (file: File) => void;
  onExample: () => void;
  busy: boolean;
}) {
  const t = useT();
  const points = [
    t("welcome.point.methods"),
    t("welcome.point.formats"),
    t("welcome.point.quality"),
  ];

  return (
    <div>
      <div style={{ maxWidth: 640, marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1.3rem", margin: "0 0 0.6rem", lineHeight: 1.3 }}>
          {t("welcome.title")}
        </h2>
        <p style={{ color: c.text, lineHeight: 1.55, margin: "0 0 0.9rem" }}>
          {t("welcome.lead")}
        </p>
        <ul
          style={{
            margin: "0 0 1rem",
            paddingLeft: "1.1rem",
            color: c.muted,
            fontSize: "0.9rem",
            lineHeight: 1.6,
          }}
        >
          {points.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
        <div
          style={{
            padding: "0.6rem 0.85rem",
            background: c.okBg,
            color: c.ok,
            borderRadius: 6,
            fontSize: "0.85rem",
            lineHeight: 1.5,
          }}
        >
          {t("welcome.privacy")}
        </div>
      </div>

      <FileDrop onFile={onFile} busy={busy} />

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.7rem",
          margin: "1rem 0 0",
        }}
      >
        <span style={{ color: c.muted, fontSize: "0.85rem" }}>
          {t("welcome.or")}
        </span>
        <button
          onClick={onExample}
          disabled={busy}
          style={{
            ...s.button,
            ...(busy ? s.buttonDisabled : {}),
            background: "transparent",
            color: c.accent,
            border: `1px solid ${c.accent}`,
          }}
        >
          {t("welcome.tryExample")}
        </button>
      </div>
    </div>
  );
}

/* ------------------------------ Загрузка ------------------------------ */

export function FileDrop({
  onFile,
  busy,
}: {
  onFile: (file: File) => void;
  busy: boolean;
}) {
  const t = useT();
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  return (
    <div
      style={{ ...s.drop, ...(over ? s.dropActive : {}) }}
      onClick={() => !busy && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const file = e.dataTransfer.files[0];
        if (file && !busy) onFile(file);
      }}
    >
      <input
        ref={input}
        type="file"
        accept=".csv,.txt,.tsv"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
      <div style={{ fontSize: "1rem", marginBottom: "0.3rem" }}>
        {busy ? t("drop.parsing") : t("drop.hint")}
      </div>
      <div style={{ color: c.muted, fontSize: "0.85rem" }}>
        {t("drop.privacy")}
      </div>
    </div>
  );
}

/* --------------------------- Качество данных -------------------------- */

const VERDICT_STYLE: Record<string, { bg: string; fg: string }> = {
  ok: { bg: c.okBg, fg: c.ok },
  warnings: { bg: c.warnBg, fg: c.warn },
  problems: { bg: c.badBg, fg: c.bad },
};

export function QualityPanel({ table }: { table: ParsedTable }) {
  const t = useT();
  const tm = useCoreMessage();
  const q = table.quality;
  const v = VERDICT_STYLE[q.verdict] ?? VERDICT_STYLE.warnings;
  const d = table.dialect;

  return (
    <div style={s.panel}>
      <h2 style={s.h2}>{t("quality.title")}</h2>
      <div
        style={{
          padding: "0.7rem 0.9rem",
          background: v.bg,
          color: v.fg,
          borderRadius: 6,
          fontWeight: 600,
          marginBottom: "0.9rem",
        }}
      >
        {t(`quality.${q.verdict}`)}
      </div>

      <div style={{ ...s.row, gap: "1.5rem", fontSize: "0.85rem" }}>
        <Stat label={t("quality.rows")} value={q.n_rows.toLocaleString()} />
        <Stat label={t("quality.columns")} value={q.n_columns} />
        <Stat label={t("quality.missing")} value={q.n_missing} />
        <Stat label={t("quality.mismatch")} value={q.n_type_mismatch} />
      </div>

      <div
        style={{
          marginTop: "0.9rem",
          fontSize: "0.8rem",
          color: c.muted,
          ...s.mono,
        }}
      >
        {d.encoding} · {t("dialect.delimiter")} {JSON.stringify(d.delimiter)} ·{" "}
        {t("dialect.decimal")} {JSON.stringify(d.decimal)} · {t("dialect.header")}{" "}
        {d.has_header ? t("dialect.header.yes") : t("dialect.header.no")}
      </div>

      {q.top_issues.length > 0 && (
        <ul style={{ margin: "0.9rem 0 0", paddingLeft: "1.1rem", fontSize: "0.85rem" }}>
          {q.top_issues.map((it, i) => (
            <li key={i} style={{ marginBottom: "0.25rem" }}>
              <strong>{it.column}</strong>: {tm(it)}
            </li>
          ))}
        </ul>
      )}

      {table.warnings.map((w, i) => (
        <div key={i} style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: c.warn }}>
          {tm(w)}
        </div>
      ))}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div style={{ color: c.muted, fontSize: "0.78rem" }}>{label}</div>
      <div style={{ fontSize: "1.15rem", fontWeight: 600 }}>{value}</div>
    </div>
  );
}

/* ---------------------------- Предпросмотр ---------------------------- */

export function PreviewPanel({
  table,
  overrides,
  onTypeChange,
  onLoadRows,
  busy,
}: {
  table: ParsedTable;
  overrides: Record<string, ColumnType>;
  onTypeChange: (column: string, type: ColumnType) => void;
  onLoadRows: (start: number, count: number) => Promise<string[][]>;
  busy: boolean;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [pageSize, setPageSize] = useState(15);
  const [rows, setRows] = useState<string[][]>(table.preview_rows.slice(0, 15));
  const [loading, setLoading] = useState(false);
  const byName = new Map(table.profiles.map((p) => [p.name, p]));

  // При смене таблицы (новый файл, смена типа) — сброс к первой странице.
  useEffect(() => {
    setRows(table.preview_rows.slice(0, pageSize));
  }, [table]);

  const hasMore = rows.length < table.n_rows;

  const loadMore = async () => {
    if (loading || !hasMore) return;
    setLoading(true);
    try {
      const next = await onLoadRows(rows.length, pageSize);
      setRows((prev) => [...prev, ...next]);
    } finally {
      setLoading(false);
    }
  };

  // Автоподгрузка при прокрутке к низу контейнера.
  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 80) {
      void loadMore();
    }
  };

  const changePageSize = (size: number) => {
    setPageSize(size);
    // Перезагружаем первую порцию нового размера с начала.
    void (async () => {
      setLoading(true);
      try {
        const first = await onLoadRows(0, size);
        setRows(first);
      } finally {
        setLoading(false);
      }
    })();
  };

  return (
    <div style={s.panel}>
      <h2 style={s.h2}>
        {t("preview.title")}
        <span style={{ fontWeight: 400, color: c.muted, fontSize: "0.85rem" }}>
          {" "}
          · {t("preview.shown")} {rows.length} {t("preview.of")}{" "}
          {table.n_rows.toLocaleString()}
        </span>
      </h2>

      <div style={{ ...s.row, marginBottom: "0.6rem" }}>
        <label style={{ fontSize: "0.82rem", color: c.muted }}>
          {t("preview.pageSize")}:{" "}
          <select
            value={pageSize}
            onChange={(e) => changePageSize(Number(e.target.value))}
            style={{ ...s.select, minWidth: 70, padding: "0.2rem 0.4rem" }}
          >
            {[15, 30, 50, 100, 200].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ maxHeight: 420, overflow: "auto" }} onScroll={onScroll}>
        <table style={s.table}>
          <thead>
            <tr>
              {table.header.map((h) => {
                const p = byName.get(h);
                return (
                  <th
                    key={h}
                    style={{ ...s.th, position: "sticky", top: 0, background: c.panel }}
                  >
                    <div>{h}</div>
                    {p && (
                      <span style={typeBadgeStyle(p.detected_type)}>
                        {t(`type.${p.detected_type}`)}
                      </span>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {table.header.map((_, j) => (
                  <td key={j} style={{ ...s.td, ...s.mono }}>
                    {row[j] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {loading && (
          <div style={{ padding: "0.6rem", color: c.muted, fontSize: "0.82rem" }}>
            {t("preview.loading")}
          </div>
        )}
      </div>

      {hasMore && (
        <button
          onClick={loadMore}
          disabled={loading}
          style={{
            ...s.button,
            background: "transparent",
            color: c.accent,
            padding: "0.5rem 0",
            marginTop: "0.4rem",
          }}
        >
          {t("preview.showMore")} {pageSize}
        </button>
      )}

      <div>
        <button
          onClick={() => setOpen(!open)}
          style={{
            ...s.button,
            background: "transparent",
            color: c.accent,
            padding: "0.5rem 0",
            marginTop: "0.4rem",
          }}
        >
          {open ? t("preview.hideDetails") : t("preview.details")}
        </button>
      </div>

      {open && (
        <ColumnDetails
          profiles={table.profiles}
          overrides={overrides}
          onTypeChange={onTypeChange}
          busy={busy}
        />
      )}
    </div>
  );
}

function ColumnDetails({
  profiles,
  overrides,
  onTypeChange,
  busy,
}: {
  profiles: ColumnProfile[];
  overrides: Record<string, ColumnType>;
  onTypeChange: (column: string, type: ColumnType) => void;
  busy: boolean;
}) {
  const t = useT();
  const tm = useCoreMessage();
  // Проблемные колонки и колонки с подсказкой — наверх.
  const hint = (p: ColumnProfile) =>
    Number(p.likely_categorical || p.likely_id);
  const sorted = [...profiles].sort(
    (a, b) => hint(b) - hint(a) || b.issues.length - a.issues.length,
  );
  const TYPES: ColumnType[] = ["numeric", "categorical", "date", "id"];

  return (
    <div style={{ marginTop: "0.8rem", display: "grid", gap: "0.6rem" }}>
      {sorted.map((p) => {
        const highlight =
          p.issues.length > 0 || p.likely_categorical || p.likely_id;
        const current = overrides[p.name] ?? p.detected_type;
        return (
          <div
            key={p.name}
            style={{
              border: `1px solid ${highlight ? "#fde68a" : c.border}`,
              background: highlight ? c.warnBg : c.panel,
              borderRadius: 6,
              padding: "0.6rem 0.8rem",
            }}
          >
            <div
              style={{
                display: "flex",
                gap: "0.5rem",
                alignItems: "center",
                flexWrap: "wrap",
              }}
            >
              <strong>{p.name}</strong>
              <span style={typeBadgeStyle(p.detected_type)}>
                {t(`type.${p.detected_type}`)}
              </span>
              <span style={{ color: c.muted, fontSize: "0.8rem" }}>
                {t("col.confidence")} {(p.confidence * 100).toFixed(0)}% ·{" "}
                {t("col.missing")} {p.n_missing} (
                {(p.missing_ratio * 100).toFixed(0)}%) · {t("col.unique")}{" "}
                {p.n_unique}
              </span>
              <label
                style={{
                  marginLeft: "auto",
                  fontSize: "0.8rem",
                  color: c.muted,
                  display: "flex",
                  alignItems: "center",
                  gap: "0.35rem",
                }}
              >
                {t("col.type")}:
                <select
                  value={current}
                  disabled={busy}
                  onChange={(e) =>
                    onTypeChange(p.name, e.target.value as ColumnType)
                  }
                  style={{ ...s.select, minWidth: 130, padding: "0.25rem 0.4rem" }}
                >
                  {TYPES.map((tp) => (
                    <option key={tp} value={tp}>
                      {t(`type.${tp}`)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {p.issues.map((issue, i) => (
              <div key={i} style={{ fontSize: "0.85rem", marginTop: "0.35rem" }}>
                {tm(issue)}
              </div>
            ))}
            {p.mismatch_examples.length > 0 && (
              <div
                style={{
                  fontSize: "0.8rem",
                  marginTop: "0.3rem",
                  color: c.muted,
                  ...s.mono,
                }}
              >
                {t("col.examples")}:{" "}
                {p.mismatch_examples.map((e) => JSON.stringify(e)).join(", ")}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* --------------------------- Выбор и запуск --------------------------- */

export function MethodPanel({
  table,
  onRun,
  busy,
}: {
  table: ParsedTable;
  onRun: (
    method: string,
    columns: ColumnSelection,
    kwargs: Record<string, unknown>,
  ) => void;
  busy: boolean;
}) {
  const t = useT();
  // Группируем методы по категориям, сохраняя порядок появления.
  const byGroup = new Map<string, MethodSuggestion[]>();
  for (const sug of table.suggestions) {
    const list = byGroup.get(sug.group) ?? [];
    list.push(sug);
    byGroup.set(sug.group, list);
  }
  const groups = [...byGroup.keys()];

  const [activeGroup, setActiveGroup] = useState(groups[0] ?? "");
  const [method, setMethod] = useState(table.suggestions[0]?.method ?? "");
  const [columns, setColumns] = useState<ColumnSelection>({});
  const [params, setParams] = useState<Record<string, unknown>>({});

  // При смене таблицы (новый файл, смена типа колонки) состав методов
  // меняется — возвращаемся к первой вкладке и первому методу.
  useEffect(() => {
    setActiveGroup(groups[0] ?? "");
    setMethod(table.suggestions[0]?.method ?? "");
    setColumns({});
    const defaults: Record<string, unknown> = {};
    table.suggestions[0]?.params.forEach((p) => (defaults[p.name] = p.default));
    setParams(defaults);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [table]);

  const suggestion = table.suggestions.find((x) => x.method === method);
  const slots = suggestion ? Object.keys(suggestion.needs) : [];
  const ready = slots.every((slot) => columns[slot as keyof ColumnSelection]);

  const selectMethod = (next: string) => {
    setMethod(next);
    setColumns({});
    const spec = table.suggestions.find((x) => x.method === next);
    const defaults: Record<string, unknown> = {};
    spec?.params.forEach((p) => (defaults[p.name] = p.default));
    setParams(defaults);
  };

  const selectGroup = (group: string) => {
    setActiveGroup(group);
    // При переходе на вкладку выбираем первый метод этой группы.
    const first = byGroup.get(group)?.[0];
    if (first) selectMethod(first.method);
  };

  if (table.suggestions.length === 0) {
    return (
      <div style={s.panel}>
        <h2 style={s.h2}>{t("method.title")}</h2>
        <div style={{ color: c.muted }}>{t("method.none")}</div>
      </div>
    );
  }

  const groupMethods = byGroup.get(activeGroup) ?? [];

  return (
    <div style={s.panel}>
      <h2 style={s.h2}>{t("method.title")}</h2>

      {/* Вкладки категорий */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.4rem",
          borderBottom: `1px solid ${c.border}`,
          marginBottom: "0.9rem",
        }}
      >
        {groups.map((g) => {
          const active = g === activeGroup;
          return (
            <button
              key={g}
              onClick={() => selectGroup(g)}
              style={{
                border: "none",
                background: "transparent",
                padding: "0.5rem 0.7rem",
                fontSize: "0.88rem",
                cursor: "pointer",
                color: active ? c.accent : c.muted,
                fontWeight: active ? 600 : 400,
                borderBottom: `2px solid ${active ? c.accent : "transparent"}`,
                marginBottom: -1,
              }}
            >
              {g}
            </button>
          );
        })}
      </div>

      {/* Карточки методов активной категории */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
          gap: "0.5rem",
          marginBottom: "1rem",
        }}
      >
        {groupMethods.map((m) => {
          const selected = m.method === method;
          return (
            <button
              key={m.method}
              onClick={() => selectMethod(m.method)}
              style={{
                textAlign: "left",
                border: `1px solid ${selected ? c.accent : c.border}`,
                background: selected ? "#f0fdfa" : c.panel,
                borderRadius: 6,
                padding: "0.6rem 0.75rem",
                cursor: "pointer",
              }}
            >
              <div style={{ fontWeight: 600, fontSize: "0.9rem", color: c.text }}>
                {m.label}
              </div>
              <div
                style={{ fontSize: "0.78rem", color: c.muted, marginTop: "0.2rem" }}
              >
                {m.description}
              </div>
            </button>
          );
        })}
      </div>

      {/* Колонки и параметры выбранного метода */}
      <div style={s.row}>
        {slots.map((slot) => (
          <div key={slot}>
            <label style={s.label}>{suggestion?.slot_labels[slot] ?? slot}</label>
            <select
              style={s.select}
              value={columns[slot as keyof ColumnSelection] ?? ""}
              onChange={(e) => setColumns({ ...columns, [slot]: e.target.value })}
            >
              <option value="">{t("method.choose")}</option>
              {(suggestion?.needs[slot] ?? []).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        ))}

        {suggestion?.params.map((p) => (
          <ParamField
            key={p.name}
            param={p}
            value={params[p.name]}
            onChange={(v) => setParams({ ...params, [p.name]: v })}
            summaryValues={table.summary_values}
            selectedColumn={columns.value}
          />
        ))}

        <button
          style={{ ...s.button, ...(ready && !busy ? {} : s.buttonDisabled) }}
          disabled={!ready || busy}
          onClick={() => onRun(method, columns, params)}
        >
          {busy ? t("method.running") : t("method.run")}
        </button>
      </div>

      {suggestion?.n_groups === 2 && (
        <div style={{ marginTop: "0.7rem", fontSize: "0.85rem", color: c.muted }}>
          {t("method.twoGroups")}
        </div>
      )}
    </div>
  );
}

/** Поле ввода параметра метода: число, дробь или выбор из вариантов. */
function ParamField({
  param,
  value,
  onChange,
  summaryValues,
  selectedColumn,
}: {
  param: MethodParam;
  value: unknown;
  onChange: (v: unknown) => void;
  summaryValues: Record<string, Record<string, number>>;
  selectedColumn?: string;
}) {
  const t = useT();
  // Источник подстановки: "" = ручной ввод, иначе имя колонки.
  const [source, setSource] = useState<string>("");

  if (param.kind === "select") {
    return (
      <div>
        <label style={s.label}>{param.label}</label>
        <select
          style={s.select}
          value={String(value ?? param.default)}
          onChange={(e) => onChange(e.target.value)}
        >
          {param.options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  const step = param.kind === "float" ? 0.01 : 1;
  // Колонки, у которых есть нужная метрика для подстановки.
  const metric = param.from_summary;
  const sources = metric
    ? Object.keys(summaryValues).filter(
        (col) => summaryValues[col]?.[metric] !== undefined,
      )
    : [];

  const applySource = (col: string) => {
    setSource(col);
    if (col && summaryValues[col]?.[metric] !== undefined) {
      onChange(summaryValues[col][metric]);
    }
  };

  // Подстановка среднего той же колонки в popmean — вырожденный тест.
  const sameColumnWarning =
    param.name === "popmean" &&
    source &&
    selectedColumn &&
    source === selectedColumn;

  return (
    <div>
      <label style={s.label}>{param.label}</label>
      {sources.length > 0 && (
        <select
          style={{ ...s.select, minWidth: 200, marginBottom: "0.3rem" }}
          value={source}
          onChange={(e) => applySource(e.target.value)}
        >
          <option value="">{t("param.manual")}</option>
          {sources.map((col) => (
            <option key={col} value={col}>
              {t(`m.${metric}`)} · {col} ={" "}
              {summaryValues[col][metric].toFixed(3)}
            </option>
          ))}
        </select>
      )}
      <input
        type="number"
        step={step}
        min={param.min ?? undefined}
        max={param.max ?? undefined}
        style={{ ...s.select, minWidth: 120, display: "block" }}
        value={String(value ?? param.default)}
        onChange={(e) => {
          setSource(""); // ручная правка сбрасывает источник
          onChange(Number(e.target.value));
        }}
      />
      {sameColumnWarning && (
        <div
          style={{
            fontSize: "0.75rem",
            color: c.warn,
            marginTop: "0.2rem",
            maxWidth: 220,
          }}
        >
          {t("param.sameColumnWarning")}
        </div>
      )}
    </div>
  );
}

/* ----------------------------- Результаты ----------------------------- */

const SEVERITY_STYLE: Record<string, { bg: string; fg: string }> = {
  info: { bg: "#eff6ff", fg: "#1e40af" },
  warning: { bg: c.warnBg, fg: c.warn },
  error: { bg: c.badBg, fg: c.bad },
};

export function ResultPanel({ result }: { result: StatResult }) {
  const t = useT();
  const tm = useCoreMessage();

  // Сводная статистика — особый рендер: таблица метрик по колонкам.
  if (result.method === "describe_all") {
    return <SummaryTable result={result} />;
  }

  const stats = Object.entries(result.statistics);

  return (
    <div style={s.panel}>
      <h2 style={s.h2}>
        {t("result.title")}
        <span style={{ fontWeight: 400, color: c.muted, fontSize: "0.85rem" }}>
          {" "}
          · {result.method} · n = {result.n}
        </span>
      </h2>

      {result.p_value !== null && (
        <div
          style={{
            padding: "0.7rem 0.9rem",
            background: "#f4f4f5",
            borderRadius: 6,
            marginBottom: "0.9rem",
            fontSize: "1.05rem",
          }}
        >
          <strong>p = {formatNumber(result.p_value)}</strong>
          <span style={{ color: c.muted, fontSize: "0.85rem" }}>
            {" "}
            {result.p_value < 0.05
              ? t("result.significant")
              : t("result.notSignificant")}
          </span>
        </div>
      )}

      {result.estimates.length > 0 && (
        <table style={{ ...s.table, marginBottom: "0.9rem" }}>
          <thead>
            <tr>
              <th style={s.th}>{t("result.estimate")}</th>
              <th style={s.th}>{t("result.value")}</th>
              <th style={s.th}>{t("result.ci")}</th>
              <th style={s.th}>{t("result.se")}</th>
            </tr>
          </thead>
          <tbody>
            {result.estimates.map((e) => (
              <tr key={e.name}>
                <td style={s.td}>{e.name}</td>
                <td style={{ ...s.td, ...s.mono }}>{formatNumber(e.value)}</td>
                <td style={{ ...s.td, ...s.mono }}>
                  {e.ci_lower !== null && e.ci_upper !== null
                    ? `[${formatNumber(e.ci_lower)}, ${formatNumber(e.ci_upper)}]`
                    : "—"}
                </td>
                <td style={{ ...s.td, ...s.mono }}>
                  {e.std_error !== null ? formatNumber(e.std_error) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {stats.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
            gap: "0.5rem",
            fontSize: "0.85rem",
          }}
        >
          {stats.map(([k, v]) => (
            <div key={k}>
              <span style={{ color: c.muted }}>{k}</span>{" "}
              <span style={s.mono}>{formatNumber(v)}</span>
            </div>
          ))}
        </div>
      )}

      {result.warnings.length > 0 && (
        <div style={{ marginTop: "1rem", display: "grid", gap: "0.4rem" }}>
          {result.warnings.map((w, i) => {
            const st = SEVERITY_STYLE[w.severity] ?? SEVERITY_STYLE.warning;
            return (
              <div
                key={i}
                style={{
                  padding: "0.55rem 0.8rem",
                  background: st.bg,
                  color: st.fg,
                  borderRadius: 6,
                  fontSize: "0.85rem",
                }}
              >
                {tm(w)}
              </div>
            );
          })}
        </div>
      )}

      <ExportButtons result={result} />
    </div>
  );
}

/** Порядок метрик в сводной таблице — от центра к форме и выбросам. */
const SUMMARY_ROWS = [
  "n",
  "missing",
  "mean",
  "median",
  "std",
  "variance",
  "sem",
  "min",
  "q1",
  "q3",
  "max",
  "range",
  "iqr",
  "mad",
  "cv_pct",
  "relative_iqr_pct",
  "skewness",
  "kurtosis",
  "zero_count",
  "zero_pct",
  "outliers_iqr",
  "outliers_zscore",
];

/** Сводная описательная статистика: метрики слева, колонки датасета сверху. */
function SummaryTable({ result }: { result: StatResult }) {
  const t = useT();
  const summary = result.parameters.summary as
    | Record<string, Record<string, number>>
    | undefined;
  const columns = (result.parameters.columns as string[] | undefined) ?? [];

  if (!summary || columns.length === 0) {
    return (
      <div style={s.panel}>
        <h2 style={s.h2}>{t("summary.title")}</h2>
        <div style={{ color: c.muted }}>{t("method.none")}</div>
      </div>
    );
  }

  // Показываем только те метрики, что реально есть хотя бы у одной колонки.
  const rows = SUMMARY_ROWS.filter((m) =>
    columns.some((col) => summary[col]?.[m] !== undefined),
  );

  return (
    <div style={s.panel}>
      <h2 style={s.h2}>{t("summary.title")}</h2>
      <div style={{ overflowX: "auto" }}>
        <table style={s.table}>
          <thead>
            <tr>
              <th style={{ ...s.th, position: "sticky", left: 0, background: c.panel }}>
                {""}
              </th>
              {columns.map((col) => (
                <th key={col} style={{ ...s.th, textAlign: "right" }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((metric) => (
              <tr key={metric}>
                <td
                  style={{
                    ...s.td,
                    position: "sticky",
                    left: 0,
                    background: c.panel,
                    fontWeight: 500,
                    color: c.muted,
                  }}
                >
                  {t(`m.${metric}`)}
                </td>
                {columns.map((col) => (
                  <td key={col} style={{ ...s.td, ...s.mono, textAlign: "right" }}>
                    {fmtSummary(summary[col]?.[metric])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ExportButtons result={result} />
    </div>
  );
}

/** Округление до 3 знаков; целые — без дробной части. */
function fmtSummary(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return "—";
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(3);
}

function ExportButtons({ result }: { result: StatResult }) {
  const t = useT();
  const download = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `biostat_${result.method}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={download}
      style={{
        ...s.button,
        background: "transparent",
        color: c.accent,
        padding: "0.5rem 0",
        marginTop: "0.8rem",
      }}
    >
      {t("result.download")}
    </button>
  );
}

/** Формат числа: значимые знаки без длинных хвостов. */
function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Number.isInteger(v)) return v.toString();
  if (Math.abs(v) < 0.0001) return v.toExponential(3);
  return v.toPrecision(6).replace(/\.?0+$/, "");
}
