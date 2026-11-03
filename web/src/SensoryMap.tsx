/**
 * Сенсорная карта выбора метода (decision-tree-spec v2, раздел 7).
 *
 * Дерево показано колонками: каждая колонка — развилка (вопрос) и её
 * ветки; видны три последние развилки пути, остальные — в цепочке сверху.
 * Заливка — путь к методу, выбранному в форме; цвет левой кромки — что
 * подсказывают данные (распознаватели ядра); «!» — заметки и контекст.
 */

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { children, getNode, leafCount, nodeLabel, pathTo, ROOT, type TreeNode } from "./decisionTree";
import { recMessage } from "./decisionTree";
import { useLang, useT, type Lang } from "./i18n";
import {
  awaitsComputation, buildMarks, childStates, contextNote, leafForMethod,
  owners, type Context, type Marks, type NodeState,
} from "./mapState";
import type { FiredSignal, MethodSuggestion, Recognition } from "./pyodide/bridge";
import { c, s } from "./styles";

const VISIBLE_COLUMNS = 3;
const CONTEXTS: Context[] = ["experiment", "observation", "unknown"];
const EDGE: Record<NodeState, string> = {
  strong: "#16a34a", hint: "#d97706", conflict: "#d97706", off: c.border, none: c.border,
};

export interface MapSelection {
  method: string;
  params: Record<string, unknown>;
}

interface Props {
  suggestions: MethodSuggestion[];
  selection: MapSelection | null;
  recognition: Recognition | null;
  hasRoles: boolean;
  context: Context;
  onContext: (ctx: Context) => void;
  /** Клик по листу с доступным методом: выбрать метод (и пресет) в форме. */
  onPickLeaf: (leaf: TreeNode) => void;
}

const SOURCE_WORDS: Record<"en" | "pt", [string, string][]> = {
  en: [["кн. стр.", "book p."], ["кн. гл.", "book ch."], ["предл.", "proposed"],
    ["пороги", "thresholds"], ["порог", "threshold"], ["критерий", "criterion"]],
  pt: [["кн. стр.", "livro p."], ["кн. гл.", "livro cap."], ["предл.", "proposta"],
    ["пороги", "limiares"], ["порог", "limiar"], ["критерий", "critério"]],
};

/** «кн. стр. 58» → «book p. 58» и т. п.; источники в данных — по-русски. */
export function formatSource(src: string, lang: Lang): string {
  if (lang === "ru") return src;
  return SOURCE_WORDS[lang].reduce((acc, [ru, tr]) => acc.split(ru).join(tr), src);
}

export function SensoryMap(props: Props) {
  const t = useT();
  const { suggestions, selection, recognition } = props;
  const [open, setOpen] = useState(true);
  const [focus, setFocus] = useState(ROOT);
  const [preferred, setPreferred] = useState<string | undefined>();

  const fired = useMemo(() => recognition?.fired ?? [], [recognition]);
  const marks = useMemo(() => buildMarks(fired), [fired]);
  const available = useMemo(() => new Set(suggestions.map((m) => m.method)), [suggestions]);
  const methodLeaf = selection
    ? leafForMethod(selection.method, selection.params, marks, preferred)
    : undefined;
  const fill = new Set(methodLeaf ? pathTo(methodLeaf.id).map((n) => n.id) : []);

  // Метод выбран в форме — карта переходит к его листу.
  const presetKey = JSON.stringify(methodLeaf?.preset ?? {});
  useEffect(() => {
    if (methodLeaf) setFocus(methodLeaf.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.method, presetKey]);

  const pick = (node: TreeNode) => {
    setFocus(node.id);
    if (node.kind !== "leaf") return;
    setPreferred(node.id);
    if (node.method && available.has(node.method)) props.onPickLeaf(node);
  };

  return (
    <div style={s.panel}>
      <MapHeader open={open} onToggle={() => setOpen(!open)} context={props.context} onContext={props.onContext} />
      {open && (
        <>
          <Breadcrumbs focus={focus} fill={fill} onPick={(id) => setFocus(id)} />
          <Columns
            focus={focus} fill={fill} marks={marks} available={available}
            registry={recognition?.registry ?? []} context={props.context} onPick={pick}
          />
          <Details
            focus={focus} fired={fired} marks={marks} available={available}
            selected={selection?.method} methodLeaf={methodLeaf?.id}
            registry={recognition?.registry ?? []} context={props.context} hasRoles={props.hasRoles}
          />
          <Legend />
        </>
      )}
      {!open && <div style={{ color: c.muted, fontSize: "0.85rem" }}>{t("map.title")}</div>}
    </div>
  );
}

function MapHeader(props: {
  open: boolean; onToggle: () => void; context: Context; onContext: (ctx: Context) => void;
}) {
  const t = useT();
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.6rem", marginBottom: "0.7rem" }}>
      <h2 style={{ ...s.h2, margin: 0, flex: "1 1 auto" }}>{t("map.title")}</h2>
      <span style={{ fontSize: "0.82rem", color: c.muted }}>{t("map.ctx.label")}</span>
      <div style={{ display: "flex", gap: "0.25rem" }}>
        {CONTEXTS.map((ctx) => (
          <button key={ctx} onClick={() => props.onContext(ctx)} style={chip(props.context === ctx)}>
            {t(`map.ctx.${ctx}`)}
          </button>
        ))}
      </div>
      <button onClick={props.onToggle} style={{ ...chip(false), color: c.muted }}>
        {props.open ? t("map.hide") : t("map.show")}
      </button>
    </div>
  );
}

function chip(active: boolean): CSSProperties {
  return {
    border: `1px solid ${active ? c.accent : c.border}`,
    background: active ? "#f0fdfa" : c.panel,
    color: active ? c.accent : c.text,
    borderRadius: 6,
    padding: "0.2rem 0.55rem",
    fontSize: "0.8rem",
    cursor: "pointer",
    fontWeight: active ? 600 : 400,
  };
}

/** Весь путь к фокусу — компактной цепочкой; клик возвращает к развилке. */
function Breadcrumbs(props: { focus: string; fill: Set<string>; onPick: (id: string) => void }) {
  const lang = useLang();
  const path = pathTo(props.focus);
  return (
    <nav style={{ display: "flex", flexWrap: "wrap", gap: "0.2rem", fontSize: "0.78rem", marginBottom: "0.6rem" }}>
      {path.map((n, i) => (
        <span key={n.id} style={{ display: "inline-flex", alignItems: "center", gap: "0.2rem" }}>
          {i > 0 && <span style={{ color: c.muted }}>›</span>}
          <button
            onClick={() => props.onPick(n.id)}
            style={{
              border: "none", background: "transparent", padding: "0.1rem 0.2rem", cursor: "pointer",
              color: props.fill.has(n.id) ? c.accent : c.muted,
              fontWeight: n.id === props.focus ? 600 : 400,
            }}
          >
            {nodeLabel(n.id, lang)}
          </button>
        </span>
      ))}
    </nav>
  );
}

interface ColumnsProps {
  focus: string;
  fill: Set<string>;
  marks: Marks;
  available: Set<string>;
  registry: Recognition["registry"];
  context: Context;
  onPick: (node: TreeNode) => void;
}

/** Развилки пути к фокусу: у листа — его соседи, у узла — его ветки. */
function Columns(props: ColumnsProps) {
  const path = pathTo(props.focus);
  const parents = path.filter((n) => n.kind !== "leaf");
  const shown = parents.slice(-VISIBLE_COLUMNS);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.6rem" }}>
      {shown.map((parent) => {
        const next = path[path.findIndex((n) => n.id === parent.id) + 1]?.id;
        return <Column key={parent.id} parent={parent} next={next} {...props} />;
      })}
    </div>
  );
}

function Column(props: ColumnsProps & { parent: TreeNode; next?: string }) {
  const lang = useLang();
  const { parent, marks } = props;
  const states = childStates(parent.id, marks);
  const pending = awaitsComputation(parent, props.registry)
    && ![...states.values()].some((st) => st === "strong");
  const notes = marks.notes.get(parent.id)?.length ?? 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem", minWidth: 0 }}>
      <div style={{ fontSize: "0.8rem", fontWeight: 600, color: c.muted, minHeight: "2.1em" }}>
        {nodeLabel(parent.id, lang)}
        {notes > 0 && <Badge text="!" tone="warn" />}
        {pending && <Badge text="⧗" tone="muted" />}
      </div>
      {children(parent.id).map((node) => (
        <NodeButton
          key={node.id} node={node} state={states.get(node.id) ?? "none"}
          onFill={props.fill.has(node.id)} expanded={node.id === props.next}
          available={!node.method || props.available.has(node.method)}
          flagged={(marks.notes.get(node.id)?.length ?? 0) > 0 || Boolean(contextNote(node.id, props.context, lang))}
          onPick={props.onPick}
        />
      ))}
    </div>
  );
}

function Badge({ text, tone }: { text: string; tone: "warn" | "muted" }) {
  return (
    <span style={{
      marginLeft: "0.35rem", fontSize: "0.72rem", fontWeight: 700, padding: "0 0.3rem", borderRadius: 4,
      color: tone === "warn" ? c.warn : c.muted, background: tone === "warn" ? c.warnBg : c.bg,
    }}>
      {text}
    </span>
  );
}

function kindMark(node: TreeNode): string {
  if (node.status === "missing") return "✗ ";
  if (node.status === "experimental") return "⚗ ";
  if (node.kind === "pre") return "⇢ ";
  if (node.kind === "post") return "↳ ";
  return "";
}

function NodeButton(props: {
  node: TreeNode; state: NodeState; onFill: boolean; expanded: boolean;
  available: boolean; flagged: boolean; onPick: (node: TreeNode) => void;
}) {
  const lang = useLang();
  const { node, state, onFill } = props;
  const leaf = node.kind === "leaf";
  const dim = state === "off" || node.status === "missing" || !props.available;
  return (
    <button
      onClick={() => props.onPick(node)}
      title={node.source ? formatSource(node.source, lang) : undefined}
      style={{
        display: "flex", alignItems: "center", gap: "0.3rem", textAlign: "left", cursor: "pointer",
        borderTop: `1px solid ${props.expanded ? c.accent : c.border}`,
        borderRight: `1px solid ${props.expanded ? c.accent : c.border}`,
        borderBottom: `1px solid ${props.expanded ? c.accent : c.border}`,
        borderLeft: `4px solid ${EDGE[state]}`,
        borderRadius: 6, padding: "0.35rem 0.5rem", fontSize: "0.84rem",
        background: onFill ? "#f0fdfa" : c.panel,
        color: onFill ? c.accent : c.text,
        fontWeight: onFill ? 600 : 400,
        opacity: dim && !onFill ? 0.5 : 1,
      }}
    >
      <span style={{ flex: "1 1 auto", minWidth: 0 }}>
        {kindMark(node)}{nodeLabel(node.id, lang)}
        {props.flagged && <Badge text="!" tone="warn" />}
      </span>
      {!leaf && <span style={{ color: c.muted, fontSize: "0.75rem" }}>{leafCount(node.id)} ›</span>}
    </button>
  );
}

interface DetailsProps {
  focus: string;
  fired: FiredSignal[];
  marks: Marks;
  available: Set<string>;
  selected?: string;
  methodLeaf?: string;
  registry: Recognition["registry"];
  context: Context;
  hasRoles: boolean;
}

/** Сигналы узлов на пути к фокусу и в его ветках (без повторов). */
function signalsAround(focus: string, fired: FiredSignal[]): FiredSignal[] {
  const ids = new Set([...pathTo(focus), ...children(focus)].map((n) => n.id));
  return fired.filter((sig) => owners(sig.node).some((o) => ids.has(o.id)));
}

/** Пояснения под колонками: лист, заметки контекста, сигналы данных. */
function Details(props: DetailsProps) {
  const t = useT();
  const lang = useLang();
  const node = getNode(props.focus);
  const visible = [...pathTo(props.focus), ...children(props.focus)];
  const ctxNotes = visible
    .map((n) => contextNote(n.id, props.context, lang))
    .filter((x): x is string => Boolean(x));
  const pending = pathTo(props.focus).some((n) => awaitsComputation(n, props.registry));
  const signals = signalsAround(props.focus, props.fired);
  const lines: { text: string; tone: "warn" | "muted" | "bad" }[] = [];
  if (node?.status === "missing") lines.push({ text: t("map.missing"), tone: "bad" });
  if (node?.status === "experimental") lines.push({ text: t("map.experimental"), tone: "warn" });
  if (node?.method && !props.available.has(node.method)) lines.push({ text: t("map.unavailable"), tone: "muted" });
  if (node?.kind === "leaf" && node.id === props.methodLeaf) lines.push({ text: t("map.inForm"), tone: "muted" });
  ctxNotes.forEach((text) => lines.push({ text, tone: "warn" }));
  if (pending) lines.push({ text: t("map.awaits"), tone: "muted" });
  if (!props.hasRoles) lines.push({ text: t("map.noRoles"), tone: "muted" });

  return (
    <div style={{ marginTop: "0.8rem", fontSize: "0.84rem" }}>
      {lines.map((l, i) => (
        <div key={i} style={{ color: l.tone === "bad" ? c.bad : l.tone === "warn" ? c.warn : c.muted, marginBottom: "0.3rem" }}>
          {l.text}
        </div>
      ))}
      {signals.length > 0 && <SignalList signals={signals} />}
    </div>
  );
}

function SignalList({ signals }: { signals: FiredSignal[] }) {
  const t = useT();
  const lang = useLang();
  return (
    <div style={{ marginTop: "0.4rem" }}>
      <div style={{ fontWeight: 600, marginBottom: "0.3rem" }}>{t("map.signals")}</div>
      {signals.map((sig) => (
        <div key={sig.id} style={{ display: "flex", gap: "0.45rem", alignItems: "baseline", marginBottom: "0.25rem" }}>
          <span
            title={t(`map.strength.${sig.strength}`)}
            style={{ width: 8, height: 8, borderRadius: 4, flex: "0 0 8px", background: sig.strength === "strong" ? EDGE.strong : EDGE.hint }}
          />
          <span style={{ flex: "1 1 auto" }}>{recMessage(sig, lang)}</span>
          <span style={{ color: c.muted, fontSize: "0.75rem", whiteSpace: "nowrap" }}>
            {sig.id}{sig.source && sig.source !== "—" ? ` · ${formatSource(sig.source, lang)}` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

function Legend() {
  const t = useT();
  const edge = (color: string, label: string) => (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
      <span style={{ width: 4, height: 12, background: color }} />
      {label}
    </span>
  );
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem 0.9rem", marginTop: "0.8rem", fontSize: "0.74rem", color: c.muted }}>
      {edge(EDGE.strong, t("map.legend.strong"))}
      {edge(EDGE.hint, t("map.legend.hint"))}
      <span style={{ opacity: 0.5 }}>{t("map.legend.off")}</span>
      <span style={{ color: c.accent, fontWeight: 600, background: "#f0fdfa", padding: "0 0.3rem", borderRadius: 4 }}>
        {t("map.legend.path")}
      </span>
      <span>{t("map.legend.marks")}</span>
    </div>
  );
}
