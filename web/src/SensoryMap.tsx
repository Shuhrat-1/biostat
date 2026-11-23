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
import { deadEndNote, recMessage } from "./decisionTree";
import { useLang, useT, type Lang } from "./i18n";
import {
  awaitsComputation, buildMarks, childStates, contextNote, leafForMethod,
  owners, type Context, type Marks, type NodeState,
} from "./mapState";
import type { FiredSignal, MethodSuggestion, Recognition } from "./pyodide/bridge";
import { c, mapEdge, s } from "./styles";

const VISIBLE_COLUMNS = 3;
const CONTEXTS: Context[] = ["experiment", "observation", "unknown"];
const EDGE: Record<NodeState, string> = {
  strong: mapEdge.strong, hint: mapEdge.hint, conflict: mapEdge.hint, off: mapEdge.off, none: c.border,
};

export interface MapSelection {
  method: string;
  params: Record<string, unknown>;
}

interface Props {
  suggestions: MethodSuggestion[];
  selection: MapSelection | null;
  recognition: Recognition | null;
  /** Идёт расчётная фаза распознавателей (подгонка модели). */
  computing: boolean;
  hasRoles: boolean;
  /** Колонке назначена роль «блок»: листья методов без блока — «без блоков». */
  hasBlock: boolean;
  context: Context;
  onContext: (ctx: Context) => void;
  /** Клик по листу с доступным методом: выбрать метод (и пресет) в форме. */
  onPickLeaf: (leaf: TreeNode) => void;
}

const SOURCE_WORDS: Record<"en" | "pt", [string, string][]> = {
  en: [["кн. стр.", "book p."], ["кн. гл.", "book ch."], ["предл.", "proposed"],
    ["пороги", "thresholds"], ["порог", "threshold"], ["критерий", "criterion"],
    ["гл.", "ch."], ["стр.", "p."], ["ранговый аналог", "rank analogue of"], ["по логике", "following"],
    ["поправки", "corrections"], ["две группы", "two groups"], ["ДИ", "CI"]],
  pt: [["кн. стр.", "livro p."], ["кн. гл.", "livro cap."], ["предл.", "proposta"],
    ["пороги", "limiares"], ["порог", "limiar"], ["критерий", "critério"],
    ["гл.", "cap."], ["стр.", "p."], ["ранговый аналог", "análogo por ordens de"], ["по логике", "segundo"],
    ["поправки", "correções"], ["две группы", "dois grupos"], ["ДИ", "IC"]],
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
  const blockless = useMemo(
    () => (props.hasBlock ? withoutBlocks(suggestions) : new Set<string>()),
    [props.hasBlock, suggestions],
  );
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
            focus={focus} fill={fill} marks={marks} available={available} blockless={blockless}
            registry={recognition?.registry ?? []} computing={props.computing}
            context={props.context} onPick={pick}
          />
          <Details
            focus={focus} suggestions={suggestions} fired={fired} marks={marks} available={available}
            blockless={blockless}
            selected={selection?.method} methodLeaf={methodLeaf?.id}
            registry={recognition?.registry ?? []} computing={props.computing}
            context={props.context} hasRoles={props.hasRoles}
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
      <div role="group" aria-label={t("map.ctx.label").replace(/:\s*$/, "")} style={{ display: "flex", gap: "0.25rem" }}>
        {CONTEXTS.map((ctx) => (
          <button key={ctx} onClick={() => props.onContext(ctx)} style={chip(props.context === ctx)}
            aria-pressed={props.context === ctx}>
            {t(`map.ctx.${ctx}`)}
          </button>
        ))}
      </div>
      <button onClick={props.onToggle} style={{ ...chip(false), color: c.muted }} aria-expanded={props.open}>
        {props.open ? t("map.hide") : t("map.show")}
      </button>
    </div>
  );
}

function chip(active: boolean): CSSProperties {
  return {
    border: `1px solid ${active ? c.accent : c.border}`,
    background: active ? c.selectedBg : c.panel,
    color: active ? c.accent : c.text,
    borderRadius: 6,
    padding: "0.2rem 0.55rem",
    minHeight: 24, // цель не меньше 24 × 24 px (WCAG 2.5.8, аудит 04-11)
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
              border: "none", background: "transparent", padding: "0.1rem 0.3rem", cursor: "pointer",
              minHeight: 24, minWidth: 24,
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
  blockless: Set<string>;
  registry: Recognition["registry"];
  computing: boolean;
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
  const pending = props.computing && awaitsComputation(parent, props.registry)
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
          blockless={Boolean(node.method && props.blockless.has(node.method))}
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
      whiteSpace: "nowrap",
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
  available: boolean; blockless: boolean; flagged: boolean; onPick: (node: TreeNode) => void;
}) {
  const t = useT();
  const lang = useLang();
  const { node, state, onFill } = props;
  const leaf = node.kind === "leaf";
  const dim = state === "off" || node.status === "missing" || !props.available || props.blockless;
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
        background: onFill ? c.selectedBg : c.panel,
        // Узел вне пути — серым текстом, а не прозрачностью: при opacity 0.5 текст
        // был 1.98:1 на белом, бледный лист — 3.36:1 (аудит 04-06).
        color: onFill ? c.accent : dim ? c.muted : c.text,
        fontWeight: onFill ? 600 : 400,
      }}
    >
      <span style={{ flex: "1 1 auto", minWidth: 0 }}>
        {kindMark(node)}{nodeLabel(node.id, lang)}
        {props.flagged && <Badge text="!" tone="warn" />}
        {props.blockless && <Badge text={t("map.noBlocksShort")} tone="muted" />}
      </span>
      {!leaf && <span style={{ color: c.muted, fontSize: "0.75rem" }}>{leafCount(node.id)} ›</span>}
    </button>
  );
}

interface DetailsProps {
  focus: string;
  suggestions: MethodSuggestion[];
  fired: FiredSignal[];
  marks: Marks;
  available: Set<string>;
  blockless: Set<string>;
  selected?: string;
  methodLeaf?: string;
  registry: Recognition["registry"];
  computing: boolean;
  context: Context;
  hasRoles: boolean;
}

/** Узел, под которым лежат все планы с независимыми единицами (линейные модели). */
const DESIGNS_ROOT = "B1";

/**
 * Сигналы узлов на пути к фокусу и в его ветках, плюс диагностика (DIAG):
 * выбросы и серийная корреляция касаются любой модели, где бы ни был фокус.
 * Сигналы о допущениях (B4) видны на любом плане опыта: факториал, split-plot
 * или ANCOVA опираются на те же нормальность и однородность, что и
 * однофакторный анализ, хотя их листья и стоят выше развилки допущений.
 */
function signalsAround(focus: string, fired: FiredSignal[]): FiredSignal[] {
  const path = pathTo(focus);
  const ids = new Set([...path, ...children(focus)].map((n) => n.id));
  const design = path.some((n) => n.id === DESIGNS_ROOT);
  return fired.filter((sig) => sig.node === "DIAG" || (design && sig.node === "B4")
    || owners(sig.node).some((o) => ids.has(o.id)));
}

/** Слоты, в которые идёт колонка блоков: блок, субъект (Фридман, rm_anova), ряд и колонка, страты. */
const BLOCK_SLOTS = ["block", "subject", "row", "column", "stratum_1"];
/** Слоты модели с фактором или x: только такие методы и проигнорировали бы блок. */
const MODEL_SLOTS = ["group", "factor_a", "condition", "x"];
/** Описание по группам выводов не делает: блок ему не нужен. */
const DESCRIPTIVE = new Set(["describe_by_group"]);

/**
 * Методы модели без места для блока: при назначенной роли «блок» их лист на
 * карте помечается «без блоков» — метод проигнорировал бы структуру опыта.
 */
export function withoutBlocks(suggestions: MethodSuggestion[]): Set<string> {
  const has = (m: MethodSuggestion, slots: string[]) => slots.some((k) => k in m.needs);
  return new Set(suggestions
    .filter((m) => has(m, MODEL_SLOTS) && !has(m, BLOCK_SLOTS) && !DESCRIPTIVE.has(m.method))
    .map((m) => m.method));
}

/** Сигналы, которые советуют преобразовать отклик. */
const TRANSFORM_SIGNALS = new Set(["B4.3", "B4.4", "R2.7"]);

/** У метода листа есть свой параметр «Трансформация» — отдельный метод не нужен. */
function hasOwnTransform(method: string | undefined, suggestions: MethodSuggestion[]): boolean {
  if (!method || method === "transformed_means") return false;
  const spec = suggestions.find((m) => m.method === method);
  return Boolean(spec?.params.some((p) => p.name === "transform_kind"));
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
  const pending = props.computing
    && pathTo(props.focus).some((n) => awaitsComputation(n, props.registry));
  const signals = signalsAround(props.focus, props.fired);
  const lines: { text: string; tone: "warn" | "muted" | "bad" }[] = [];
  if (node?.status === "missing") {
    lines.push({ text: deadEndNote(node.id, lang) ?? t("map.missing"), tone: "bad" });
  }
  if (node?.status === "experimental") lines.push({ text: t("map.experimental"), tone: "warn" });
  if (node?.method && !props.available.has(node.method)) lines.push({ text: t("map.unavailable"), tone: "muted" });
  if (node?.method && props.blockless.has(node.method)) lines.push({ text: t("map.noBlocks"), tone: "warn" });
  if (node?.kind === "leaf" && node.id === props.methodLeaf) lines.push({ text: t("map.inForm"), tone: "muted" });
  ctxNotes.forEach((text) => lines.push({ text, tone: "warn" }));
  if (pending) lines.push({ text: t("map.awaits"), tone: "muted" });
  if (signals.some((sig) => TRANSFORM_SIGNALS.has(sig.id))
    && hasOwnTransform(node?.method, props.suggestions)) {
    lines.push({ text: t("map.transformParam"), tone: "warn" });
  }
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
        <div key={sig.id} style={{ display: "flex", flexWrap: "wrap", gap: "0.1rem 0.45rem", alignItems: "baseline", marginBottom: "0.35rem" }}>
          <span
            aria-hidden="true"
            style={{ width: 8, height: 8, borderRadius: 4, flex: "0 0 8px", background: sig.strength === "strong" ? EDGE.strong : EDGE.hint }}
          />
          {/* Сила — словом, не только цветом точки; на узком экране источник уходит под текст. */}
          <span style={{ flex: "1 1 14rem" }}>
            <strong style={{ fontWeight: 600 }}>{t(`map.strength.${sig.strength}`)}:</strong>{" "}
            {recMessage(sig, lang)}
          </span>
          <span style={{ color: c.muted, fontSize: "0.75rem", overflowWrap: "anywhere" }}>
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
      {edge(mapEdge.off, t("map.legend.off"))}
      <span style={{ color: c.accent, fontWeight: 600, background: c.selectedBg, padding: "0 0.3rem", borderRadius: 4 }}>
        {t("map.legend.path")}
      </span>
      <span>{t("map.legend.marks")}</span>
    </div>
  );
}
