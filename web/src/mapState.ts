/**
 * Состояние карты: контур по сигналам распознавателей, путь выбранного
 * метода, заметки контекста (decision-tree-spec v2, разделы 2, 5, 7).
 *
 * Только чистые функции над деревом: компонент карты их вызывает, а сами
 * функции не знают о React и о мосте.
 */

import tree from "./decisionTree.json";
import labels from "./treeLabels.json";
import { NODES, children, getNode, leavesFor, pathTo, type TreeNode } from "./decisionTree";
import type { Lang } from "./i18n";
import type { FiredSignal, RecognizerMeta } from "./pyodide/bridge";

export type Context = "experiment" | "observation" | "unknown";
/** Состояние узла в своей колонке: зелёный, янтарный, конфликт, вне пути. */
export type NodeState = "strong" | "hint" | "conflict" | "off" | "none";

export interface Marks {
  /** Узел → сигналы, которые ведут через него к цели. */
  path: Map<string, FiredSignal[]>;
  /** Узел-владелец → сигналы-заметки (не выбирают ветку, «!»). */
  notes: Map<string, FiredSignal[]>;
}

const TARGETS = (tree as unknown as { targets: Record<string, string[]> }).targets;
const CTX = labels as unknown as Record<Lang, { ctx: Record<string, Partial<Record<Context, string>>> }>;

/** Узлы дерева, которые показывают сигналы узла реестра (B4 → B4a, B4b). */
export function owners(rnode: string): TreeNode[] {
  return NODES.filter((n) => n.rnode === rnode);
}

function isUnder(id: string, ancestor: string): boolean {
  return pathTo(id).some((n) => n.id === ancestor);
}

/** Листья метода внутри поддеревьев владельцев; ближайшие к корню. */
function methodTargets(method: string, own: TreeNode[]): string[] {
  const leaves = leavesFor(method);
  const inside = leaves.filter((l) => own.some((o) => isUnder(l.id, o.id)));
  if (!inside.length) return leaves.map((l) => l.id);
  const depth = Math.min(...inside.map((l) => pathTo(l.id).length));
  return inside.filter((l) => pathTo(l.id).length === depth).map((l) => l.id);
}

/**
 * На какие узлы указывает сигнал. Порядок: явная ветка в параметрах
 * (B3.1), метод в параметрах (A1.5), suggests как узел или метод,
 * таблица targets (с учётом variant). [] — заметка.
 */
export function resolveTargets(signal: FiredSignal): string[] {
  const params = signal.message.params ?? {};
  const own = owners(signal.node);
  if (typeof params.branch === "string" && getNode(params.branch)) return [params.branch];
  if (typeof params.method === "string" && leavesFor(params.method).length) {
    return methodTargets(params.method, own);
  }
  if (getNode(signal.suggests)) return [signal.suggests];
  if (leavesFor(signal.suggests).length) return methodTargets(signal.suggests, own);
  const keyed = params.variant ? TARGETS[`${signal.suggests}#${params.variant}`] : undefined;
  return keyed ?? TARGETS[signal.suggests] ?? [];
}

function push(map: Map<string, FiredSignal[]>, id: string, sig: FiredSignal): void {
  const list = map.get(id) ?? [];
  if (!list.includes(sig)) map.set(id, [...list, sig]);
}

/**
 * Разложить сигналы по узлам. Сигнал отмечает путь от своего узла-
 * владельца (не включая его) вниз до цели: распознаватель структуры
 * опыта не решает за пользователя тип отклика выше по дереву.
 */
export function buildMarks(fired: FiredSignal[]): Marks {
  const marks: Marks = { path: new Map(), notes: new Map() };
  for (const sig of fired) {
    const own = owners(sig.node);
    const targets = resolveTargets(sig);
    if (!targets.length) {
      own.forEach((o) => push(marks.notes, o.id, sig));
      continue;
    }
    for (const target of targets) {
      const path = pathTo(target).map((n) => n.id);
      const from = Math.max(...own.map((o) => path.indexOf(o.id)));
      path.slice(from < 0 ? path.length - 1 : from + 1).forEach((id) => push(marks.path, id, sig));
    }
  }
  return focusNotes(marks);
}

/**
 * Заметка узла реестра с несколькими владельцами (B4 → B4a, B4b) остаётся
 * только у владельца, на которого указали другие сигналы (B3.1 → B4b):
 * «дисперсии неоднородны» при четырёх группах не относится к «Две группы».
 */
function focusNotes(marks: Marks): Marks {
  for (const [id, signals] of marks.notes) {
    const node = getNode(id);
    const siblings = node?.rnode ? owners(node.rnode) : [];
    const chosen = siblings.filter((o) => marks.path.has(o.id));
    if (chosen.length && !chosen.some((o) => o.id === id)) marks.notes.delete(id);
    else marks.notes.set(id, signals);
  }
  return marks;
}

function strengthOf(signals: FiredSignal[] | undefined): "strong" | "hint" | null {
  if (!signals?.length) return null;
  return signals.some((s) => s.strength === "strong") ? "strong" : "hint";
}

/**
 * Состояния детей узла. Один strong — зелёный, остальные без сигналов
 * серые; два и больше strong — противоречие, все янтарные; только hint —
 * янтарные подсказки.
 */
export function childStates(parent: string, marks: Marks): Map<string, NodeState> {
  const kids = children(parent);
  const level = new Map(kids.map((k) => [k.id, strengthOf(marks.path.get(k.id))]));
  const strong = kids.filter((k) => level.get(k.id) === "strong");
  const out = new Map<string, NodeState>();
  for (const k of kids) {
    const s = level.get(k.id);
    if (s === "strong") out.set(k.id, strong.length > 1 ? "conflict" : "strong");
    else if (s === "hint") out.set(k.id, "hint");
    // post-узел — шаг после метода, а не альтернатива ему: не гасим.
    else out.set(k.id, strong.length && k.kind !== "post" ? "off" : "none");
  }
  return out;
}

/** Test-узел с расчётными распознавателями: «⧗», пока идёт расчётная фаза. */
export function awaitsComputation(node: TreeNode, registry: RecognizerMeta[]): boolean {
  return Boolean(node.rnode) && registry.some((r) => r.node === node.rnode && r.cls === "computed");
}

function presetMatches(node: TreeNode, params: Record<string, unknown>): boolean {
  return Object.entries(node.preset ?? {}).every(([k, v]) => params[k] === undefined || params[k] === v);
}

/**
 * Лист карты для метода формы. Метод бывает в нескольких местах: берём
 * лист, по которому пришёл пользователь, иначе совпавший по пресету и
 * отмеченный сигналами, иначе первый.
 */
export function leafForMethod(
  method: string,
  params: Record<string, unknown>,
  marks: Marks,
  preferred?: string,
): TreeNode | undefined {
  const leaves = leavesFor(method).filter((l) => presetMatches(l, params));
  const pick = leaves.find((l) => l.id === preferred)
    ?? leaves.find((l) => strengthOf(marks.path.get(l.id)) === "strong")
    ?? leaves.find((l) => marks.path.has(l.id));
  return pick ?? leaves[0] ?? leavesFor(method)[0];
}

/** Заметка контекста для узла; при «не указано» — просьба указать. */
/**
 * Методы, результат которых не читают как «вариант вызвал различие»: описание,
 * проверка допущений, сравнение с эталоном, планирование. Остальные сравнивают
 * группы или связывают переменные — к ним относится оговорка о причинности.
 */
const NO_CAUSAL_READING = new Set([
  "describe_all", "describe_by_group", "frequency_table", "mean_ci", "chi2_goodness_of_fit",
  "normality_test", "bartlett_test", "anova_residuals",
  "one_proportion", "one_sample_t", "one_sample_wilcoxon",
  "replication_power", "sampling_scheme_power",
]);

/** Результату метода в наблюдательном исследовании нужна оговорка [кн. стр. 3]. */
export function needsCausalityCaveat(method: string, ctx: Context): boolean {
  return ctx === "observation" && !NO_CAUSAL_READING.has(method);
}

export function contextNote(nodeId: string, ctx: Context, lang: Lang): string | undefined {
  const notes = CTX[lang].ctx[nodeId] ?? CTX.ru.ctx[nodeId];
  return notes?.[ctx];
}
