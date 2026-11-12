/**
 * Дерево выбора метода как данные (decision-tree-spec v2, раздел 9).
 *
 * Узлы — в decisionTree.json, подписи — в treeLabels.json. Целостность
 * (все методы каталога на карте, пресеты, подписи на трёх языках,
 * совпадение шаблонов с правилами ядра) проверяет tests/test_decision_tree.py.
 */

import tree from "./decisionTree.json";
import labels from "./treeLabels.json";
import type { Lang } from "./i18n";
import type { FiredSignal } from "./pyodide/bridge";

export type NodeKind = "choice" | "test" | "pre" | "leaf" | "post";

export interface TreeNode {
  id: string;
  kind: NodeKind;
  parent: string | null;
  /** Лист: метод каталога ядра. */
  method?: string;
  /** Лист: параметры, которые выбор листа подставляет в форму. */
  preset?: Record<string, string>;
  /** Лист: experimental — GLMM; missing — тупик (в BioStat нет). */
  status?: "experimental" | "missing";
  /** Узел реестра распознавателей, сигналы которого показывает узел. */
  rnode?: string;
  /** Источник правила: «кн. стр. N», «BioStat», «предл.». */
  source?: string;
}

export const ROOT = "R0";
export const NODES = (tree as unknown as { nodes: TreeNode[] }).nodes;

const BY_ID = new Map(NODES.map((n) => [n.id, n]));
const KIDS = new Map<string, TreeNode[]>();
for (const n of NODES) {
  if (n.parent) KIDS.set(n.parent, [...(KIDS.get(n.parent) ?? []), n]);
}

type LabelDict = Record<Lang, {
  nodes: Record<string, string>;
  rec: Record<string, string>;
  /** Оговорки, которые ядро дописывает к сигналу (выброс). */
  notes: Record<string, string>;
  /** Тупики: что это за метод и где его сделать (по id листа). */
  dead: Record<string, string>;
}>;
const LABELS = labels as LabelDict;

export function getNode(id: string): TreeNode | undefined {
  return BY_ID.get(id);
}

export function children(id: string): TreeNode[] {
  return KIDS.get(id) ?? [];
}

/** Путь от корня до узла включительно. */
export function pathTo(id: string): TreeNode[] {
  const path: TreeNode[] = [];
  for (let cur = BY_ID.get(id); cur; cur = cur.parent ? BY_ID.get(cur.parent) : undefined) {
    path.unshift(cur);
  }
  return path;
}

/** Все листья, ведущие к методу (метод бывает в нескольких местах карты). */
export function leavesFor(method: string): TreeNode[] {
  return NODES.filter((n) => n.kind === "leaf" && n.method === method);
}

/** Число листьев-методов в поддереве (для счётчика свёрнутой ветки). */
export function leafCount(id: string): number {
  const kids = children(id);
  if (!kids.length) return BY_ID.get(id)?.kind === "leaf" ? 1 : 0;
  return kids.reduce((sum, k) => sum + leafCount(k.id), 0);
}

export function nodeLabel(id: string, lang: Lang): string {
  return LABELS[lang].nodes[id] ?? LABELS.ru.nodes[id] ?? id;
}

/** Текст тупика: свой у каждого листа; нет — undefined (общий текст интерфейса). */
export function deadEndNote(id: string, lang: Lang): string | undefined {
  return LABELS[lang].dead[id] ?? LABELS.ru.dead[id];
}

/** Сигналы распознавателей, которые относятся к узлу (по rnode). */
export function signalsAt(node: TreeNode, fired: FiredSignal[]): FiredSignal[] {
  return node.rnode ? fired.filter((f) => f.node === node.rnode) : [];
}

/**
 * Форма слова при числе n: две формы (1 / много) или три русские (1 / 2–4 / 5+).
 * Дробное число в русском требует вторую форму («в 2.5 раза»). Та же функция —
 * `plural_form` в ядре (core/validation/recognize.py).
 */
export function pluralForm(n: unknown, forms: string[]): string {
  if (forms.length === 2) return n === 1 ? forms[0] : forms[1];
  if (typeof n !== "number" || !Number.isInteger(n)) return forms[1];
  const k = Math.abs(n);
  if (k % 10 === 1 && k % 100 !== 11) return forms[0];
  const few = k % 10 >= 2 && k % 10 <= 4 && !(k % 100 >= 12 && k % 100 <= 14);
  return few ? forms[1] : forms[2];
}

/** Шаблон: {param} — значение, {param|форма|…} — слово, согласованное с числом. */
export function fillTemplate(template: string, params: Record<string, unknown>): string {
  return template.replace(/\{(\w+)((?:\|[^{}|]+)*)\}/g, (hole, name: string, forms: string) => {
    if (params[name] === undefined) return hole;
    return forms ? pluralForm(params[name], forms.slice(1).split("|")) : String(params[name]);
  });
}

/**
 * Текст сработавшего распознавателя на нужном языке.
 *
 * Шаблон ищется по id (или id#variant), {param} подставляются из
 * параметров; шаблона нет — русский текст, который прислало ядро.
 */
export function recMessage(signal: FiredSignal, lang: Lang): string {
  const params = signal.message.params ?? {};
  const key = params.variant ? `${signal.id}#${params.variant}` : signal.id;
  const base = LABELS[lang].rec[key] ?? LABELS.ru.rec[key];
  if (!base) return signal.message.message;
  // Сигнал, пониженный из-за кандидата в выбросы, несёт оговорку о нём.
  const note = params.outlier_row === undefined ? "" : LABELS[lang].notes.outlier;
  return fillTemplate(base + note, params);
}
