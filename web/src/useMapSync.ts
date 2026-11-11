/**
 * Связь карты и формы метода: что выбрано в форме, какие колонки уже
 * получили роли (для распознавателей), запросы карты к форме.
 */

import { useEffect, useRef, useState } from "react";
import { ACTIVE_ONLY, equivalents, knownRoles } from "./columnFill";
import type { MethodRequest } from "./components";
import type { TreeNode } from "./decisionTree";
import type { Context } from "./mapState";
import type { MapSelection } from "./SensoryMap";
import {
  recognizeStructure, type ColumnSelection, type FiredSignal, type ParsedTable, type Recognition,
} from "./pyodide/bridge";
import { CancelledError } from "./pyodide/runtime";

/** Сбои распознавателей — в консоль: пользователю не показываем, контур не меняем. */
function warnRuleErrors(phase: string, result: Recognition): void {
  for (const { id, error } of result.errors ?? []) {
    console.warn(`[map] rule ${id} failed (${phase}): ${error}`);
  }
}

/** Сбой всего вызова распознавателей; остановка расчёта пользователем — не сбой. */
function warnCallFailure(phase: string, err: unknown): void {
  if (!(err instanceof CancelledError)) console.warn(`[map] recognition failed (${phase}):`, err);
}

/**
 * Роли копятся по всем методам формы: выбранный однажды блок остаётся
 * известен карте и после перехода к методу без слота «блок».
 * Пустое значение слота снимает роль. Слоты одного смысла вытесняют друг
 * друга (новый отклик y снимает прежний value), а x действует, только
 * пока он выбран в форме текущего метода.
 */
export function mergeRoles(prev: ColumnSelection, columns: ColumnSelection): ColumnSelection {
  const next: Record<string, string> = { ...(prev as Record<string, string>) };
  for (const slot of ACTIVE_ONLY) {
    if (!columns[slot]) delete next[slot];
  }
  for (const [slot, name] of Object.entries(columns)) {
    if (!name) {
      delete next[slot];
      continue;
    }
    equivalents(slot as keyof ColumnSelection).slice(1).forEach((twin) => delete next[twin]);
    next[slot] = name;
  }
  return JSON.stringify(next) === JSON.stringify(prev) ? prev : next;
}

/** Двухфазный вызов распознавателей; ошибка расчётной фазы не гасит структурную. */
async function runRecognition(
  roles: ColumnSelection,
  method: string | null,
  cache: Map<string, FiredSignal[]>,
  isStale: () => boolean,
  setRecognition: (r: Recognition | null) => void,
  setComputing: (v: boolean) => void,
): Promise<void> {
  let structural: Recognition;
  try {
    structural = await recognizeStructure(roles, "structural", method);
  } catch (err) {
    warnCallFailure("structural", err);
    if (!isStale()) setRecognition(null);
    return;
  }
  warnRuleErrors("structural", structural);
  if (isStale()) return;
  const key = JSON.stringify([method, Object.entries(roles).sort()]);
  const cached = cache.get(key);
  setRecognition({ ...structural, fired: [...structural.fired, ...(cached ?? [])] });
  if (cached || !Object.keys(roles).length) {
    setComputing(false);
    return;
  }
  setComputing(true);
  try {
    const result = await recognizeStructure(roles, "computed", method);
    warnRuleErrors("computed", result);
    const computed = result.fired;
    cache.set(key, computed);
    if (!isStale()) setRecognition({ ...structural, fired: [...structural.fired, ...computed] });
  } catch (err) {
    // Расчётные — подсказка, не обязательная часть карты.
    warnCallFailure("computed", err);
  } finally {
    if (!isStale()) setComputing(false);
  }
}

export function useMapSync(table: ParsedTable | null) {
  const [context, setContext] = useState<Context>("unknown");
  const [selection, setSelection] = useState<MapSelection | null>(null);
  const [roles, setRoles] = useState<ColumnSelection>({});
  const [recognition, setRecognition] = useState<Recognition | null>(null);
  const [computing, setComputing] = useState(false);
  // Расчётные сигналы по набору ролей: повторный выбор тех же колонок не считает заново.
  const computedCache = useRef(new Map<string, FiredSignal[]>());
  const [request, setRequest] = useState<MethodRequest | null>(null);
  const seq = useRef(0);

  // Новый файл (или смена типа колонки) — роли и сигналы прежнего файла не годятся.
  useEffect(() => {
    setRoles({});
    setRecognition(null);
    computedCache.current.clear();
  }, [table]);

  // Метод, открытый в форме: правила ветки GLM считаются, только когда человек в ней.
  const method = selection?.method ?? null;

  // Сначала структурные (сразу), затем расчётные (подгонка модели, кэш);
  // устаревший ответ отбрасываем.
  useEffect(() => {
    if (!table) return;
    let stale = false;
    void runRecognition(
      roles, method, computedCache.current, () => stale, setRecognition, setComputing,
    );
    return () => {
      stale = true;
    };
  }, [table, roles, method]);

  const onSelection = (
    method: string,
    columns: ColumnSelection,
    params: Record<string, unknown>,
  ) => {
    setSelection({ method, params });
    setRoles((prev) => mergeRoles(prev, columns));
  };

  const onPickLeaf = (leaf: TreeNode) => {
    if (!leaf.method) return;
    seq.current += 1;
    const known = knownRoles(roles, recognition?.fired ?? []);
    setRequest({ method: leaf.method, preset: leaf.preset, roles: known, seq: seq.current });
  };

  return {
    context, setContext, selection, recognition, computing, request, onSelection, onPickLeaf,
    hasRoles: Object.keys(roles).length > 0,
  };
}
