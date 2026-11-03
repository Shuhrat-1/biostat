/**
 * Связь карты и формы метода: что выбрано в форме, какие колонки уже
 * получили роли (для распознавателей), запросы карты к форме.
 */

import { useEffect, useRef, useState } from "react";
import type { MethodRequest } from "./components";
import type { TreeNode } from "./decisionTree";
import type { Context } from "./mapState";
import type { MapSelection } from "./SensoryMap";
import { recognizeStructure, type ColumnSelection, type ParsedTable, type Recognition } from "./pyodide/bridge";

/**
 * Роли копятся по всем методам формы: выбранный однажды блок остаётся
 * известен карте и после перехода к методу без слота «блок».
 * Пустое значение слота снимает роль.
 */
export function mergeRoles(prev: ColumnSelection, columns: ColumnSelection): ColumnSelection {
  const next: Record<string, string> = { ...(prev as Record<string, string>) };
  for (const [slot, name] of Object.entries(columns)) {
    if (name) next[slot] = name;
    else delete next[slot];
  }
  return JSON.stringify(next) === JSON.stringify(prev) ? prev : next;
}

export function useMapSync(table: ParsedTable | null) {
  const [context, setContext] = useState<Context>("unknown");
  const [selection, setSelection] = useState<MapSelection | null>(null);
  const [roles, setRoles] = useState<ColumnSelection>({});
  const [recognition, setRecognition] = useState<Recognition | null>(null);
  const [request, setRequest] = useState<MethodRequest | null>(null);
  const seq = useRef(0);

  // Новый файл (или смена типа колонки) — роли и сигналы прежнего файла не годятся.
  useEffect(() => {
    setRoles({});
    setRecognition(null);
  }, [table]);

  // Распознаватели ядра по текущим ролям; устаревший ответ отбрасываем.
  useEffect(() => {
    if (!table) return;
    let stale = false;
    recognizeStructure(roles)
      .then((r) => !stale && setRecognition(r))
      .catch(() => !stale && setRecognition(null));
    return () => {
      stale = true;
    };
  }, [table, roles]);

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
    setRequest({ method: leaf.method, preset: leaf.preset, seq: seq.current });
  };

  return {
    context, setContext, selection, recognition, request, onSelection, onPickLeaf,
    hasRoles: Object.keys(roles).length > 0,
  };
}
