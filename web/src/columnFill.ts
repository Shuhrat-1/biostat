/**
 * Подстановка колонок в форму метода по клику на лист карты
 * (decision-tree-spec v2, раздел 10, этап 3).
 *
 * Источник — роли, которые карта уже знает: колонки, выбранные в формах
 * других методов, плюс колонка-кандидат в блоки, найденная по имени (B1.1).
 * Слоты разных методов называются по-разному (value / y, group / factor_a),
 * поэтому роли сопоставляются по семействам.
 */

import families from "./roleFamilies.json";
import type { ColumnSelection, FiredSignal, MethodSuggestion } from "./pyodide/bridge";

type Slot = keyof ColumnSelection;

/**
 * Слоты одного смысла: отклик и основной фактор в разных методах. Таблица
 * общая с ядром (core/validation/recognize.py, состав сверяет pytest).
 */
const FAMILIES: Slot[][] = [families.response as Slot[], families.factor as Slot[]];

/**
 * Слоты, роль которых действует, только пока слот есть в форме текущего
 * метода: по наличию x ядро выбирает модель для диагностики (регрессия
 * или модель вариантов).
 */
export const ACTIVE_ONLY = families.activeOnly as Slot[];

/** Слоты, из которых можно взять колонку для данного (сначала — тот же). */
export function equivalents(slot: Slot): Slot[] {
  const family = FAMILIES.find((f) => f.includes(slot));
  return family ? [slot, ...family.filter((s) => s !== slot)] : [slot];
}

/** Роли карты + блок, распознанный по имени колонки (если роли блока нет). */
export function knownRoles(roles: ColumnSelection, fired: FiredSignal[]): ColumnSelection {
  const block = fired.find((f) => f.id === "B1.1")?.message.params?.c;
  if (roles.block || typeof block !== "string") return roles;
  return { ...roles, block };
}

/**
 * Колонки для нового метода. Уже выбранные и подходящие слоты не трогаем;
 * пустые заполняем из ролей, если колонка подходит слоту по типу и ещё не
 * занята другим слотом. Возвращает колонки и список подставленных слотов.
 */
export function fillColumns(
  current: ColumnSelection,
  roles: ColumnSelection,
  spec: MethodSuggestion,
): { columns: ColumnSelection; filled: Slot[] } {
  const columns: ColumnSelection = {};
  for (const slot of Object.keys(spec.needs) as Slot[]) {
    const name = current[slot];
    if (name && spec.needs[slot].includes(name)) columns[slot] = name;
  }
  const filled: Slot[] = [];
  for (const slot of Object.keys(spec.needs) as Slot[]) {
    if (columns[slot]) continue;
    const used = new Set(Object.values(columns));
    const pick = equivalents(slot)
      .map((s) => roles[s])
      .find((name): name is string => Boolean(name) && spec.needs[slot].includes(name!) && !used.has(name!));
    if (pick) {
      columns[slot] = pick;
      filled.push(slot);
    }
  }
  return { columns, filled };
}
