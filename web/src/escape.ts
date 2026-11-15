/** Обезвреживание текста из файла пользователя перед вставкой в HTML и в CSV (аудит 03-01, 03-04). */

const HTML_ENTITIES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/**
 * Текст для подсказки графика. ECharts вставляет строку из `tooltip.formatter`
 * как HTML, поэтому метка группы вида `<img onerror=…>` исполнялась бы как код.
 */
export function escapeHtml(text: unknown): string {
  return String(text ?? "").replace(/[&<>"']/g, (ch) => HTML_ENTITIES[ch]);
}

/** Начало ячейки, которое Excel читает как формулу (OWASP, CSV Injection). */
const FORMULA_START = /^[=+\-@\t\r]/;

/**
 * Ячейка CSV. Строка, начинающаяся с `=`, `+`, `-`, `@`, табуляции или возврата
 * каретки, получает впереди апостроф — Excel покажет её текстом, а не выполнит.
 * Числа (в том числе отрицательные) не трогаются.
 */
export function csvCell(value: string | number, delimiter = ";"): string {
  if (typeof value === "number") return String(value);
  const text = FORMULA_START.test(value) ? `'${value}` : value;
  return text.includes('"') || text.includes(delimiter) || /[,\n\r]/.test(text)
    ? `"${text.replace(/"/g, '""')}"`
    : text;
}
